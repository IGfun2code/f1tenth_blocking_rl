import os
import sys
import csv
import json
import math
import argparse
from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np

# Simulator uses classic gym registration
import gym as legacy_gym

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

def _find_repo_root(start: str) -> str:
    cur = os.path.abspath(start)
    for _ in range(5):
        if os.path.isdir(os.path.join(cur, "src")):
            return cur
        cur = os.path.dirname(cur)
    return os.getcwd()


REPO_ROOT = _find_repo_root(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from f110_gym.envs.base_classes import Integrator
from src.controllers.pure_pursuit import DynamicPurePursuit, nearest_point_on_trajectory
from src.planners.nominal_planner import load_waypoints_csv
from src.planners.blocking_planner import BlockingPlanner
from src.planners.rrt_star_overtake import RRTStarOvertakePlanner


DEFAULT_REWARD_WEIGHTS = {
    # positive
    "progress": 4.0,
    "lead_hold": 0.02,
    "defense_under_threat": 0.05,
    "completion_bonus": 10.0,
    # negative
    "block_usage": 0.03,
    "lateral_error": 0.20,
    "passed_penalty": 3.0,
    "ego_track_collision": 12.0,
    "ego_vehicle_collision": 1.0,
    "time_penalty": 0.002,
}


@dataclass
class RLActionParams:
    offset: float
    hold_time: float
    return_rate: float


# ------------------------------------------------------------
# Helper functions adapted from your blocking demo
# ------------------------------------------------------------
def resolve_path(path_str: str, base_dir: str) -> str:
    if os.path.isabs(path_str):
        return path_str
    return os.path.normpath(os.path.join(base_dir, path_str))


def relative_pose_in_ego_frame(ego_x, ego_y, ego_yaw, opp_x, opp_y):
    dx = opp_x - ego_x
    dy = opp_y - ego_y
    rel_x = np.cos(ego_yaw) * dx + np.sin(ego_yaw) * dy
    rel_y = -np.sin(ego_yaw) * dx + np.cos(ego_yaw) * dy
    return rel_x, rel_y


def cumulative_arclength(points_xy: np.ndarray) -> np.ndarray:
    ds = np.linalg.norm(np.diff(points_xy, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(ds)))


def wrap_angle(theta: float) -> float:
    return np.arctan2(np.sin(theta), np.cos(theta))


def tangent_and_normal(points_xy: np.ndarray, idx: int):
    n = len(points_xy)
    p_prev = points_xy[(idx - 1) % n]
    p_next = points_xy[(idx + 1) % n]

    tangent = p_next - p_prev
    norm = np.linalg.norm(tangent)
    if norm < 1e-8:
        tangent = np.array([1.0, 0.0], dtype=np.float32)
    else:
        tangent = tangent / norm

    normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
    yaw = float(np.arctan2(tangent[1], tangent[0]))
    return tangent, normal, yaw


def index_from_arclength_backward(s_vals: np.ndarray, start_idx: int, backward_dist: float) -> int:
    total_len = float(s_vals[-1])
    start_s = float(s_vals[start_idx])
    target_s = start_s - backward_dist
    if target_s < 0.0:
        target_s += total_len
    idx = int(np.argmin(np.abs(s_vals - target_s)))
    return idx


def sample_spawn_poses_from_nominal(
    nominal_waypoints: np.ndarray,
    rng: np.random.Generator,
    gap_min: float,
    gap_max: float,
    ego_lat_rand: float,
    opp_lat_rand: float,
    yaw_rand: float,
):
    path_xy = nominal_waypoints[:, :2]
    n = len(path_xy)
    if n < 5:
        raise ValueError("Need at least 5 nominal waypoints for random spawn sampling.")

    s_vals = cumulative_arclength(path_xy)

    ego_idx = int(rng.integers(0, n))
    gap = float(rng.uniform(gap_min, gap_max))
    opp_idx = index_from_arclength_backward(s_vals, ego_idx, gap)

    _, ego_normal, ego_yaw = tangent_and_normal(path_xy, ego_idx)
    _, opp_normal, opp_yaw = tangent_and_normal(path_xy, opp_idx)

    ego_lat = float(rng.uniform(-ego_lat_rand, ego_lat_rand))
    opp_lat_extra = float(rng.uniform(-opp_lat_rand, opp_lat_rand))

    ego_xy = path_xy[ego_idx] + ego_lat * ego_normal
    opp_xy = path_xy[opp_idx] + opp_lat_extra * opp_normal

    ego_yaw = wrap_angle(ego_yaw + float(rng.uniform(-yaw_rand, yaw_rand)))
    opp_yaw = wrap_angle(opp_yaw + float(rng.uniform(-yaw_rand, yaw_rand)))

    ego_pose = np.array([ego_xy[0], ego_xy[1], ego_yaw], dtype=np.float32)
    opp_pose = np.array([opp_xy[0], opp_xy[1], opp_yaw], dtype=np.float32)

    meta = {
        "ego_idx": ego_idx,
        "opp_idx": opp_idx,
        "gap": gap,
        "ego_lat": ego_lat,
        "opp_lat": opp_lat_extra,
    }
    return ego_pose, opp_pose, meta


class BlockDefenseEnv(gym.Env):
    metadata = {"render_modes": ["human", "human_fast", None]}

    def __init__(
        self,
        config_path: str,
        seed: int = 123,
        render_mode: Optional[str] = None,
        max_steps: int = 1000,
        timestep: float = 0.01,
        ego_max_speed: float = 4.0,
        opp_max_speed: float = 4.8,
        max_lookahead: float = 1.0,
        min_lookahead_scale: float = 0.20,
        min_speed_scale: float = 1.0,
        lookahead_turn_gain: float = 0.9,
        speed_turn_gain: float = 1.2,
        scan_angle_min: float = -2.35,
        scan_angle_max: float = 2.35,
        max_scan_range: Optional[float] = None,
        opp_rrt_replan_every: int = 5,
        track_width: float = 1.0,
        random_spawn: bool = True,
        spawn_gap_min: float = 1.0,
        spawn_gap_max: float = 2.5,
        ego_lateral_offset_rand: float = 0.05,
        opp_lateral_offset_rand: float = 0.20,
        spawn_yaw_rand: float = 0.05,
        spawn_max_tries: int = 20,
        history_len: int = 6,
        probe_window: int = 12,
        max_block_offset: float = 0.45,
        min_hold_time: float = 0.10,
        max_hold_time: float = 1.50,
        min_return_rate: float = 0.5,
        max_return_rate: float = 8.0,
        engage_offset_threshold: float = 0.03,
        threat_distance: float = 2.5,
        reward_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.timestep = timestep
        self.ego_max_speed = ego_max_speed
        self.opp_max_speed = opp_max_speed
        self.scan_angle_min = scan_angle_min
        self.scan_angle_max = scan_angle_max
        self.max_scan_range = max_scan_range
        self.opp_rrt_replan_every = max(1, opp_rrt_replan_every)
        self.track_width = track_width
        self.random_spawn = random_spawn
        self.spawn_gap_min = spawn_gap_min
        self.spawn_gap_max = spawn_gap_max
        self.ego_lateral_offset_rand = ego_lateral_offset_rand
        self.opp_lateral_offset_rand = opp_lateral_offset_rand
        self.spawn_yaw_rand = spawn_yaw_rand
        self.spawn_max_tries = spawn_max_tries
        self.history_len = history_len
        self.probe_window = probe_window
        self.max_block_offset = max_block_offset
        self.min_hold_time = min_hold_time
        self.max_hold_time = max_hold_time
        self.min_return_rate = min_return_rate
        self.max_return_rate = max_return_rate
        self.engage_offset_threshold = engage_offset_threshold  # deprecated: kept only for backward compatibility
        self.threat_distance = threat_distance
        self.block_metric_offset_epsilon = 1e-3
        self.reward_weights = DEFAULT_REWARD_WEIGHTS.copy()
        if reward_weights is not None:
            self.reward_weights.update(reward_weights)

        self.rng = np.random.default_rng(seed)

        config_path = resolve_path(config_path, REPO_ROOT)
        with open(config_path) as f:
            conf_dict = json.loads(json.dumps(__import__('yaml').load(f, Loader=__import__('yaml').FullLoader)))
        # above keeps simple serializable conversion
        self.conf = argparse.Namespace(**conf_dict)
        config_dir = os.path.dirname(config_path)
        self.conf.map_path = resolve_path(self.conf.map_path, config_dir)
        self.conf.wpt_path = resolve_path(self.conf.wpt_path, config_dir)

        self.nominal_waypoints = load_waypoints_csv(
            self.conf.wpt_path,
            delimiter=self.conf.wpt_delim,
            skiprows=self.conf.wpt_rowskip,
            x_idx=self.conf.wpt_xind,
            y_idx=self.conf.wpt_yind,
            v_idx=self.conf.wpt_vind,
        )
        self.nominal_xy = self.nominal_waypoints[:, :2]
        self.path_s = cumulative_arclength(self.nominal_xy)
        self.total_len = float(self.path_s[-1])

        self.ego_controller = DynamicPurePursuit(
            wheelbase=0.15875 + 0.17145,
            max_steering_angle=0.4189,
            max_speed=self.ego_max_speed,
            max_look_ahead=max_lookahead,
            min_look_ahead_scale=min_lookahead_scale,
            min_speed_scale=min_speed_scale,
            look_ahead_turn_gain=lookahead_turn_gain,
            speed_turn_gain=speed_turn_gain,
        )
        self.opp_controller = DynamicPurePursuit(
            wheelbase=0.15875 + 0.17145,
            max_steering_angle=0.4189,
            max_speed=self.opp_max_speed,
            max_look_ahead=max_lookahead,
            min_look_ahead_scale=min_lookahead_scale,
            min_speed_scale=min_speed_scale,
            look_ahead_turn_gain=lookahead_turn_gain,
            speed_turn_gain=speed_turn_gain,
        )
        self.ego_controller.set_waypoints(self.nominal_waypoints)

        self.blocking_planner = BlockingPlanner(
            nominal_waypoints=self.nominal_waypoints,
            horizon_points=120,
            ramp_distance=0.7,
            max_offset=max_block_offset,
        )
        self.opp_overtake = RRTStarOvertakePlanner(
            nominal_waypoints=self.nominal_waypoints,
            goal_lookahead=3.0,
            max_path_speed=self.opp_max_speed,
        )

        self.sim = legacy_gym.make(
            "f110_gym:f110-v0",
            map=self.conf.map_path,
            map_ext=self.conf.map_ext,
            num_agents=2,
            timestep=self.timestep,
            integrator=Integrator.RK4,
        )
        if self.render_mode is not None:
            self.sim.add_render_callback(self._render_callback)

        obs_dim = history_len * 4 + 12
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._episode_step = 0
        self._last_obs = None
        self._last_rel = None
        self._history = deque(maxlen=self.history_len)
        self._probe_signs = deque(maxlen=self.probe_window)
        self._cached_opp_rrt_path = self.nominal_waypoints.copy()
        self._cached_opp_rrt_info = {"used_rrt": False, "goal_global": None, "max_deviation": 0.0}
        self._block_hold_steps_remaining = 0
        self._block_side = 1.0
        self._active_params = RLActionParams(0.0, self.min_hold_time, self.min_return_rate)
        self._active_block = False
        self._prev_progress_s = 0.0
        self._prev_gap_s = 0.0
        self._metrics = {}
        self._last_reward_breakdown = {}

    # ---------- rendering ----------
    def _render_callback(self, env_renderer):
        e = env_renderer
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 800
        e.right = right + 800
        e.top = top + 800
        e.bottom = bottom - 800

        self.blocking_planner.render_debug(e, draw_nominal=True)
        self.opp_overtake.render_debug(
            e,
            nominal_rgb=(50, 50, 90),
            path_rgb=(255, 170, 0),
            goal_rgb=(255, 255, 0),
        )

    # ---------- gymnasium api ----------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if self.random_spawn:
            spawn_ok = False
            spawn_meta = None
            for _ in range(self.spawn_max_tries):
                ego_pose, opp_pose, spawn_meta = sample_spawn_poses_from_nominal(
                    nominal_waypoints=self.nominal_waypoints,
                    rng=self.rng,
                    gap_min=self.spawn_gap_min,
                    gap_max=self.spawn_gap_max,
                    ego_lat_rand=self.ego_lateral_offset_rand,
                    opp_lat_rand=self.opp_lateral_offset_rand,
                    yaw_rand=self.spawn_yaw_rand,
                )
                if np.linalg.norm(ego_pose[:2] - opp_pose[:2]) > 0.6:
                    spawn_ok = True
                    break
            if not spawn_ok:
                raise RuntimeError("Failed to find a valid randomized spawn.")
        else:
            ego_pose = np.array([self.conf.sx, self.conf.sy, self.conf.stheta], dtype=np.float32)
            forward = np.array([np.cos(self.conf.stheta), np.sin(self.conf.stheta)], dtype=np.float32)
            left = np.array([-np.sin(self.conf.stheta), np.cos(self.conf.stheta)], dtype=np.float32)
            opp_lat = float(self.rng.uniform(-self.opp_lateral_offset_rand, self.opp_lateral_offset_rand))
            opp_xy = ego_pose[:2] - 2.0 * forward + opp_lat * left
            opp_pose = np.array([opp_xy[0], opp_xy[1], self.conf.stheta], dtype=np.float32)
            spawn_meta = {"gap": 2.0, "ego_lat": 0.0, "opp_lat": opp_lat}

        poses = np.vstack([ego_pose, opp_pose])
        obs, _, _, _ = self.sim.reset(poses)
        self._last_obs = obs
        self._episode_step = 0
        self._cached_opp_rrt_path = self.nominal_waypoints.copy()
        self._cached_opp_rrt_info = {"used_rrt": False, "goal_global": None, "max_deviation": 0.0}
        self._block_hold_steps_remaining = 0
        self._block_side = 1.0
        self._active_params = RLActionParams(0.0, self.min_hold_time, self.min_return_rate)
        self._active_block = False
        self._history.clear()
        self._probe_signs.clear()

        ego_s = self._project_progress(obs["poses_x"][0], obs["poses_y"][0])
        opp_s = self._project_progress(obs["poses_x"][1], obs["poses_y"][1])
        self._prev_progress_s = ego_s
        self._prev_gap_s = self._signed_gap_s(ego_s, opp_s)
        self._last_rel = None
        self._prev_action_vec = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self._metrics = {
            "spawn_gap": float(spawn_meta["gap"]),
            "spawn_ego_lat": float(spawn_meta["ego_lat"]),
            "spawn_opp_lat": float(spawn_meta["opp_lat"]),
            "block_hold_steps": 0,
            "threat_steps": 0,
            "lead_hold_steps": 0,
            "vehicle_contact_collisions": 0,
            "ego_track_collisions": 0,
            "opp_track_collisions": 0,
            "passed_events": 0,
            "replans": 0,
            "opp_rrt_steps": 0,
            "progress_total": 0.0,
            "avg_offset_cmd": 0.0,
            "avg_hold_cmd": 0.0,
            "avg_return_cmd": 0.0,
            "decision_steps": 0,
            "probe_switches": 0,
            "episode_steps": 0,
        }

        obs_vec = self._build_observation(obs)
        if self.render_mode is not None:
            self.sim.render(mode=self.render_mode)
        return obs_vec, {}

    def step(self, action):
        obs = self._last_obs
        ego_x, ego_y, ego_yaw = obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0]
        opp_x, opp_y, opp_yaw = obs["poses_x"][1], obs["poses_y"][1], obs["poses_theta"][1]

        rel_x, rel_y = relative_pose_in_ego_frame(ego_x, ego_y, ego_yaw, opp_x, opp_y)
        rel_dist = float(np.hypot(rel_x, rel_y))
        ego_speed_actual = float(obs["linear_vels_x"][0])
        opp_speed_actual = float(obs["linear_vels_x"][1])
        closing_speed = opp_speed_actual - ego_speed_actual
        threat = bool(rel_x < 0.0 and rel_dist < self.threat_distance and closing_speed > 0.0)

        rel_vx, rel_vy = self._relative_velocity(rel_x, rel_y)
        self._history.append(np.array([rel_x, rel_y, rel_vx, rel_vy], dtype=np.float32))
        self._update_probe_history(rel_y, threat)

        # follower path: RRT* baseline
        replan_now = (self._episode_step == 0 or self._episode_step % self.opp_rrt_replan_every == 0)
        if replan_now:
            scan_n = len(obs["scans"][1])
            angle_increment = (self.scan_angle_max - self.scan_angle_min) / max(scan_n - 1, 1)
            new_rrt_path, new_rrt_info = self.opp_overtake.plan(
                opp_x,
                opp_y,
                opp_yaw,
                obs["scans"][1],
                self.scan_angle_min,
                angle_increment,
                max_scan_range=self.max_scan_range,
            )
            if new_rrt_path is not None and len(new_rrt_path) >= 2:
                self._cached_opp_rrt_path = new_rrt_path
                self._cached_opp_rrt_info = new_rrt_info
            else:
                self._cached_opp_rrt_info = {**self._cached_opp_rrt_info, "used_rrt": False}
            self._metrics["replans"] += 1

        self.opp_controller.set_waypoints(self._cached_opp_rrt_path)
        if self._cached_opp_rrt_info.get("used_rrt", False):
            self._metrics["opp_rrt_steps"] += 1

        # RL chooses block parameters continuously.
        # There is no engage/disengage threshold in control anymore:
        # offset ~= 0 means "do not block", larger offset means stronger block.
        params = self._decode_action(action)
        self._metrics["avg_offset_cmd"] += params.offset
        self._metrics["avg_hold_cmd"] += params.hold_time
        self._metrics["avg_return_cmd"] += params.return_rate
        self._metrics["decision_steps"] += 1

        self._block_side = 1.0 if self._probe_side_score() >= 0.0 else -1.0

        block_path = self.blocking_planner.build_blocking_path(
            ego_x=ego_x,
            ego_y=ego_y,
            side_sign=self._block_side,
            offset_magnitude=params.offset,
            hold_time=params.hold_time,
            return_rate=params.return_rate,
            current_speed=max(ego_speed_actual, 0.5),
        )
        self.ego_controller.set_waypoints(block_path)

        offset_norm = params.offset / max(self.max_block_offset, 1e-6)
        self._active_block = bool(offset_norm > self.block_metric_offset_epsilon)
        if self._active_block:
            # Metric only: approximates how long the policy commanded a meaningful block.
            self._metrics["block_hold_steps"] += 1

        ego_speed_cmd, ego_steer, ego_info = self.ego_controller.plan(ego_x, ego_y, ego_yaw)
        opp_speed_cmd, opp_steer, _ = self.opp_controller.plan(opp_x, opp_y, opp_yaw)

        sim_action = np.array([
            [ego_steer, ego_speed_cmd],
            [opp_steer, opp_speed_cmd],
        ], dtype=np.float32)

        next_obs, _, done, _ = self.sim.step(sim_action)
        self._last_obs = next_obs
        self._episode_step += 1
        self._metrics["episode_steps"] = self._episode_step

        reward, reward_breakdown, event_info = self._compute_reward(obs, next_obs, params)
        self._last_reward_breakdown = reward_breakdown

        terminated = bool(done or event_info["ego_collision"] or event_info["vehicle_contact"])
        truncated = bool(self._episode_step >= self.max_steps)

        next_obs_vec = self._build_observation(next_obs)
        info = {
            "reward_breakdown": reward_breakdown,
            "rl_action": params.__dict__.copy(),
            "block_active": self._active_block,
            "block_side": self._block_side,
            "threat": threat,
            "opp_rrt_used": bool(self._cached_opp_rrt_info.get("used_rrt", False)),
        }

        if self.render_mode is not None:
            self.sim.render(mode=self.render_mode)

        if terminated or truncated:
            info["episode_metrics"] = self._finalize_episode_metrics()
            info["episode"] = {
                "r": float(sum(reward_breakdown.values())),
                "l": int(self._episode_step),
            }

        return next_obs_vec, float(reward), terminated, truncated, info

    # ---------- observation ----------
    def _relative_velocity(self, rel_x: float, rel_y: float) -> Tuple[float, float]:
        if self._last_rel is None:
            self._last_rel = (rel_x, rel_y)
            return 0.0, 0.0
        vx = (rel_x - self._last_rel[0]) / self.timestep
        vy = (rel_y - self._last_rel[1]) / self.timestep
        self._last_rel = (rel_x, rel_y)
        return float(vx), float(vy)

    def _update_probe_history(self, rel_y: float, threat: bool):
        if threat:
            sign = 1 if rel_y > 0.05 else (-1 if rel_y < -0.05 else 0)
            self._probe_signs.append(sign)
        else:
            self._probe_signs.append(0)

    def _probe_side_score(self) -> float:
        if len(self._probe_signs) == 0:
            return 0.0
        vals = np.array(self._probe_signs, dtype=np.float32)
        return float(np.mean(vals))

    def _probe_switch_rate(self) -> float:
        if len(self._probe_signs) < 2:
            return 0.0
        signs = [s for s in self._probe_signs if s != 0]
        if len(signs) < 2:
            return 0.0
        switches = sum(1 for a, b in zip(signs[:-1], signs[1:]) if a != b)
        self._metrics["probe_switches"] = switches
        return switches / max(1, len(signs) - 1)

    def _project_progress(self, x: float, y: float) -> float:
        point = np.array([x, y], dtype=np.float32)
        _, _, t, idx = nearest_point_on_trajectory(point, self.nominal_xy)
        idx = int(idx)
        next_idx = min(idx + 1, len(self.nominal_xy) - 1)
        seg_len = np.linalg.norm(self.nominal_xy[next_idx] - self.nominal_xy[idx])
        return float(self.path_s[idx] + t * seg_len)

    def _signed_gap_s(self, ego_s: float, opp_s: float) -> float:
        gap = ego_s - opp_s
        if gap > self.total_len / 2.0:
            gap -= self.total_len
        elif gap < -self.total_len / 2.0:
            gap += self.total_len
        return float(gap)

    def _progress_delta(self, curr_s: float, prev_s: float) -> float:
        d = curr_s - prev_s
        if d < -self.total_len / 2.0:
            d += self.total_len
        elif d > self.total_len / 2.0:
            d -= self.total_len
        return float(d)

    def _signed_lateral_position(self, x: float, y: float) -> float:
        point = np.array([x, y], dtype=np.float32)
        nearest_pt, _, _, idx = nearest_point_on_trajectory(point, self.nominal_xy)
        tangent, normal, _ = tangent_and_normal(self.nominal_xy, int(idx))
        return float(np.dot(point - nearest_pt, normal))

    def _curvature_features(self, x: float, y: float) -> Tuple[float, float, float]:
        point = np.array([x, y], dtype=np.float32)
        _, _, _, idx = nearest_point_on_trajectory(point, self.nominal_xy)
        idx = int(idx)
        horizon_points = 40
        pts = np.array([self.nominal_xy[(idx + i) % len(self.nominal_xy)] for i in range(horizon_points)], dtype=np.float32)
        curvs = []
        dists = cumulative_arclength(pts)
        for i in range(1, len(pts) - 1):
            a = pts[i] - pts[i - 1]
            b = pts[i + 1] - pts[i]
            la = np.linalg.norm(a)
            lb = np.linalg.norm(b)
            if la < 1e-6 or lb < 1e-6:
                curvs.append(0.0)
                continue
            ha = math.atan2(a[1], a[0])
            hb = math.atan2(b[1], b[0])
            dtheta = wrap_angle(hb - ha)
            curvs.append(abs(dtheta) / max((la + lb) * 0.5, 1e-6))
        curvs = np.array(curvs if len(curvs) > 0 else [0.0], dtype=np.float32)
        # features over about next 1m and 2m
        mean_1m = float(np.mean(curvs[: min(len(curvs), 10)]))
        max_2m = float(np.max(curvs[: min(len(curvs), 20)]))
        turn_density = float(np.sum(curvs[: min(len(curvs), 20)]))
        return mean_1m, max_2m, turn_density

    def _build_observation(self, obs) -> np.ndarray:
        ego_x, ego_y, ego_yaw = obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0]
        opp_x, opp_y = obs["poses_x"][1], obs["poses_y"][1]
        rel_x, rel_y = relative_pose_in_ego_frame(ego_x, ego_y, ego_yaw, opp_x, opp_y)
        rel_vx, rel_vy = self._relative_velocity(rel_x, rel_y)
        if len(self._history) == 0:
            for _ in range(self.history_len):
                self._history.append(np.array([rel_x, rel_y, rel_vx, rel_vy], dtype=np.float32))
        while len(self._history) < self.history_len:
            self._history.appendleft(self._history[0].copy())
        hist = np.concatenate(list(self._history), axis=0)

        mean_curv_1m, max_curv_2m, turn_density = self._curvature_features(ego_x, ego_y)
        ego_lat = self._signed_lateral_position(ego_x, ego_y)
        ego_s = self._project_progress(ego_x, ego_y)
        opp_s = self._project_progress(obs["poses_x"][1], obs["poses_y"][1])
        gap_s = self._signed_gap_s(ego_s, opp_s)
        ego_speed = float(obs["linear_vels_x"][0])
        opp_speed = float(obs["linear_vels_x"][1])
        closing_speed = opp_speed - ego_speed
        probe_switch = self._probe_switch_rate()
        probe_side_score = self._probe_side_score()
        threat = float(rel_x < 0.0 and np.hypot(rel_x, rel_y) < self.threat_distance and closing_speed > 0.0)

        extra = np.array([
            rel_x,
            rel_y,
            float(np.hypot(rel_x, rel_y)),
            closing_speed,
            mean_curv_1m,
            max_curv_2m,
            turn_density,
            self.track_width,
            ego_lat,
            gap_s,
            probe_switch,
            probe_side_score,
        ], dtype=np.float32)

        obs_vec = np.concatenate([hist, extra], axis=0).astype(np.float32)
        return obs_vec

    # ---------- action / reward ----------
    def _decode_action(self, action) -> RLActionParams:
        a = np.clip(np.asarray(action, dtype=np.float32), 0.0, 1.0)
        offset = float(a[0] * self.max_block_offset)
        hold_time = float(self.min_hold_time + a[1] * (self.max_hold_time - self.min_hold_time))
        return_rate = float(self.min_return_rate + a[2] * (self.max_return_rate - self.min_return_rate))
        return RLActionParams(offset, hold_time, return_rate)

    def _infer_collision_types(self, next_obs) -> Dict[str, bool]:
        rel_x, rel_y = relative_pose_in_ego_frame(
            next_obs["poses_x"][0], next_obs["poses_y"][0], next_obs["poses_theta"][0],
            next_obs["poses_x"][1], next_obs["poses_y"][1]
        )
        rel_dist = float(np.hypot(rel_x, rel_y))
        ego_coll = bool(next_obs["collisions"][0] > 0.5)
        opp_coll = bool(next_obs["collisions"][1] > 0.5)
        vehicle_contact = bool((ego_coll or opp_coll) and rel_dist < 0.45)
        return {
            "ego_collision": ego_coll,
            "opp_collision": opp_coll,
            "vehicle_contact": vehicle_contact,
            "ego_track_collision": ego_coll and not vehicle_contact,
            "opp_track_collision": opp_coll and not vehicle_contact,
        }


    def _compute_probe_score(self) -> float:
        """
        Continuous probe-intent score in [0, 1].
        High when the opponent has shown persistent lateral bias
        and/or repeated side switching while attacking from behind.
        """
        side_score = abs(self._probe_side_score())
        switch_score = np.clip(self._probe_switch_rate(), 0.0, 1.0)
        return float(np.clip(0.6 * side_score + 0.4 * switch_score, 0.0, 1.0))

    def _compute_opponent_pressure(self, obs):
        ego_x = float(obs["poses_x"][0])
        ego_y = float(obs["poses_y"][0])
        ego_yaw = float(obs["poses_theta"][0])

        opp_x = float(obs["poses_x"][1])
        opp_y = float(obs["poses_y"][1])

        rel_x, rel_y = relative_pose_in_ego_frame(ego_x, ego_y, ego_yaw, opp_x, opp_y)
        rel_dist = float(np.hypot(rel_x, rel_y))

        ego_v = float(obs["linear_vels_x"][0])
        opp_v = float(obs["linear_vels_x"][1])
        closing_speed = opp_v - ego_v

        behind_score = 1.0 / (1.0 + np.exp((rel_x + 0.2) / 0.35))
        close_score = np.exp(- (rel_dist / 2.0) ** 2)
        closing_score = np.clip((closing_speed + 0.2) / 1.2, 0.0, 1.0)
        lateral_alignment = np.exp(- (abs(rel_y) / 0.7) ** 2)
        probe_score = self._compute_probe_score()

        pressure = behind_score * close_score * (0.35 + 0.65 * closing_score) * (0.7 * lateral_alignment + 0.3 * probe_score)
        pressure = float(np.clip(pressure, 0.0, 1.0))

        return pressure, rel_x, rel_y, rel_dist, closing_speed

    def _compute_reward(self, obs, next_obs, params: RLActionParams):
        ego_s = self._project_progress(next_obs["poses_x"][0], next_obs["poses_y"][0])
        opp_s = self._project_progress(next_obs["poses_x"][1], next_obs["poses_y"][1])

        progress_delta = self._progress_delta(ego_s, self._prev_progress_s)
        self._prev_progress_s = ego_s

        gap_s = self._signed_gap_s(ego_s, opp_s)
        was_ahead = self._prev_gap_s > 0.0
        is_ahead = gap_s > 0.0
        self._prev_gap_s = gap_s

        lateral_error = abs(
            self._signed_lateral_position(
                next_obs["poses_x"][0],
                next_obs["poses_y"][0]
            )
        )
        collision_types = self._infer_collision_types(next_obs)

        pressure, rel_x, rel_y, rel_dist, closing_speed = self._compute_opponent_pressure(next_obs)

        if pressure > 0.2:
            self._metrics["threat_steps"] += 1
        if is_ahead:
            self._metrics["lead_hold_steps"] += 1
        if was_ahead and not is_ahead:
            self._metrics["passed_events"] += 1
        if collision_types["vehicle_contact"]:
            self._metrics["vehicle_contact_collisions"] += 1
        if collision_types["ego_track_collision"]:
            self._metrics["ego_track_collisions"] += 1
        if collision_types["opp_track_collision"]:
            self._metrics["opp_track_collisions"] += 1

        self._metrics["progress_total"] += max(progress_delta, 0.0)

        rw = self.reward_weights

        offset_norm = np.clip(params.offset / max(self.max_block_offset, 1e-6), 0.0, 1.0)
        lat_norm = np.clip(lateral_error / max(self.track_width * 0.5, 1e-6), 0.0, 2.0)

        action_vec = np.array(
            [params.offset, params.hold_time, params.return_rate],
            dtype=np.float32
        )

        action_delta = np.linalg.norm(action_vec - self._prev_action_vec)
        self._prev_action_vec = action_vec

        breakdown = {
            "progress": rw["progress"] * progress_delta,
            "lead_hold": rw["lead_hold"] * (1.0 if is_ahead else 0.0),
            "defense_under_pressure": rw["defense_under_pressure"] * pressure * (1.0 if is_ahead else 0.0),
            "block_usage": -rw["block_usage"] * offset_norm,
            "lateral_error": -rw["lateral_error"] * lat_norm * (1.25 - 0.5 * pressure),
            "action_smoothness": -rw["action_smoothness"] * action_delta,
            "passed_penalty": -rw["passed_penalty"] * (1.0 if was_ahead and not is_ahead else 0.0),
            "ego_track_collision": -rw["ego_track_collision"] * (1.0 if collision_types["ego_track_collision"] else 0.0),
            "ego_vehicle_collision": -rw["ego_vehicle_collision"] * (1.0 if collision_types["vehicle_contact"] else 0.0),
            "time_penalty": -rw["time_penalty"],
            "completion_bonus": 0.0,
        }

        if self._episode_step + 1 >= self.max_steps and is_ahead:
            breakdown["completion_bonus"] = rw["completion_bonus"]

        total_reward = float(sum(breakdown.values()))
        return total_reward, breakdown, collision_types

    def _finalize_episode_metrics(self):
        time_s = self._episode_step * self.timestep
        decision_steps = max(1, self._metrics["decision_steps"])
        metrics = dict(self._metrics)
        metrics.update({
            "block_hold_time_s": self._metrics["block_hold_steps"] * self.timestep,
            "threat_time_s": self._metrics["threat_steps"] * self.timestep,
            "position_held_time_s": self._metrics["lead_hold_steps"] * self.timestep,
            "episode_time_s": time_s,
            "completion_speed_mps": self._metrics["progress_total"] / max(time_s, 1e-6),
            "avg_offset_cmd": self._metrics["avg_offset_cmd"] / decision_steps,
            "avg_hold_cmd": self._metrics["avg_hold_cmd"] / decision_steps,
            "avg_return_cmd": self._metrics["avg_return_cmd"] / decision_steps,
            "successful_defense": int(self._metrics["passed_events"] == 0 and self._metrics["ego_track_collisions"] == 0),
            "latest_reward_breakdown": self._last_reward_breakdown,
        })
        return metrics


# ------------------------------------------------------------
# Stable-Baselines helpers
# ------------------------------------------------------------
class EpochRewardLogger:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.epoch = 0
        self._ensure_header()

    def _ensure_header(self):
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "timesteps", "ep_rew_mean", "ep_len_mean"])

    def callback_class(self):
        from stable_baselines3.common.callbacks import BaseCallback

        logger_outer = self

        class _CB(BaseCallback):
            def _on_step(self):
                return True

            def _on_rollout_end(self):
                logger_outer.epoch += 1
                ep_rew_mean = None
                ep_len_mean = None
                if len(self.model.ep_info_buffer) > 0:
                    ep_rew_mean = float(np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]))
                    ep_len_mean = float(np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer]))
                with open(logger_outer.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        logger_outer.epoch,
                        int(self.num_timesteps),
                        ep_rew_mean if ep_rew_mean is not None else "",
                        ep_len_mean if ep_len_mean is not None else "",
                    ])

        return _CB


def build_env_from_args(args, render_mode=None):
    return BlockDefenseEnv(
        config_path=args.config,
        seed=args.seed,
        render_mode=render_mode,
        max_steps=args.steps,
        timestep=args.timestep,
        ego_max_speed=args.ego_max_speed,
        opp_max_speed=args.opp_max_speed,
        max_lookahead=args.max_lookahead,
        min_lookahead_scale=args.min_lookahead_scale,
        min_speed_scale=args.min_speed_scale,
        lookahead_turn_gain=args.lookahead_turn_gain,
        speed_turn_gain=args.speed_turn_gain,
        scan_angle_min=args.scan_angle_min,
        scan_angle_max=args.scan_angle_max,
        max_scan_range=args.max_scan_range,
        opp_rrt_replan_every=args.opp_rrt_replan_every,
        track_width=args.track_width,
        random_spawn=args.random_spawn,
        spawn_gap_min=args.spawn_gap_min,
        spawn_gap_max=args.spawn_gap_max,
        ego_lateral_offset_rand=args.ego_lateral_offset_rand,
        opp_lateral_offset_rand=args.opp_lateral_offset_rand,
        spawn_yaw_rand=args.spawn_yaw_rand,
        spawn_max_tries=args.spawn_max_tries,
        history_len=args.history_len,
        probe_window=args.probe_window,
        max_block_offset=args.max_block_offset,
        min_hold_time=args.min_hold_time,
        max_hold_time=args.max_hold_time,
        min_return_rate=args.min_return_rate,
        max_return_rate=args.max_return_rate,
        engage_offset_threshold=args.engage_offset_threshold,
        threat_distance=args.threat_distance,
    )

class PeriodicRenderEvalCallback:
    def __init__(self, csv_path: str, build_env_fn, every_episodes: int = 50,
                 n_eval_episodes: int = 1, max_steps: int = 600):
        self.csv_path = csv_path
        self.build_env_fn = build_env_fn
        self.every_episodes = every_episodes
        self.n_eval_episodes = n_eval_episodes
        self.max_steps = max_steps
        self._ensure_header()

    def _ensure_header(self):
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "trigger_episode",
                    "eval_idx",
                    "episode_reward",
                    "successful_defense",
                    "block_hold_time_s",
                    "threat_time_s",
                    "position_held_time_s",
                    "vehicle_contact_collisions",
                    "ego_track_collisions",
                    "opp_track_collisions",
                    "passed_events",
                    "completion_speed_mps",
                ])

    def callback_class(self):
        from stable_baselines3.common.callbacks import BaseCallback

        outer = self

        class _CB(BaseCallback):
            def __init__(self):
                super().__init__()
                self.completed_episodes = 0

            def _run_visual_eval(self):
                eval_env = outer.build_env_fn()
                try:
                    for eval_idx in range(outer.n_eval_episodes):
                        obs, _ = eval_env.reset(seed=100000 + self.completed_episodes + eval_idx)
                        done = False
                        truncated = False
                        ep_reward = 0.0
                        info = {}

                        steps = 0
                        while not (done or truncated) and steps < outer.max_steps:
                            action, _ = self.model.predict(obs, deterministic=True)
                            obs, reward, done, truncated, info = eval_env.step(action)
                            ep_reward += reward
                            steps += 1

                        metrics = info.get("episode_metrics", {})
                        with open(outer.csv_path, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                self.completed_episodes,
                                eval_idx,
                                float(ep_reward),
                                metrics.get("successful_defense", ""),
                                metrics.get("block_hold_time_s", ""),
                                metrics.get("threat_time_s", ""),
                                metrics.get("position_held_time_s", ""),
                                metrics.get("vehicle_contact_collisions", ""),
                                metrics.get("ego_track_collisions", ""),
                                metrics.get("opp_track_collisions", ""),
                                metrics.get("passed_events", ""),
                                metrics.get("completion_speed_mps", ""),
                            ])
                finally:
                    try:
                        eval_env.close()
                    except Exception:
                        pass

            def _on_step(self):
                if outer.every_episodes <= 0:
                    return True

                dones = self.locals.get("dones", None)
                if dones is None:
                    return True

                # works for single-env and vec-env cases
                completed_now = int(np.sum(dones))
                if completed_now <= 0:
                    return True

                for _ in range(completed_now):
                    self.completed_episodes += 1
                    if self.completed_episodes % outer.every_episodes == 0:
                        print(f"\\n[visual-eval] Rendering episode {self.completed_episodes}")
                        self._run_visual_eval()

                return True

        return _CB

def train_ppo(args):
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
    except Exception as e:
        raise ImportError(
            "stable-baselines3 is required for training. Install it in your env first."
        ) from e

    os.makedirs(args.output_dir, exist_ok=True)
    env = build_env_from_args(args, render_mode=None)
    env = Monitor(env, filename=os.path.join(args.output_dir, "monitor.csv"))

    epoch_logger = EpochRewardLogger(os.path.join(args.output_dir, "epoch_metrics.csv"))
    epoch_cb = epoch_logger.callback_class()()

    ckpt_cb = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq),
        save_path=os.path.join(args.output_dir, "checkpoints"),
        name_prefix="ppo_block",
    )

    visual_cb = None
    if args.visualize_every_episodes > 0:
        visual_logger = PeriodicRenderEvalCallback(
            csv_path=os.path.join(args.output_dir, "visual_eval_metrics.csv"),
            every_episodes=args.visualize_every_episodes,
            n_eval_episodes=args.visualize_episodes,
            max_steps=args.visualize_max_steps,
            build_env_fn=lambda: build_env_from_args(
                args,
                render_mode=args.visualize_render_mode
            ),
        )
        visual_cb = visual_logger.callback_class()()

    callbacks = [epoch_cb, ckpt_cb]
    if visual_cb is not None:
        callbacks.append(visual_cb)

    callback = CallbackList(callbacks)

    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=os.path.join(args.output_dir, "tb"),
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        policy_kwargs=policy_kwargs,
        device=args.device,
        seed=args.seed,
    )
    model.learn(total_timesteps=args.total_timesteps, callback=callback, progress_bar=True)

    final_model = os.path.join(args.output_dir, "ppo_block_final")
    model.save(final_model)
    print(f"Saved model to {final_model}.zip")
    print(f"Monitor CSV: {os.path.join(args.output_dir, 'monitor.csv')}")
    print(f"Epoch metrics CSV: {os.path.join(args.output_dir, 'epoch_metrics.csv')}")


def evaluate_ppo(args):
    try:
        from stable_baselines3 import PPO
    except Exception as e:
        raise ImportError(
            "stable-baselines3 is required for evaluation. Install it in your env first."
        ) from e

    os.makedirs(args.output_dir, exist_ok=True)
    env = build_env_from_args(args, render_mode=None if args.headless else args.render_mode)
    model = PPO.load(args.model_path, device=args.device)

    csv_path = os.path.join(args.output_dir, "eval_metrics.csv")
    rows = []
    for ep in range(args.eval_episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        truncated = False
        ep_reward = 0.0
        final_info = {}
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            final_info = info
        metrics = final_info.get("episode_metrics", {})
        row = {"episode": ep, "episode_reward": ep_reward, **metrics}
        rows.append(row)
        print(f"Eval episode {ep}: reward={ep_reward:.2f}, success={metrics.get('successful_defense', 0)}")

    # write csv
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    if rows:
        numeric_keys = [k for k in fieldnames if k not in ("episode",)]
        for k in numeric_keys:
            vals = [r[k] for r in rows if isinstance(r.get(k), (int, float, np.floating, np.integer))]
            if len(vals) > 0:
                summary[k] = float(np.mean(vals))
    summary_path = os.path.join(args.output_dir, "eval_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Saved eval metrics to {csv_path}")
    print(f"Saved eval summary to {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="PPO blocking manager for F1TENTH demo")
    sub = parser.add_subparsers(dest="mode", required=True)

    def add_common(p):
        p.add_argument("--config", type=str, default="maps/config_example_map.yaml")
        p.add_argument("--steps", type=int, default=1000)
        p.add_argument("--timestep", type=float, default=0.01)
        p.add_argument("--ego-max-speed", type=float, default=4.0)
        p.add_argument("--opp-max-speed", type=float, default=4.8)
        p.add_argument("--max-lookahead", type=float, default=1.0)
        p.add_argument("--min-lookahead-scale", type=float, default=0.20)
        p.add_argument("--min-speed-scale", type=float, default=1.0)
        p.add_argument("--lookahead-turn-gain", type=float, default=0.9)
        p.add_argument("--speed-turn-gain", type=float, default=1.2)
        p.add_argument("--scan-angle-min", type=float, default=-2.35)
        p.add_argument("--scan-angle-max", type=float, default=2.35)
        p.add_argument("--max-scan-range", type=float, default=None)
        p.add_argument("--opp-rrt-replan-every", type=int, default=5)
        p.add_argument("--track-width", type=float, default=1.0)
        p.add_argument("--seed", type=int, default=123)
        p.add_argument("--random-spawn", action="store_true")
        p.add_argument("--spawn-gap-min", type=float, default=1.0)
        p.add_argument("--spawn-gap-max", type=float, default=2.5)
        p.add_argument("--ego-lateral-offset-rand", type=float, default=0.05)
        p.add_argument("--opp-lateral-offset-rand", type=float, default=0.20)
        p.add_argument("--spawn-yaw-rand", type=float, default=0.05)
        p.add_argument("--spawn-max-tries", type=int, default=20)
        p.add_argument("--history-len", type=int, default=6)
        p.add_argument("--probe-window", type=int, default=12)
        p.add_argument("--max-block-offset", type=float, default=0.45)
        p.add_argument("--min-hold-time", type=float, default=0.10)
        p.add_argument("--max-hold-time", type=float, default=1.50)
        p.add_argument("--min-return-rate", type=float, default=0.5)
        p.add_argument("--max-return-rate", type=float, default=8.0)
        p.add_argument("--engage-offset-threshold", type=float, default=0.03,
                       help="Deprecated and ignored in control; offset itself now determines engagement")
        p.add_argument("--threat-distance", type=float, default=2.5)
        p.add_argument("--render-mode", type=str, default="human_fast", choices=["human", "human_fast"])
        p.add_argument("--headless", action="store_true")
        p.add_argument("--output-dir", type=str, default="runs/block_rl")
        p.add_argument("--device", type=str, default="auto")
        p.add_argument("--visualize-every-episodes", type=int, default=50, help="Run one rendered eval episode every N completed training episodes; 0 disables it")
        p.add_argument("--visualize-episodes", type=int, default=1, help="How many rendered eval episodes to run each visualization trigger")
        p.add_argument("--visualize-max-steps", type=int, default=600, help="Max steps for each rendered visualization episode")
        p.add_argument("--visualize-render-mode", type=str, default="human_fast", choices=["human", "human_fast"], help="Render mode for periodic visualization episodes")

    train_p = sub.add_parser("train")
    add_common(train_p)
    train_p.add_argument("--total-timesteps", type=int, default=200000)
    train_p.add_argument("--learning-rate", type=float, default=3e-4)
    train_p.add_argument("--n-steps", type=int, default=2048)
    train_p.add_argument("--batch-size", type=int, default=256)
    train_p.add_argument("--gamma", type=float, default=0.995)
    train_p.add_argument("--gae-lambda", type=float, default=0.95)
    train_p.add_argument("--ent-coef", type=float, default=0.01)
    train_p.add_argument("--clip-range", type=float, default=0.2)
    train_p.add_argument("--checkpoint-freq", type=int, default=10000)

    eval_p = sub.add_parser("eval")
    add_common(eval_p)
    eval_p.add_argument("--model-path", type=str, required=True)
    eval_p.add_argument("--eval-episodes", type=int, default=20)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "train":
        train_ppo(args)
    elif args.mode == "eval":
        evaluate_ppo(args)


if __name__ == "__main__":
    main()
