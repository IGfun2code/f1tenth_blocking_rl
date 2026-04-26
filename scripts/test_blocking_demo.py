import os
import sys
import math
import gym
import yaml
import argparse
import numpy as np
from argparse import Namespace

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from f110_gym.envs.base_classes import Integrator
from src.controllers.pure_pursuit import DynamicPurePursuit
from src.planners.nominal_planner import load_waypoints_csv
from src.planners.blocking_planner import BlockingPlanner
from src.planners.rrt_star_overtake import RRTStarOvertakePlanner


def parse_args():
    parser = argparse.ArgumentParser(description="Blocking baseline demo: ego blocks, follower overtakes with RRT*")

    parser.add_argument("--config", type=str, default="maps/config_example_map.yaml")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--timestep", type=float, default=0.01)

    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--render-mode", type=str, default="human_fast", choices=["human", "human_fast"])
    parser.add_argument("--render-every", type=int, default=10)
    parser.add_argument("--print-every", type=int, default=10)

    parser.add_argument("--ego-max-speed", type=float, default=4.0)
    parser.add_argument("--opp-max-speed", type=float, default=4.8)

    parser.add_argument("--gap", type=float, default=2.0)
    parser.add_argument("--opp-lateral-offset", type=float, default=0.10)
    parser.add_argument("--opp-lateral-offset-rand", type=float, default=0.0, help="Uniform random spawn jitter added to follower lateral offset: U[-rand, rand]")

    parser.add_argument("--max-lookahead", type=float, default=1.0)
    parser.add_argument("--min-lookahead-scale", type=float, default=0.20)
    parser.add_argument("--min-speed-scale", type=float, default=1.0)
    parser.add_argument("--lookahead-turn-gain", type=float, default=0.9)
    parser.add_argument("--speed-turn-gain", type=float, default=1.2)

    # blocking path parameters (these are what RL will later choose)
    parser.add_argument("--block-offset", type=float, default=0.20)
    parser.add_argument("--block-hold-time", type=float, default=1.0)
    parser.add_argument("--block-return-rate", type=float, default=6.0)

    # baseline engage/disengage logic for demo only
    parser.add_argument("--block-trigger-distance", type=float, default=2.2)
    parser.add_argument("--block-release-distance", type=float, default=2.8)
    parser.add_argument("--block-trigger-closing-speed", type=float, default=0.02)

    # RRT* overtaking params
    parser.add_argument("--scan-angle-min", type=float, default=-2.35)
    parser.add_argument("--scan-angle-max", type=float, default=2.35)
    parser.add_argument("--max-scan-range", type=float, default=None)

    
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducible spawn perturbations")
    parser.add_argument("--opp-rrt-replan-every", type=int, default=5, help="Recompute follower RRT* every N steps; reuse last path in between")

    parser.add_argument("--random-spawn", action="store_true", help="Sample ego and follower spawn from the nominal path")
    parser.add_argument("--spawn-gap-min", type=float, default=1.0, help="Minimum follower gap behind ego in meters")
    parser.add_argument("--spawn-gap-max", type=float, default=2.5, help="Maximum follower gap behind ego in meters")
    parser.add_argument("--ego-lateral-offset-rand", type=float, default=0.05, help="Uniform random ego lateral offset in meters")
    parser.add_argument("--spawn-yaw-rand", type=float, default=0.05, help="Uniform random yaw perturbation in radians")
    parser.add_argument("--spawn-max-tries", type=int, default=20, help="Maximum attempts to find a valid randomized spawn")
    
    return parser.parse_args()


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

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    config_path = resolve_path(args.config, REPO_ROOT)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find config file: {config_path}")

    with open(config_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    config_dir = os.path.dirname(config_path)
    conf.map_path = resolve_path(conf.map_path, config_dir)
    conf.wpt_path = resolve_path(conf.wpt_path, config_dir)

    nominal_waypoints = load_waypoints_csv(
        conf.wpt_path,
        delimiter=conf.wpt_delim,
        skiprows=conf.wpt_rowskip,
        x_idx=conf.wpt_xind,
        y_idx=conf.wpt_yind,
        v_idx=conf.wpt_vind,
    )

    ego_controller = DynamicPurePursuit(
        wheelbase=0.15875 + 0.17145,
        max_steering_angle=0.4189,
        max_speed=args.ego_max_speed,
        max_look_ahead=args.max_lookahead,
        min_look_ahead_scale=args.min_lookahead_scale,
        min_speed_scale=args.min_speed_scale,
        look_ahead_turn_gain=args.lookahead_turn_gain,
        speed_turn_gain=args.speed_turn_gain,
    )

    opp_controller = DynamicPurePursuit(
        wheelbase=0.15875 + 0.17145,
        max_steering_angle=0.4189,
        max_speed=args.opp_max_speed,
        max_look_ahead=args.max_lookahead,
        min_look_ahead_scale=args.min_lookahead_scale,
        min_speed_scale=args.min_speed_scale,
        look_ahead_turn_gain=args.lookahead_turn_gain,
        speed_turn_gain=args.speed_turn_gain,
    )

    # ego follows nominal unless blocking baseline says otherwise
    ego_controller.set_waypoints(nominal_waypoints)

    # follower always goes through RRT* overtaking planner interface
    opp_overtake = RRTStarOvertakePlanner(
        nominal_waypoints=nominal_waypoints,
        goal_lookahead=3.0,
        max_path_speed=args.opp_max_speed,
    )

    blocking_planner = BlockingPlanner(
        nominal_waypoints=nominal_waypoints,
        horizon_points=120,
        ramp_distance=0.7,
        max_offset=0.45,
    )

    def render_callback(env_renderer):
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

        # ego nominal + block path
        blocking_planner.render_debug(e, draw_nominal=True)
        # follower nominal horizon + RRT* path + goal
        opp_overtake.render_debug(
            e,
            nominal_rgb=(50, 50, 90),
            path_rgb=(255, 170, 0),
            goal_rgb=(255, 255, 0),
        )

    env = gym.make(
        "f110_gym:f110-v0",
        map=conf.map_path,
        map_ext=conf.map_ext,
        num_agents=2,
        timestep=args.timestep,
        integrator=Integrator.RK4,
    )
    env.add_render_callback(render_callback)

    if args.random_spawn:
        spawn_ok = False
        spawn_meta = None

        for _ in range(args.spawn_max_tries):
            ego_pose, opp_pose, spawn_meta = sample_spawn_poses_from_nominal(
                nominal_waypoints=nominal_waypoints,
                rng=rng,
                gap_min=args.spawn_gap_min,
                gap_max=args.spawn_gap_max,
                ego_lat_rand=args.ego_lateral_offset_rand,
                opp_lat_rand=args.opp_lateral_offset_rand,
                yaw_rand=args.spawn_yaw_rand,
            )

            if np.linalg.norm(ego_pose[:2] - opp_pose[:2]) > 0.6:
                spawn_ok = True
                break

        if not spawn_ok:
            raise RuntimeError("Failed to find a valid randomized spawn.")

        print(
            f"Random spawn | ego_idx={spawn_meta['ego_idx']} "
            f"opp_idx={spawn_meta['opp_idx']} "
            f"gap={spawn_meta['gap']:.2f} "
            f"ego_lat={spawn_meta['ego_lat']:.2f} "
            f"opp_lat={spawn_meta['opp_lat']:.2f}"
        )

    else:
        ego_pose = np.array([conf.sx, conf.sy, conf.stheta], dtype=np.float32)

        forward = np.array([np.cos(conf.stheta), np.sin(conf.stheta)], dtype=np.float32)
        left = np.array([-np.sin(conf.stheta), np.cos(conf.stheta)], dtype=np.float32)

        lateral_jitter = rng.uniform(
            -args.opp_lateral_offset_rand,
            args.opp_lateral_offset_rand
        )
        opp_lateral_offset = args.opp_lateral_offset + lateral_jitter
        opp_xy = ego_pose[:2] - args.gap * forward + opp_lateral_offset * left
        opp_pose = np.array([opp_xy[0], opp_xy[1], conf.stheta], dtype=np.float32)

        print(
            f"Follower spawn lateral offset = {opp_lateral_offset:.3f} "
            f"(base={args.opp_lateral_offset:.3f}, jitter={lateral_jitter:.3f})"
        )

    poses = np.vstack([ego_pose, opp_pose])
    obs, _, done, _ = env.reset(poses)

    if not args.headless:
        env.render()

    block_active = False
    block_side = 1.0
    hold_steps_remaining = 0

    cached_opp_rrt_path = nominal_waypoints.copy()
    cached_opp_rrt_info = {
        "used_rrt": False,
        "goal_global": None,
        "max_deviation": 0.0,
    }

    for step in range(args.steps):
        ego_x, ego_y, ego_yaw = obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0]
        opp_x, opp_y, opp_yaw = obs["poses_x"][1], obs["poses_y"][1], obs["poses_theta"][1]

        rel_x, rel_y = relative_pose_in_ego_frame(ego_x, ego_y, ego_yaw, opp_x, opp_y)
        rel_dist = float(np.hypot(rel_x, rel_y))

        ego_actual_speed = float(obs["linear_vels_x"][0])
        opp_actual_speed = float(obs["linear_vels_x"][1])
        closing_speed = opp_actual_speed - ego_actual_speed

        # follower plans with RRT* only every few steps
        replan_now = (
            step == 0 or
            step % max(1, args.opp_rrt_replan_every) == 0
        )

        if replan_now:
            scan_n = len(obs["scans"][1])
            angle_increment = (args.scan_angle_max - args.scan_angle_min) / max(scan_n - 1, 1)

            new_rrt_path, new_rrt_info = opp_overtake.plan(
                opp_x,
                opp_y,
                opp_yaw,
                obs["scans"][1],
                args.scan_angle_min,
                angle_increment,
                max_scan_range=args.max_scan_range,
            )

            # only accept paths that pure pursuit can actually follow
            if new_rrt_path is not None and len(new_rrt_path) >= 2:
                cached_opp_rrt_path = new_rrt_path
                cached_opp_rrt_info = new_rrt_info
            else:
                # keep using the last valid path
                cached_opp_rrt_info = {
                    **cached_opp_rrt_info,
                    "used_rrt": False,
                }

        opp_controller.set_waypoints(cached_opp_rrt_path)
        opp_rrt_info = cached_opp_rrt_info

        # baseline blocking manager for demo only
        threat = (
            rel_x < 0.0 and
            rel_dist < args.block_trigger_distance and
            closing_speed > args.block_trigger_closing_speed
        )
        desired_side = 1.0 if rel_y >= 0.0 else -1.0

        if (not block_active) and threat:
            block_active = True
            block_side = desired_side
            hold_steps_remaining = max(1, int(round(args.block_hold_time / args.timestep)))
        elif block_active:
            hold_steps_remaining = max(0, hold_steps_remaining - 1)
            # update the side if the follower has clearly switched sides
            if abs(rel_y) > 0.10:
                block_side = desired_side
            release_ready = (
                hold_steps_remaining == 0 and
                (
                    rel_dist > args.block_release_distance or
                    closing_speed <= 0.0 or
                    rel_x > -0.10
                )
            )
            if release_ready:
                block_active = False

        if block_active:
            block_path = blocking_planner.build_blocking_path(
                ego_x=ego_x,
                ego_y=ego_y,
                side_sign=block_side,
                offset_magnitude=args.block_offset,
                hold_time=args.block_hold_time,
                return_rate=args.block_return_rate,
                current_speed=max(ego_actual_speed, 0.5),
            )
            ego_controller.set_waypoints(block_path)
        else:
            blocking_planner.clear_debug_path()
            ego_controller.set_waypoints(nominal_waypoints)

        ego_speed, ego_steer, ego_info = ego_controller.plan(ego_x, ego_y, ego_yaw)
        opp_speed, opp_steer, opp_info = opp_controller.plan(opp_x, opp_y, opp_yaw)

        actions = np.array([
            [ego_steer, ego_speed],
            [opp_steer, opp_speed],
        ], dtype=np.float32)

        obs, _, done, _ = env.step(actions)

        if not args.headless and (step % args.render_every == 0):
            env.render(mode=args.render_mode)

        if step % args.print_every == 0:
            print(
                f"step={step:03d} | "
                f"block={int(block_active)} side={'L' if block_side > 0 else 'R'} hold={hold_steps_remaining:03d} | "
                f"rel_x={rel_x:.2f} rel_y={rel_y:.2f} dist={rel_dist:.2f} closing={closing_speed:.2f} | "
                f"opp_rrt={int(opp_rrt_info['used_rrt'])} "
                f"replan={int(replan_now)} dev={opp_rrt_info['max_deviation']:.2f} | "
                f"ego_v={ego_speed:.2f} opp_v={opp_speed:.2f}"
            )

        if np.any(obs["collisions"]):
            print(f"Collision detected at step {step}.")
            break

        if done:
            print(f"Done at step {step}.")
            break

    print("Blocking demo complete.")


if __name__ == "__main__":
    main()
