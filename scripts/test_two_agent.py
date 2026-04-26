import os
import sys
import time
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


def parse_args():
    parser = argparse.ArgumentParser(description="Two-agent F1TENTH waypoint-follow test")

    parser.add_argument(
        "--config",
        type=str,
        default="maps/config_example_map.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of simulation steps",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=0.01,
        help="Simulator timestep",
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without rendering",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="human_fast",
        choices=["human", "human_fast"],
        help="Render mode",
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=5,
        help="Render every N steps",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Print every N steps",
    )

    parser.add_argument(
        "--ego-max-speed",
        type=float,
        default=4.0,
        help="Ego controller max speed",
    )
    parser.add_argument(
        "--opp-max-speed",
        type=float,
        default=3.8,
        help="Opponent controller max speed",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=0.8,
        help="Initial gap behind ego for opponent",
    )

    parser.add_argument(
        "--max-lookahead",
        type=float,
        default=1.0,
        help="Maximum lookahead distance",
    )
    parser.add_argument(
        "--min-lookahead-scale",
        type=float,
        default=0.20,
        help="Minimum lookahead scaling factor",
    )
    parser.add_argument(
        "--min-speed-scale",
        type=float,
        default=1.0,
        help="Minimum speed scaling factor",
    )
    parser.add_argument(
        "--lookahead-turn-gain",
        type=float,
        default=0.9,
        help="Turn sensitivity for dynamic lookahead",
    )
    parser.add_argument(
        "--speed-turn-gain",
        type=float,
        default=1.2,
        help="Turn sensitivity for dynamic speed scaling",
    )

    return parser.parse_args()


def resolve_path(path_str: str, base_dir: str) -> str:
    if os.path.isabs(path_str):
        return path_str
    return os.path.normpath(os.path.join(base_dir, path_str))


def main():
    args = parse_args()

    config_path = resolve_path(args.config, REPO_ROOT)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find config file: {config_path}")

    with open(config_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    config_dir = os.path.dirname(config_path)

    # Resolve paths in the YAML relative to the YAML file location
    conf.map_path = resolve_path(conf.map_path, config_dir)
    conf.wpt_path = resolve_path(conf.wpt_path, config_dir)

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

    nominal_waypoints = load_waypoints_csv(
        conf.wpt_path,
        delimiter=conf.wpt_delim,
        skiprows=conf.wpt_rowskip,
        x_idx=conf.wpt_xind,
        y_idx=conf.wpt_yind,
        v_idx=conf.wpt_vind,
    )

    ego_controller.set_waypoints(nominal_waypoints)
    opp_controller.set_waypoints(nominal_waypoints)

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

    env = gym.make(
        "f110_gym:f110-v0",
        map=conf.map_path,
        map_ext=conf.map_ext,
        num_agents=2,
        timestep=args.timestep,
        integrator=Integrator.RK4,
    )
    env.add_render_callback(render_callback)

    ego_pose = np.array([conf.sx, conf.sy, conf.stheta], dtype=np.float32)

    opp_pose = np.array([
        conf.sx - args.gap * np.cos(conf.stheta),
        conf.sy - args.gap * np.sin(conf.stheta),
        conf.stheta,
    ], dtype=np.float32)

    poses = np.vstack([ego_pose, opp_pose])
    obs, _, done, _ = env.reset(poses)

    if not args.headless:
        env.render()

    for step in range(args.steps):
        ego_speed, ego_steer, ego_info = ego_controller.plan(
            obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0]
        )
        opp_speed, opp_steer, opp_info = opp_controller.plan(
            obs["poses_x"][1], obs["poses_y"][1], obs["poses_theta"][1]
        )

        actions = np.array([
            [ego_steer, ego_speed],
            [opp_steer, opp_speed],
        ], dtype=np.float32)

        obs, _, done, info = env.step(actions)

        if not args.headless and (step % args.render_every == 0):
            env.render(mode=args.render_mode)

        if step % args.print_every == 0:
            print(
                f"step={step:03d} | "
                f"ego_v={ego_speed:.2f}, ego_delta={ego_steer:.2f}, ego_LA={ego_info['lookahead']:.2f} | "
                f"opp_v={opp_speed:.2f}, opp_delta={opp_steer:.2f}"
            )

        if np.any(obs["collisions"]):
            print(f"Collision detected at step {step}.")
            break

        if done:
            print(f"Done at step {step}.")
            break

    print("Two-agent waypoint-follow test complete.")


if __name__ == "__main__":
    main()