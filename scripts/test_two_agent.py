import os
import sys
import time
import numpy as np

# Allow imports when running from repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from f110_gym.envs.f110_env import F110Env


def main():
    """
    Minimal two-agent sanity test for direct f1tenth_gym usage.

    What it does:
    1. Creates a 2-agent environment
    2. Resets ego and opponent with fixed poses
    3. Steps both cars forward with constant commands
    4. Prints positions, speeds, and collision flags
    """

    # Update this path if your local gym repo is elsewhere
    map_dir = os.path.expanduser("~/f1tenth_gym/gym/f110_gym/envs/maps")
    map_name = "vegas"
    map_path = os.path.join(map_dir, map_name)

    if not os.path.exists(map_path + ".yaml"):
        raise FileNotFoundError(
            f"Could not find map yaml at {map_path + '.yaml'}\n"
            f"Check your f1tenth_gym path and map name."
        )

    env = F110Env(
        map=map_path,
        map_ext=".png",
        num_agents=2,
    )

    # [x, y, theta] for each car
    poses = np.array([
        [0.0, 0.0, 0.0],    # ego
        [-1.0, 0.0, 0.0],   # opponent behind ego
    ], dtype=np.float32)

    obs, step_reward, done, info = env.reset(poses=poses)

    print("Initial observation keys:", obs.keys())
    print("Initial poses_x:", obs["poses_x"])
    print("Initial poses_y:", obs["poses_y"])
    print("Initial theta:", obs["poses_theta"])
    print()

    max_steps = 300

    for step in range(max_steps):
        # Action format: [steering, speed]
        actions = np.array([
            [0.0, 1.5],   # ego
            [0.0, 1.2],   # opponent
        ], dtype=np.float32)

        obs, step_reward, done, info = env.step(actions)

        ego_x, opp_x = obs["poses_x"][0], obs["poses_x"][1]
        ego_y, opp_y = obs["poses_y"][0], obs["poses_y"][1]
        ego_v, opp_v = obs["linear_vels_x"][0], obs["linear_vels_x"][1]
        collisions = obs["collisions"]

        print(
            f"step={step:03d} | "
            f"ego=({ego_x:.2f}, {ego_y:.2f}) v={ego_v:.2f} | "
            f"opp=({opp_x:.2f}, {opp_y:.2f}) v={opp_v:.2f} | "
            f"collisions={collisions}"
        )

        if np.any(collisions):
            print("Collision detected. Ending test.")
            break

        if done:
            print("Environment returned done=True. Ending test.")
            break

        # Small delay so prints are readable
        time.sleep(0.02)

    print("Two-agent test complete.")


if __name__ == "__main__":
    main()