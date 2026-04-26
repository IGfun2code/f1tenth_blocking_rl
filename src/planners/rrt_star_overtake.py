import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.controllers.pure_pursuit import nearest_point_on_trajectory


@dataclass
class TreeNode:
    x: float
    y: float
    parent: Optional[int] = None
    cost: float = 0.0


class RRTStarOvertakePlanner:
    """
    Direct-Gym RRT* overtaking planner.

    Input:
      - current vehicle pose
      - current LiDAR scan
      - nominal/global waypoints

    Output:
      - local/overtake path converted to global coordinates with an optional speed column

    Behavior:
      - if nominal corridor is free, returns nominal local horizon
      - if nominal corridor is blocked, returns an RRT* overtaking path
    """

    def __init__(
        self,
        nominal_waypoints: np.ndarray,
        meters_per_cell: float = 0.05,
        local_grid_width: float = 6.0,
        local_grid_height: float = 6.0,
        safety_radius: float = 0.15,
        max_iter: int = 350,
        step_size: float = 0.20,
        goal_bubble: float = 0.25,
        rewire_radius: float = 0.60,
        goal_lookahead: float = 3.0,
        goal_sample_rate: float = 0.25,
        nominal_corridor_halfwidth: float = 0.18,
        nominal_horizon_points: int = 120,
        max_path_speed: float = 5.0,
        overtake_speed_gain: float = 0.35,
        overtake_deviation_fullscale: float = 0.60,
    ):
        nominal_waypoints = np.asarray(nominal_waypoints, dtype=np.float32)
        if nominal_waypoints.ndim != 2 or nominal_waypoints.shape[1] not in (2, 3):
            raise ValueError("nominal_waypoints must be Nx2 or Nx3")

        self.nominal_waypoints = nominal_waypoints

        self.meters_per_cell = meters_per_cell
        self.local_grid_width = local_grid_width
        self.local_grid_height = local_grid_height
        self.width = int(local_grid_width / meters_per_cell)
        self.height = int(local_grid_height / meters_per_cell)

        self.origin_x = 0.0
        self.origin_y = -local_grid_width / 2.0

        self.safety_radius = safety_radius
        self.safety_cells = int(np.ceil(safety_radius / meters_per_cell))

        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_bubble = goal_bubble
        self.rewire_radius = rewire_radius
        self.goal_lookahead = goal_lookahead
        self.goal_sample_rate = goal_sample_rate
        self.nominal_corridor_halfwidth = nominal_corridor_halfwidth
        self.nominal_horizon_points = nominal_horizon_points

        self.max_path_speed = max_path_speed
        self.overtake_speed_gain = overtake_speed_gain
        self.overtake_deviation_fullscale = overtake_deviation_fullscale

        self.occ_grid = np.zeros((self.height, self.width), dtype=np.uint8)

        # debug storage for visualization
        self.last_nominal_global = None
        self.last_rrt_global = None
        self.last_goal_global = None
        self.last_tree_global = []

    # ----------------------------
    # Geometry helpers
    # ----------------------------
    @staticmethod
    def wrap_index(i: int, n: int) -> int:
        return i % n

    @staticmethod
    def transform_global_to_local(px, py, yaw, gx, gy):
        dx = gx - px
        dy = gy - py
        lx = math.cos(yaw) * dx + math.sin(yaw) * dy
        ly = -math.sin(yaw) * dx + math.cos(yaw) * dy
        return lx, ly

    @staticmethod
    def transform_local_to_global(px, py, yaw, lx, ly):
        gx = px + math.cos(yaw) * lx - math.sin(yaw) * ly
        gy = py + math.sin(yaw) * lx + math.cos(yaw) * ly
        return gx, gy

    def point_to_cell(self, x: float, y: float):
        row = int((x - self.origin_x) / self.meters_per_cell)
        col = int((y - self.origin_y) / self.meters_per_cell)
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return None
        return row, col

    def occupied(self, x: float, y: float) -> bool:
        cell = self.point_to_cell(x, y)
        if cell is None:
            return True
        return self.occ_grid[cell[0], cell[1]] == 1

    def apply_safety_bubble(self, row: int, col: int):
        for dr in range(-self.safety_cells, self.safety_cells + 1):
            for dc in range(-self.safety_cells, self.safety_cells + 1):
                if dr * dr + dc * dc <= self.safety_cells * self.safety_cells:
                    rr = row + dr
                    cc = col + dc
                    if 0 <= rr < self.height and 0 <= cc < self.width:
                        self.occ_grid[rr, cc] = 1

    # ----------------------------
    # Scan -> occupancy grid
    # ----------------------------
    def scan_to_grid(
        self,
        scan_ranges: np.ndarray,
        angle_min: float,
        angle_increment: float,
        max_scan_range: Optional[float] = None,
    ):
        self.occ_grid.fill(0)

        angle = angle_min
        for r in scan_ranges:
            if np.isfinite(r):
                if max_scan_range is not None:
                    r = min(r, max_scan_range)
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                cell = self.point_to_cell(x, y)
                if cell is not None:
                    self.apply_safety_bubble(cell[0], cell[1])
            angle += angle_increment

    # ----------------------------
    # Nominal path / goal logic
    # ----------------------------
    def _nominal_xy(self):
        return self.nominal_waypoints[:, :2]

    def _nominal_speed_at_idx(self, idx: int) -> float:
        if self.nominal_waypoints.shape[1] >= 3:
            return float(self.nominal_waypoints[idx, 2])
        return self.max_path_speed

    def find_nominal_goal_and_horizon(self, pose_x: float, pose_y: float):
        point = np.array([pose_x, pose_y], dtype=np.float32)
        nominal_xy = self._nominal_xy()
        _, _, _, nearest_idx = nearest_point_on_trajectory(point, nominal_xy)

        n = nominal_xy.shape[0]
        horizon_idxs = [self.wrap_index(nearest_idx + i, n) for i in range(self.nominal_horizon_points)]
        horizon = self.nominal_waypoints[horizon_idxs].copy()

        # walk along horizon until goal_lookahead distance is reached
        accum = 0.0
        goal_idx_local = len(horizon) - 1
        for i in range(1, len(horizon)):
            accum += np.linalg.norm(horizon[i, :2] - horizon[i - 1, :2])
            if accum >= self.goal_lookahead:
                goal_idx_local = i
                break

        goal_global = horizon[goal_idx_local].copy()
        return goal_global, horizon

    def nominal_horizon_local(self, pose_x: float, pose_y: float, pose_theta: float, horizon_global: np.ndarray):
        local = []
        for p in horizon_global:
            lx, ly = self.transform_global_to_local(pose_x, pose_y, pose_theta, p[0], p[1])
            if p.shape[0] >= 3:
                local.append([lx, ly, p[2]])
            else:
                local.append([lx, ly])
        return np.asarray(local, dtype=np.float32)

    def nominal_corridor_blocked(self, local_horizon: np.ndarray) -> bool:
        pts = local_horizon[:, :2]
        for i in range(len(pts) - 1):
            p0 = pts[i]
            p1 = pts[i + 1]
            seg = p1 - p0
            seg_len = np.linalg.norm(seg)
            samples = max(2, int(seg_len / 0.05))
            for s in range(samples + 1):
                t = s / samples
                x = p0[0] + t * seg[0]
                y = p0[1] + t * seg[1]

                # sample a small lateral corridor around the nominal centerline
                for lateral in (-self.nominal_corridor_halfwidth, 0.0, self.nominal_corridor_halfwidth):
                    if self.occupied(x, y + lateral):
                        return True
        return False

    # ----------------------------
    # RRT* core
    # ----------------------------
    def nearest(self, tree: List[TreeNode], sample: Tuple[float, float]) -> int:
        best = 0
        best_d = 1e9
        for i, node in enumerate(tree):
            d = math.hypot(node.x - sample[0], node.y - sample[1])
            if d < best_d:
                best = i
                best_d = d
        return best

    def near(self, tree: List[TreeNode], node: TreeNode) -> List[int]:
        if len(tree) < 2:
            return [0]
        radius = self.rewire_radius
        out = []
        for i, other in enumerate(tree):
            if math.hypot(other.x - node.x, other.y - node.y) <= radius:
                out.append(i)
        return out

    def steer(self, from_node: TreeNode, to_xy: Tuple[float, float]) -> TreeNode:
        dx = to_xy[0] - from_node.x
        dy = to_xy[1] - from_node.y
        dist = math.hypot(dx, dy)

        if dist < 1e-9:
            return TreeNode(from_node.x, from_node.y, None, 0.0)

        step = min(self.step_size, dist)
        x = from_node.x + step * dx / dist
        y = from_node.y + step * dy / dist
        return TreeNode(x, y, None, 0.0)

    def line_cost(self, n1: TreeNode, n2: TreeNode) -> float:
        return math.hypot(n1.x - n2.x, n1.y - n2.y)

    def check_edge_collision_xy(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        dist = math.hypot(x1 - x0, y1 - y0)
        steps = max(2, int(dist / 0.05))
        for i in range(steps + 1):
            t = i / steps
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            if self.occupied(x, y):
                return False
        return True

    def check_edge_collision(self, n1: TreeNode, n2: TreeNode) -> bool:
        return self.check_edge_collision_xy(n1.x, n1.y, n2.x, n2.y)

    def choose_parent(self, tree: List[TreeNode], new_node: TreeNode, near_ids: List[int]) -> TreeNode:
        best_parent = None
        best_cost = 1e9

        for idx in near_ids:
            cand = tree[idx]
            if not self.check_edge_collision(cand, new_node):
                continue
            cost = cand.cost + self.line_cost(cand, new_node)
            if cost < best_cost:
                best_cost = cost
                best_parent = idx

        if best_parent is None:
            return new_node

        new_node.parent = best_parent
        new_node.cost = best_cost
        return new_node

    def rewire(self, tree: List[TreeNode], new_idx: int, near_ids: List[int]):
        new_node = tree[new_idx]
        for idx in near_ids:
            if idx == new_idx:
                continue
            other = tree[idx]
            if not self.check_edge_collision(new_node, other):
                continue
            cand_cost = new_node.cost + self.line_cost(new_node, other)
            if cand_cost < other.cost:
                other.parent = new_idx
                other.cost = cand_cost

    def sample(self, goal_xy_local: Tuple[float, float]) -> Tuple[float, float]:
        if np.random.rand() < self.goal_sample_rate:
            return goal_xy_local

        x = np.random.uniform(0.0, self.local_grid_height)
        y = np.random.uniform(-self.local_grid_width / 2.0, self.local_grid_width / 2.0)
        return (x, y)

    def extract_path(self, tree: List[TreeNode], goal_idx: int) -> np.ndarray:
        pts = []
        idx = goal_idx
        while idx is not None:
            node = tree[idx]
            pts.append([node.x, node.y])
            idx = node.parent
        pts.reverse()
        return np.asarray(pts, dtype=np.float32)

    def smooth_path(self, local_path_xy: np.ndarray) -> np.ndarray:
        if len(local_path_xy) <= 2:
            return local_path_xy

        smoothed = [local_path_xy[0]]
        i = 0
        while i < len(local_path_xy) - 1:
            j = len(local_path_xy) - 1
            found = False
            while j > i + 1:
                if self.check_edge_collision_xy(
                    local_path_xy[i, 0], local_path_xy[i, 1],
                    local_path_xy[j, 0], local_path_xy[j, 1]
                ):
                    smoothed.append(local_path_xy[j])
                    i = j
                    found = True
                    break
                j -= 1
            if not found:
                smoothed.append(local_path_xy[i + 1])
                i += 1

        if np.linalg.norm(smoothed[-1] - local_path_xy[-1]) > 1e-6:
            smoothed.append(local_path_xy[-1])

        return np.asarray(smoothed, dtype=np.float32)

    # ----------------------------
    # Path annotation / render
    # ----------------------------
    def local_to_global_path(self, pose_x: float, pose_y: float, pose_theta: float, local_xy: np.ndarray):
        out = []
        for p in local_xy:
            gx, gy = self.transform_local_to_global(pose_x, pose_y, pose_theta, p[0], p[1])
            out.append([gx, gy])
        return np.asarray(out, dtype=np.float32)

    def annotate_speed(self, global_xy: np.ndarray, nominal_goal_speed: float, nominal_global_horizon: np.ndarray) -> np.ndarray:
        if global_xy.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)

        # compute path deviation from nominal horizon
        nominal_xy = nominal_global_horizon[:, :2]
        max_dev = 0.0
        for p in global_xy:
            _, dist, _, _ = nearest_point_on_trajectory(p, nominal_xy)
            max_dev = max(max_dev, float(dist))

        dev_score = np.clip(max_dev / max(self.overtake_deviation_fullscale, 1e-6), 0.0, 1.0)
        speed_boost = 1.0 + self.overtake_speed_gain * dev_score

        out = np.zeros((global_xy.shape[0], 3), dtype=np.float32)
        out[:, :2] = global_xy
        out[:, 2] = np.clip(nominal_goal_speed * speed_boost, 0.0, self.max_path_speed)
        return out

    def _update_draw_points(self, env_renderer, draw_list, points_xy, color_rgb):
        from pyglet.gl import GL_POINTS

        if points_xy is None:
            points_xy = np.zeros((0, 2), dtype=np.float32)

        scaled = 50.0 * points_xy
        for i in range(points_xy.shape[0]):
            verts = [scaled[i, 0], scaled[i, 1], 0.0]
            if len(draw_list) <= i:
                b = env_renderer.batch.add(
                    1,
                    GL_POINTS,
                    None,
                    ("v3f/stream", verts),
                    ("c3B/stream", list(color_rgb)),
                )
                draw_list.append(b)
            else:
                draw_list[i].vertices = verts

        for j in range(points_xy.shape[0], len(draw_list)):
            draw_list[j].vertices = [-1e6, -1e6, 0.0]

    def render_debug(self, env_renderer, nominal_rgb=(100, 160, 255), path_rgb=(80, 255, 80), goal_rgb=(255, 255, 0)):
        if not hasattr(self, "_nominal_draw"):
            self._nominal_draw = []
            self._path_draw = []
            self._goal_draw = []

        nominal_xy = None if self.last_nominal_global is None else self.last_nominal_global[:, :2]
        path_xy = None if self.last_rrt_global is None else self.last_rrt_global[:, :2]
        goal_xy = None if self.last_goal_global is None else np.asarray([self.last_goal_global[:2]], dtype=np.float32)

        self._update_draw_points(env_renderer, self._nominal_draw, nominal_xy, nominal_rgb)
        self._update_draw_points(env_renderer, self._path_draw, path_xy, path_rgb)
        self._update_draw_points(env_renderer, self._goal_draw, goal_xy, goal_rgb)

    # ----------------------------
    # Main planning API
    # ----------------------------
    def plan(
        self,
        pose_x: float,
        pose_y: float,
        pose_theta: float,
        scan_ranges: np.ndarray,
        angle_min: float,
        angle_increment: float,
        max_scan_range: Optional[float] = None,
    ):
        # 1) build occupancy grid
        self.scan_to_grid(scan_ranges, angle_min, angle_increment, max_scan_range=max_scan_range)

        # 2) get nominal goal and local nominal horizon
        goal_global, nominal_horizon_global = self.find_nominal_goal_and_horizon(pose_x, pose_y)
        nominal_goal_speed = float(goal_global[2]) if goal_global.shape[0] >= 3 else self.max_path_speed

        nominal_horizon_local = self.nominal_horizon_local(
            pose_x, pose_y, pose_theta, nominal_horizon_global
        )
        goal_local = nominal_horizon_local[min(len(nominal_horizon_local) - 1, np.argmax(
            np.cumsum(
                np.r_[0.0, np.linalg.norm(np.diff(nominal_horizon_local[:, :2], axis=0), axis=1)]
            ) >= self.goal_lookahead
        ))][:2]

        # store debug
        self.last_nominal_global = nominal_horizon_global.copy()
        self.last_goal_global = goal_global.copy()

        # 3) if nominal corridor is clear, just use nominal horizon
        if not self.nominal_corridor_blocked(nominal_horizon_local):
            self.last_rrt_global = nominal_horizon_global.copy()
            return nominal_horizon_global.copy(), {
                "used_rrt": False,
                "goal_global": goal_global.copy(),
                "max_deviation": 0.0,
            }

        # 4) RRT* in local frame
        tree: List[TreeNode] = [TreeNode(0.0, 0.0, parent=None, cost=0.0)]
        best_goal_idx = None
        best_goal_cost = 1e9

        for _ in range(self.max_iter):
            sample_xy = self.sample((float(goal_local[0]), float(goal_local[1])))
            nearest_idx = self.nearest(tree, sample_xy)
            nearest_node = tree[nearest_idx]

            new_node = self.steer(nearest_node, sample_xy)
            if self.occupied(new_node.x, new_node.y):
                continue
            if not self.check_edge_collision(nearest_node, new_node):
                continue

            near_ids = self.near(tree, new_node)
            new_node.parent = nearest_idx
            new_node.cost = nearest_node.cost + self.line_cost(nearest_node, new_node)
            new_node = self.choose_parent(tree, new_node, near_ids)

            tree.append(new_node)
            new_idx = len(tree) - 1
            self.rewire(tree, new_idx, near_ids)

            dist_to_goal = math.hypot(new_node.x - goal_local[0], new_node.y - goal_local[1])
            if dist_to_goal <= self.goal_bubble:
                total_cost = new_node.cost + dist_to_goal
                if total_cost < best_goal_cost:
                    best_goal_cost = total_cost
                    best_goal_idx = new_idx

        # 5) fallback if no explicit goal connection found
        if best_goal_idx is None:
            best_idx = 0
            best_score = -1e9
            for i, node in enumerate(tree):
                score = node.x - 0.5 * abs(node.y) - 0.2 * math.hypot(node.x - goal_local[0], node.y - goal_local[1])
                if score > best_score:
                    best_score = score
                    best_idx = i
            best_goal_idx = best_idx

        local_path_xy = self.extract_path(tree, best_goal_idx)
        local_path_xy = self.smooth_path(local_path_xy)
        if local_path_xy is None or len(local_path_xy) < 2:
            self.last_rrt_global = nominal_horizon_global.copy()
            return nominal_horizon_global.copy(), {
                "used_rrt": False,
                "goal_global": goal_global.copy(),
                "max_deviation": 0.0,
            }

        # 6) convert to global and attach speed
        global_path_xy = self.local_to_global_path(pose_x, pose_y, pose_theta, local_path_xy)
        global_path = self.annotate_speed(global_path_xy, nominal_goal_speed, nominal_horizon_global)

        self.last_rrt_global = global_path.copy()

        # 7) debug meta
        max_dev = 0.0
        nominal_xy = nominal_horizon_global[:, :2]
        for p in global_path[:, :2]:
            _, d, _, _ = nearest_point_on_trajectory(p, nominal_xy)
            max_dev = max(max_dev, float(d))

        return global_path, {
            "used_rrt": True,
            "goal_global": goal_global.copy(),
            "max_deviation": max_dev,
        }