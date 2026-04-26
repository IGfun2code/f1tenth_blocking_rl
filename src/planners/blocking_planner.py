import numpy as np
from pyglet.gl import GL_POINTS

from src.controllers.pure_pursuit import nearest_point_on_trajectory


class BlockingPlanner:
    """
    Generates a local blocking trajectory by laterally shifting the nominal path.

    Important:
    - This class does NOT decide when to block.
    - This class does NOT decide left/right.
    - This class does NOT decide aggressiveness.
    It only takes in externally provided blocking parameters and builds a path.
    """

    def __init__(
        self,
        nominal_waypoints: np.ndarray,
        horizon_points: int = 120,
        ramp_distance: float = 0.7,
        max_offset: float = 0.45,
    ):
        nominal_waypoints = np.asarray(nominal_waypoints, dtype=np.float32)
        if nominal_waypoints.ndim != 2 or nominal_waypoints.shape[1] not in (2, 3):
            raise ValueError("nominal_waypoints must be Nx2 or Nx3.")

        self.nominal_waypoints = nominal_waypoints
        self.horizon_points = horizon_points
        self.ramp_distance = ramp_distance
        self.max_offset = max_offset

        self.last_block_path = None
        self._nominal_draw = []
        self._block_draw = []

    def clear_debug_path(self):
        self.last_block_path = None

    def _slice_horizon(self, start_idx: int) -> np.ndarray:
        n = self.nominal_waypoints.shape[0]
        idxs = [(start_idx + i) % n for i in range(self.horizon_points)]
        return self.nominal_waypoints[idxs].copy()

    @staticmethod
    def _compute_normals(points_xy: np.ndarray) -> np.ndarray:
        diffs = np.zeros_like(points_xy)
        diffs[1:-1] = points_xy[2:] - points_xy[:-2]
        diffs[0] = points_xy[1] - points_xy[0]
        diffs[-1] = points_xy[-1] - points_xy[-2]

        norms = np.linalg.norm(diffs, axis=1, keepdims=True)
        tangents = diffs / np.maximum(norms, 1e-6)

        # left normal
        normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
        return normals

    @staticmethod
    def _cumulative_distance(points_xy: np.ndarray) -> np.ndarray:
        ds = np.linalg.norm(np.diff(points_xy, axis=0), axis=1)
        return np.concatenate(([0.0], np.cumsum(ds)))

    def build_blocking_path(
        self,
        ego_x: float,
        ego_y: float,
        side_sign: float,
        offset_magnitude: float,
        hold_time: float,
        return_rate: float,
        current_speed: float,
    ) -> np.ndarray:
        """
        side_sign: +1 for left block, -1 for right block
        offset_magnitude: lateral offset in meters
        hold_time: seconds to approximately hold the full offset
        return_rate: larger means faster return to nominal line
        current_speed: used to convert hold_time into approximate hold distance
        """
        nominal_xy = self.nominal_waypoints[:, :2]
        ego_pos = np.array([ego_x, ego_y], dtype=np.float32)

        _, _, _, nearest_idx = nearest_point_on_trajectory(ego_pos, nominal_xy)
        horizon = self._slice_horizon(nearest_idx)

        path_xy = horizon[:, :2]
        normals = self._compute_normals(path_xy)
        s = self._cumulative_distance(path_xy)

        side_sign = 1.0 if side_sign >= 0.0 else -1.0
        offset_magnitude = float(np.clip(offset_magnitude, 0.0, self.max_offset))
        hold_distance = max(0.0, float(current_speed) * float(hold_time))
        ramp_distance = max(1e-3, self.ramp_distance)
        return_rate = max(0.0, float(return_rate))

        offset_profile = np.zeros_like(s)

        for i, si in enumerate(s):
            if si < ramp_distance:
                mag = offset_magnitude * (si / ramp_distance)
            elif si < ramp_distance + hold_distance:
                mag = offset_magnitude
            else:
                decay_s = si - (ramp_distance + hold_distance)
                mag = offset_magnitude * np.exp(-return_rate * decay_s)

            offset_profile[i] = side_sign * mag

        blocked = horizon.copy()
        blocked[:, :2] = path_xy + normals * offset_profile[:, None]

        self.last_block_path = blocked
        return blocked

    def _update_draw_points(self, env_renderer, draw_list, points_xy, color_rgb):
        if points_xy is None:
            points_xy = np.zeros((0, 2), dtype=np.float32)

        scaled_points = 50.0 * points_xy

        for i in range(points_xy.shape[0]):
            verts = [scaled_points[i, 0], scaled_points[i, 1], 0.0]
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

        # hide extra old points
        for j in range(points_xy.shape[0], len(draw_list)):
            draw_list[j].vertices = [-1e6, -1e6, 0.0]

    def render_debug(self, env_renderer, draw_nominal: bool = True):
        if draw_nominal:
            self._update_draw_points(
                env_renderer,
                self._nominal_draw,
                self.nominal_waypoints[:, :2],
                color_rgb=(120, 170, 255),   # light blue
            )

        if self.last_block_path is not None:
            self._update_draw_points(
                env_renderer,
                self._block_draw,
                self.last_block_path[:, :2],
                color_rgb=(255, 80, 80),     # red
            )
        else:
            self._update_draw_points(
                env_renderer,
                self._block_draw,
                None,
                color_rgb=(255, 80, 80),
            )