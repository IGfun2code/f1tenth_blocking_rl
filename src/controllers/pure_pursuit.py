import numpy as np


def nearest_point_on_trajectory(point: np.ndarray, trajectory: np.ndarray):
    """
    Return nearest projection point on a piecewise-linear trajectory.
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2

    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot(point - trajectory[i, :], diffs[i, :])

    t = dots / np.maximum(l2s, 1e-12)
    t = np.clip(t, 0.0, 1.0)

    projections = trajectory[:-1, :] + (t[:, None] * diffs)

    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))

    min_idx = np.argmin(dists)
    return projections[min_idx], dists[min_idx], t[min_idx], min_idx


def first_point_on_trajectory_intersecting_circle(
    point: np.ndarray,
    radius: float,
    trajectory: np.ndarray,
    t: float = 0.0,
    wrap: bool = False,
):
    """
    Find the first point along the trajectory that intersects a circle centered at `point`.
    """
    start_i = int(t)
    start_t = t % 1.0

    first_t = None
    first_i = None
    first_p = None

    for i in range(start_i, trajectory.shape[0] - 1):
        start = trajectory[i, :]
        end = trajectory[i + 1, :] + 1e-6
        V = end - start

        a = np.dot(V, V)
        b = 2.0 * np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point, point) - 2.0 * np.dot(start, point) - radius * radius
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            continue

        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0 * a)
        t2 = (-b + discriminant) / (2.0 * a)

        if i == start_i:
            if 0.0 <= t1 <= 1.0 and t1 >= start_t:
                first_t, first_i, first_p = t1, i, start + t1 * V
                break
            if 0.0 <= t2 <= 1.0 and t2 >= start_t:
                first_t, first_i, first_p = t2, i, start + t2 * V
                break
        else:
            if 0.0 <= t1 <= 1.0:
                first_t, first_i, first_p = t1, i, start + t1 * V
                break
            if 0.0 <= t2 <= 1.0:
                first_t, first_i, first_p = t2, i, start + t2 * V
                break

    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0], :]
            end = trajectory[(i + 1) % trajectory.shape[0], :] + 1e-6
            V = end - start

            a = np.dot(V, V)
            b = 2.0 * np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point, point) - 2.0 * np.dot(start, point) - radius * radius
            discriminant = b * b - 4 * a * c

            if discriminant < 0:
                continue

            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0 * a)
            t2 = (-b + discriminant) / (2.0 * a)

            if 0.0 <= t1 <= 1.0:
                first_t, first_i, first_p = t1, i, start + t1 * V
                break
            if 0.0 <= t2 <= 1.0:
                first_t, first_i, first_p = t2, i, start + t2 * V
                break

    return first_p, first_i, first_t


class DynamicPurePursuit:
    """
    Adjusts the look ahead and the speed based on the path shape
    """

    def __init__(
        self,
        wheelbase: float = 0.15875 + 0.17145,   # 0.3302 m
        max_steering_angle: float = 0.4189,
        max_speed: float = 2.0,
        max_look_ahead: float = 1.0,
        min_look_ahead_scale: float = 0.20,
        min_speed_scale: float = 0.80,
        look_ahead_turn_gain: float = 0.9,
        speed_turn_gain: float = 1.2,
        max_reacquire: float = 20.0,
    ):
        self.wheelbase = wheelbase
        self.max_steering_angle = max_steering_angle
        self.max_speed = max_speed

        self.max_look_ahead = max_look_ahead
        self.min_look_ahead_scale = min_look_ahead_scale
        self.min_speed_scale = min_speed_scale
        self.look_ahead_turn_gain = look_ahead_turn_gain
        self.speed_turn_gain = speed_turn_gain
        self.max_reacquire = max_reacquire

        self.waypoints = None

    def _xy_waypoints(self) -> np.ndarray:
        if self.waypoints is None:
            raise RuntimeError("Pure pursuit waypoints not set.")
        if self.waypoints.shape[0] < 2:
            raise RuntimeError("Pure pursuit needs at least 2 waypoints.")
        return self.waypoints[:, :2]

    def set_waypoints(self, waypoints: np.ndarray):
        """
        waypoints: Nx2 or Nx3 array
        columns:
            [x, y] or [x, y, nominal_speed]
        """
        waypoints = np.asarray(waypoints, dtype=np.float32)
        if waypoints.ndim != 2 or waypoints.shape[1] not in (2, 3):
            raise ValueError("Waypoints must be Nx2 or Nx3.")
        self.waypoints = waypoints

    def load_waypoints_from_csv(self, csv_path: str, delimiter: str = ","):
        wpts = np.loadtxt(csv_path, delimiter=delimiter)
        if wpts.ndim == 1:
            wpts = wpts.reshape(1, -1)
        self.set_waypoints(wpts)

    def get_scale(self, turning_magnitude: float, gain: float, min_scale: float) -> float:
        scale = np.exp(-gain * abs(turning_magnitude))
        return min_scale + (1.0 - min_scale) * scale

    def _xy_waypoints(self) -> np.ndarray:
        if self.waypoints is None:
            raise RuntimeError("Pure pursuit waypoints not set.")
        return self.waypoints[:, :2]

    def _waypoint_speed(self, idx: int) -> float:
        if self.waypoints.shape[1] >= 3:
            return float(self.waypoints[idx, 2])
        return self.max_speed

    def _get_current_waypoint(self, position: np.ndarray, lookahead_distance: float):
        wpts = self._xy_waypoints()

        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)

        if nearest_dist < lookahead_distance:
            lookahead_point, i2, _ = first_point_on_trajectory_intersecting_circle(
                position, lookahead_distance, wpts, i + t, wrap=True
            )
            if i2 is None:
                return None

            current_waypoint = np.empty((3,), dtype=np.float32)
            current_waypoint[0:2] = lookahead_point
            current_waypoint[2] = self._waypoint_speed(i2)
            return current_waypoint

        if nearest_dist < self.max_reacquire:
            current_waypoint = np.empty((3,), dtype=np.float32)
            current_waypoint[0:2] = wpts[i, :]
            current_waypoint[2] = self._waypoint_speed(i)
            return current_waypoint

        return None

    def _get_actuation(
        self,
        pose_theta: float,
        lookahead_point: np.ndarray,
        position: np.ndarray,
        lookahead_distance: float,
    ):
        rel = lookahead_point[0:2] - position
        waypoint_y = np.dot(
            np.array([np.sin(-pose_theta), np.cos(-pose_theta)], dtype=np.float32),
            rel,
        )
        nominal_speed = float(lookahead_point[2])

        if abs(waypoint_y) < 1e-6:
            return nominal_speed, 0.0, 0.0

        curvature = 2.0 * waypoint_y / max(lookahead_distance ** 2, 1e-6)
        steering_angle = np.arctan(self.wheelbase * curvature)
        steering_angle = float(np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle))

        return nominal_speed, steering_angle, curvature

    def plan(self, pose_x: float, pose_y: float, pose_theta: float):
        """
        Returns:
            speed, steering_angle, info_dict
        """
        position = np.array([pose_x, pose_y], dtype=np.float32)

        preview_wp = self._get_current_waypoint(position, self.max_look_ahead)
        if preview_wp is None:
            return 0.0, 0.0, {
                "lookahead": self.max_look_ahead,
                "curvature": 0.0,
                "target_point": None,
            }

        _, _, preview_curvature = self._get_actuation(
            pose_theta, preview_wp, position, self.max_look_ahead
        )
        turn_mag = abs(preview_curvature)

        lookahead_scale = self.get_scale(
            turn_mag,
            self.look_ahead_turn_gain,
            self.min_look_ahead_scale,
        )
        lookahead = self.max_look_ahead * lookahead_scale

        target_wp = self._get_current_waypoint(position, lookahead)
        if target_wp is None:
            return 0.0, 0.0, {
                "lookahead": lookahead,
                "curvature": 0.0,
                "target_point": None,
            }

        nominal_speed, steering_angle, curvature = self._get_actuation(
            pose_theta,
            target_wp,
            position,
            lookahead,
        )

        speed_scale = self.get_scale(
            abs(curvature),
            self.speed_turn_gain,
            self.min_speed_scale,
        )

        # if waypoint file has its own speed column, scale it; otherwise max_speed is the nominal
        base_speed = min(nominal_speed, self.max_speed)
        speed = base_speed * speed_scale
        speed = float(np.clip(speed, 0.0, self.max_speed))

        return speed, steering_angle, {
            "lookahead": lookahead,
            "curvature": curvature,
            "target_point": target_wp[:2].copy(),
        }