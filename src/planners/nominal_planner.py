import numpy as np


def load_waypoints_csv(
    csv_path: str,
    delimiter: str = ",",
    skiprows: int = 0,
    x_idx: int = 0,
    y_idx: int = 1,
    v_idx: int = None,
):
    raw = np.loadtxt(csv_path, delimiter=delimiter, skiprows=skiprows)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    if v_idx is None:
        return raw[:, [x_idx, y_idx]].astype(np.float32)

    return raw[:, [x_idx, y_idx, v_idx]].astype(np.float32)