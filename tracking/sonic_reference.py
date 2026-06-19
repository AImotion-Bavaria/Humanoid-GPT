from __future__ import annotations

from pathlib import Path

import numpy as np


# SONIC documents its CSV joint columns in IsaacLab/policy order. Its deployment
# visualizer and C++ FK remap those columns with this table before writing MuJoCo
# qpos. Humanoid-GPT qpos uses the named G1 joint order, which matches that
# MuJoCo-style order.
ISAACLAB_TO_HUMANOID_GPT = np.array(
    [
        0,
        3,
        6,
        9,
        13,
        17,
        1,
        4,
        7,
        10,
        14,
        18,
        2,
        5,
        8,
        11,
        15,
        19,
        21,
        23,
        25,
        27,
        12,
        16,
        20,
        22,
        24,
        26,
        28,
    ],
    dtype=np.int32,
)

DEFAULT_ROOT_Z = 0.78
VALID_JOINT_ORDERS = ("isaaclab", "mujoco", "humanoid-gpt")


def is_sonic_motion_dir(path: str | Path) -> bool:
    motion_dir = Path(path)
    return (
        motion_dir.is_dir()
        and (motion_dir / "joint_pos.csv").is_file()
        and (motion_dir / "body_quat.csv").is_file()
    )


def iter_sonic_motion_dirs(path: str | Path, recursive: bool = False) -> list[Path]:
    root = Path(path)
    if is_sonic_motion_dir(root):
        return [root]
    if not root.is_dir():
        return []

    if recursive:
        candidates = root.rglob("joint_pos.csv")
        return sorted(
            p.parent for p in candidates if is_sonic_motion_dir(p.parent)
        )

    return sorted(
        child for child in root.iterdir() if is_sonic_motion_dir(child)
    )


def load_sonic_motion_dir(
    motion_dir: str | Path,
    frequency: float = 50.0,
    joint_order: str = "isaaclab",
    rebase_root_xy: bool = True,
    fill_zero_root_height: bool = True,
    default_root_z: float = DEFAULT_ROOT_Z,
    zero_eps: float = 1e-6,
) -> dict[str, np.ndarray]:
    motion_dir = Path(motion_dir)
    joint_order = joint_order.lower()
    if joint_order not in VALID_JOINT_ORDERS:
        valid = ", ".join(VALID_JOINT_ORDERS)
        raise ValueError(f"joint_order must be one of: {valid}")

    joint_pos = _load_csv_matrix(motion_dir / "joint_pos.csv")
    body_quat = _load_csv_matrix(motion_dir / "body_quat.csv")
    body_pos = _load_csv_matrix(motion_dir / "body_pos.csv", required=False)
    joint_vel = _load_csv_matrix(motion_dir / "joint_vel.csv", required=False)

    _validate_columns(joint_pos, 29, motion_dir / "joint_pos.csv")
    _validate_columns(body_quat, 4, motion_dir / "body_quat.csv")
    if body_pos is not None:
        _validate_columns(body_pos, 3, motion_dir / "body_pos.csv")
    if joint_vel is not None:
        _validate_columns(joint_vel, 29, motion_dir / "joint_vel.csv")

    frame_count = len(joint_pos)
    for name, array in (
        ("body_quat.csv", body_quat),
        ("body_pos.csv", body_pos),
        ("joint_vel.csv", joint_vel),
    ):
        if array is not None and len(array) != frame_count:
            raise ValueError(
                f"{motion_dir}: {name} has {len(array)} frames, "
                f"expected {frame_count}."
            )

    root_pos = (
        np.zeros((frame_count, 3), dtype=np.float32)
        if body_pos is None
        else np.asarray(body_pos[:, :3], dtype=np.float32).copy()
    )
    root_height_filled = False
    if fill_zero_root_height and np.max(np.abs(root_pos[:, 2])) <= zero_eps:
        root_pos[:, 2] = default_root_z
        root_height_filled = True
    if rebase_root_xy:
        root_pos[:, :2] -= root_pos[0:1, :2]

    root_rot = _normalize_quat_wxyz(body_quat[:, :4])
    dof_pos = _convert_joint_order(joint_pos, joint_order)
    if joint_vel is None:
        dof_vel = _finite_difference(dof_pos, frequency)
    else:
        dof_vel = _convert_joint_order(joint_vel, joint_order)

    qpos = np.concatenate([root_pos, root_rot, dof_pos], axis=1).astype(np.float32)
    qvel = np.zeros((frame_count, 35), dtype=np.float32)
    qvel[:, :3] = _finite_difference(root_pos, frequency)
    qvel[:, 3:6] = _angular_velocity_wxyz(root_rot, frequency)
    qvel[:, 6:] = dof_vel

    return {
        "qpos": qpos,
        "qvel": qvel,
        "root_pos": root_pos.astype(np.float32),
        "root_rot": root_rot.astype(np.float32),
        "dof_pos": dof_pos.astype(np.float32),
        "dof_vel": dof_vel.astype(np.float32),
        "frequency": np.asarray(frequency, dtype=np.float32),
        "source_path": np.asarray(str(motion_dir)),
        "source_format": np.asarray("sonic_reference_csv"),
        "source_joint_order": np.asarray(joint_order),
        "root_xy_rebased": np.asarray(rebase_root_xy),
        "root_height_filled": np.asarray(root_height_filled),
    }


def save_sonic_motion_npz(
    motion_dir: str | Path,
    output_path: str | Path,
    **kwargs,
) -> dict[str, np.ndarray]:
    data = load_sonic_motion_dir(motion_dir, **kwargs)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **data)
    return data


def _load_csv_matrix(path: Path, required: bool = True) -> np.ndarray | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def _validate_columns(array: np.ndarray, divisor: int, path: Path) -> None:
    if array.ndim != 2 or array.shape[1] < divisor or array.shape[1] % divisor != 0:
        raise ValueError(
            f"{path} has shape {array.shape}; expected columns to be a "
            f"positive multiple of {divisor}."
        )


def _convert_joint_order(joints: np.ndarray, joint_order: str) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float32)
    if joint_order == "isaaclab":
        return joints[:, ISAACLAB_TO_HUMANOID_GPT].copy()
    return joints.copy()


def _finite_difference(values: np.ndarray, frequency: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    velocity = np.zeros_like(values)
    if len(values) > 1:
        velocity[1:] = np.diff(values, axis=0) * float(frequency)
        velocity[0] = velocity[1]
    return velocity


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    quat /= np.clip(norm, 1e-8, None)
    for i in range(1, len(quat)):
        if np.dot(quat[i - 1], quat[i]) < 0.0:
            quat[i] = -quat[i]
    return quat


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def _angular_velocity_wxyz(quat: np.ndarray, frequency: float) -> np.ndarray:
    omega = np.zeros((len(quat), 3), dtype=np.float32)
    if len(quat) <= 1:
        return omega

    for i in range(1, len(quat)):
        q_prev = quat[i - 1]
        q_curr = quat[i]
        q_prev_inv = np.array(
            [q_prev[0], -q_prev[1], -q_prev[2], -q_prev[3]],
            dtype=np.float32,
        )
        rel = _quat_mul_wxyz(q_curr, q_prev_inv)
        if rel[0] < 0.0:
            rel = -rel
        rel /= np.clip(np.linalg.norm(rel), 1e-8, None)

        axis_norm = np.linalg.norm(rel[1:])
        if axis_norm > 1e-8:
            angle = 2.0 * np.arctan2(axis_norm, rel[0])
            omega[i] = rel[1:] / axis_norm * angle * float(frequency)

    omega[0] = omega[1]
    return omega
