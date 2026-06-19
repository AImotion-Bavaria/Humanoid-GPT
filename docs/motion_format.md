# Humanoid-GPT Motion Format and SONIC Reference Notes

This document explains the motion data expected by Humanoid-GPT, the Unitree G1
joint order used by the controller, and how SONIC/GROOT reference CSV motions are
loaded for simulation or real-robot playback.

## Short Version

Humanoid-GPT's native motion input is a NumPy `.npz` file, not CSV.

The most direct raw trajectory format is:

```text
qpos:      float32, shape (T, 36)
frequency: float or scalar array, optional, default 50 Hz
```

where each `qpos` row is:

```text
[root_x, root_y, root_z,
 root_qw, root_qx, root_qy, root_qz,
 joint_0, ..., joint_28]
```

The split equivalent is also accepted:

```text
root_pos: float32, shape (T, 3)
root_rot: float32, shape (T, 4), quaternion order w, x, y, z
dof_pos:  float32, shape (T, 29)
```

The deployment code can now also read SONIC reference folders directly:

```bash
cd /home/aipl/Humanoid-GPT
conda activate h-gpt

python -m deploy.play_track \
  --track-dir /home/aipl/GR00T-WholeBodyControl/gear_sonic_deploy/reference/teleop \
  --no-mocap
```

This direct path reads the SONIC CSVs in memory and does not modify the SONIC
folder.

## Where Motion Files Are Consumed

Humanoid-GPT has two main motion-loading paths.

### Offline Inference and Evaluation

Entry point:

```bash
python -m scripts.inference \
  --load_path storage/ckpts/pns_wo_priv216.onnx \
  --mocap_path <file_or_folder> \
  --convert
```

Relevant file:

```text
scripts/inference.py
```

This path accepts `.npz` files. With `--convert`, raw `qpos` data is converted
into keypoint reference data before tracking.

### Deployment Playback

Entry point:

```bash
python -m deploy.play_track --track-dir <file_or_folder>
```

Relevant file:

```text
deploy/play_track.py
```

This path loads offline reference motions and converts them to the policy
keypoint representation in memory. It accepts:

- Humanoid-GPT `.npz` files.
- A folder containing `.npz` files.
- A SONIC reference motion folder containing CSV files.
- A SONIC reference base folder containing multiple motion subfolders.

## Native Raw `.npz` Format

### Required Option A: `qpos`

```text
qpos: float32, shape (T, 36)
```

Each row:

```text
index 0:3   root position in world frame, meters: x, y, z
index 3:7   root quaternion in world frame: w, x, y, z
index 7:36  29 G1 joint angles, radians, Humanoid-GPT joint order
```

Example:

```python
import numpy as np

np.savez_compressed(
    "my_motion.npz",
    qpos=qpos.astype(np.float32),
    frequency=np.asarray(50, dtype=np.float32),
)
```

### Required Option B: Split Root and Joint Fields

Instead of `qpos`, a file can contain:

```text
root_pos: float32, shape (T, 3)
root_rot: float32, shape (T, 4), wxyz quaternion
dof_pos:  float32, shape (T, 29)
```

The loader concatenates them into:

```python
qpos = np.concatenate([root_pos, root_rot, dof_pos], axis=1)
```

### Optional Raw Fields

```text
qvel:      float32, shape (T, 35)
frequency: scalar, source frequency in Hz
fps:       scalar, legacy frequency field
```

`qvel` row layout:

```text
index 0:3   root linear velocity
index 3:6   root angular velocity
index 6:35  29 joint velocities
```

If `qvel` is absent, the converter can recompute velocities from `qpos`.

## Keypoint `.npz` Format

The tracking policy does not only use joint targets. It also needs keypoint
poses and velocities produced by MuJoCo forward kinematics.

The keypoint representation contains:

```text
qpos:            float32, shape (T, 36)
qvel:            float32, shape (T, 35)
kpt2gv_pose:     float32, shape (T, 14, 4, 4)
kpt_cvel_in_gv:  float32, shape (T, 14, 6)
gv_vel:          float32, shape (T, 3)
gv2wrd_pose:     float32, shape (T, 4, 4)
foot_contact:    float32/bool, shape (T, 2)
```

Normally you do not hand-author these fields. Generate them from raw `qpos`:

```bash
cd /home/aipl/Humanoid-GPT
conda activate h-gpt

python tracking/convert_qpos2kpt.py \
  --mocap_npz my_motion.npz \
  --save_path my_motion_kpt.npz
```

In deployment, this conversion is usually done in memory by `deploy.play_track`.

## Humanoid-GPT 29-DOF Joint Order

Humanoid-GPT uses the order in `tracking/constants.py`,
`ACTION_JOINT_NAMES`.

This is also the order expected in raw `qpos[:, 7:]` and `dof_pos`.

| Index | Joint name |
| ---: | --- |
| 0 | `left_hip_pitch_joint` |
| 1 | `left_hip_roll_joint` |
| 2 | `left_hip_yaw_joint` |
| 3 | `left_knee_joint` |
| 4 | `left_ankle_pitch_joint` |
| 5 | `left_ankle_roll_joint` |
| 6 | `right_hip_pitch_joint` |
| 7 | `right_hip_roll_joint` |
| 8 | `right_hip_yaw_joint` |
| 9 | `right_knee_joint` |
| 10 | `right_ankle_pitch_joint` |
| 11 | `right_ankle_roll_joint` |
| 12 | `waist_yaw_joint` |
| 13 | `waist_roll_joint` |
| 14 | `waist_pitch_joint` |
| 15 | `left_shoulder_pitch_joint` |
| 16 | `left_shoulder_roll_joint` |
| 17 | `left_shoulder_yaw_joint` |
| 18 | `left_elbow_joint` |
| 19 | `left_wrist_roll_joint` |
| 20 | `left_wrist_pitch_joint` |
| 21 | `left_wrist_yaw_joint` |
| 22 | `right_shoulder_pitch_joint` |
| 23 | `right_shoulder_roll_joint` |
| 24 | `right_shoulder_yaw_joint` |
| 25 | `right_elbow_joint` |
| 26 | `right_wrist_roll_joint` |
| 27 | `right_wrist_pitch_joint` |
| 28 | `right_wrist_yaw_joint` |

## Root Pose Conventions

Root position:

```text
root_pos[:, 0] = x, meters
root_pos[:, 1] = y, meters
root_pos[:, 2] = z, meters
```

Root quaternion:

```text
root_rot[:, 0] = qw
root_rot[:, 1] = qx
root_rot[:, 2] = qy
root_rot[:, 3] = qz
```

Important notes:

- Quaternion order is `w, x, y, z`.
- Root position is in world frame.
- Joint angles are radians.
- Motions are normally 50 Hz.
- A standing G1 root height is about `0.78`.

If root `z` is accidentally all zeros, the controller receives a bad height
command. The SONIC loader fills zero-only root height with `0.78` by default.

## SONIC/GROOT CSV Reference Format

SONIC reference motions are stored as one folder per clip:

```text
motion_name/
  joint_pos.csv
  joint_vel.csv
  body_pos.csv
  body_quat.csv
  metadata.txt
```

Minimum useful files:

```text
joint_pos.csv
body_quat.csv
```

Better files:

```text
joint_pos.csv
joint_vel.csv
body_pos.csv
body_quat.csv
metadata.txt
```

### SONIC `joint_pos.csv`

Shape:

```text
(T, 29)
```

Units:

```text
radians
```

The SONIC documentation describes this as IsaacLab/policy order. Before using
the values as Humanoid-GPT qpos, the adapter remaps them.

Default SONIC-to-Humanoid-GPT index mapping:

```python
ISAACLAB_TO_HUMANOID_GPT = [
    0, 3, 6, 9, 13, 17,
    1, 4, 7, 10, 14, 18,
    2, 5, 8,
    11, 15, 19, 21, 23, 25, 27,
    12, 16, 20, 22, 24, 26, 28,
]
```

Meaning:

```python
humanoid_gpt_dof_pos = sonic_joint_pos[:, ISAACLAB_TO_HUMANOID_GPT]
```

If you later export SONIC CSVs already in Humanoid-GPT/MuJoCo joint order, pass:

```bash
--sonic-joint-order humanoid-gpt
```

or for the converter:

```bash
--joint-order humanoid-gpt
```

### SONIC `joint_vel.csv`

Shape:

```text
(T, 29)
```

Units:

```text
rad/s
```

If missing, the adapter computes finite-difference joint velocities.

### SONIC `body_pos.csv`

Each body has 3 columns:

```text
body_0_x, body_0_y, body_0_z, body_1_x, ...
```

Humanoid-GPT direct playback only uses body group 0 as the root:

```python
root_pos = body_pos[:, 0:3]
```

If `root_pos[:, 2]` is all zero, the adapter fills it with `0.78`.

### SONIC `body_quat.csv`

Each body has 4 columns:

```text
body_0_w, body_0_x, body_0_y, body_0_z, body_1_w, ...
```

Humanoid-GPT direct playback only uses body group 0 as the root:

```python
root_rot = body_quat[:, 0:4]
```

Quaternion order is already `w, x, y, z`, matching Humanoid-GPT.

## Running SONIC Motions in Humanoid-GPT

### Direct In-Memory Playback

This is the easiest path for new SONIC recordings.

```bash
cd /home/aipl/Humanoid-GPT
conda activate h-gpt

python -m deploy.play_track \
  --track-dir /home/aipl/GR00T-WholeBodyControl/gear_sonic_deploy/reference/teleop \
  --no-mocap
```

For real robot:

```bash
cd /home/aipl/Humanoid-GPT
conda activate h-gpt

mkdir -p /tmp/cyclonedds_unitree/lib
ln -sf /home/aipl/unitree_sdk2/thirdparty/lib/x86_64/libddsc.so \
  /tmp/cyclonedds_unitree/lib/libddsc.so

CYCLONEDDS_HOME=/tmp/cyclonedds_unitree \
LD_LIBRARY_PATH=/home/aipl/unitree_sdk2/thirdparty/lib/x86_64:$LD_LIBRARY_PATH \
python -m deploy.play_track \
  --real \
  --net <your_robot_nic_name> \
  --track-dir /home/aipl/GR00T-WholeBodyControl/gear_sonic_deploy/reference/teleop \
  --no-mocap
```

On the current lab machine, the working interface has been:

```text
enx144fd7d9dbe1
```

The SONIC folder is read-only in this path.

Restart `deploy.play_track` after recording new SONIC clips so the folder is
rescanned.

### Slower Playback for Robot Tests

Use `--motion-speed` to slow or speed up offline reference playback without
changing the robot control loop frequency.

Recommended first real-robot test:

```bash
CYCLONEDDS_HOME=/tmp/cyclonedds_unitree \
LD_LIBRARY_PATH=/home/aipl/unitree_sdk2/thirdparty/lib/x86_64:$LD_LIBRARY_PATH \
python -m deploy.play_track \
  --real \
  --net enx144fd7d9dbe1 \
  --track-dir /home/aipl/GR00T-WholeBodyControl/gear_sonic_deploy/reference/teleop \
  --no-mocap \
  --motion-speed 0.5
```

Speed examples:

```text
--motion-speed 1.0   normal recorded speed
--motion-speed 0.5   half speed
--motion-speed 0.25  quarter speed
```

Do not use a slower `--freq` for safety testing unless you understand the
control implications. `--freq` changes the controller loop frequency;
`--motion-speed` only stretches the reference motion.

### Convert SONIC CSVs to Humanoid-GPT `.npz`

Use this when you want persistent copies under Humanoid-GPT:

```bash
cd /home/aipl/Humanoid-GPT
conda activate h-gpt

python -m scripts.convert_sonic_reference \
  /home/aipl/GR00T-WholeBodyControl/gear_sonic_deploy/reference/teleop \
  --output-dir storage/sonic_reference/teleop
```

Then play the converted copies:

```bash
python -m deploy.play_track \
  --track-dir storage/sonic_reference/teleop \
  --no-mocap
```

Or test one clip headless:

```bash
python -m scripts.inference \
  --load_path storage/ckpts/pns_wo_priv216.onnx \
  --mocap_path storage/sonic_reference/teleop/<motion_name>.npz \
  --headless \
  --convert \
  --freq 50
```

## Mode Numbers in `deploy.play_track`

Keyboard mode mapping:

```text
0 = walking policy
1 = online retarget mode
2 = first offline motion
3 = second offline motion
...
```

Offline motion order is sorted by filename/folder name at startup.

## Development Checklist for New Motion Sources

When adding a new source, first convert it to this raw format:

```text
qpos: shape (T, 36)
frequency: 50
```

Check:

- `qpos[:, 0:3]` is root world position in meters.
- `qpos[:, 2]` is near G1 root height, not all zero.
- `qpos[:, 3:7]` is normalized quaternion in `wxyz`.
- `qpos[:, 7:]` has 29 joints in Humanoid-GPT order.
- Joint angles are radians.
- Frequency is correct.
- No NaN or inf values.
- Motion starts from a physically reasonable pose.

Then run:

```bash
python -m scripts.inference \
  --load_path storage/ckpts/pns_wo_priv216.onnx \
  --mocap_path <your_motion.npz> \
  --headless \
  --convert \
  --freq 50
```

For visual inspection:

```bash
python -m scripts.vis --path <your_motion.npz>
```

If the robot looks twisted or mirrored, the first suspect is joint order.

If the robot crouches, floats, or falls immediately, inspect root height and
root quaternion.

If translation behaves strangely, inspect root x/y and whether the first frame
should be rebased to zero.

## Current Limitation

The SONIC direct loader uses 29-DOF body joints only. `hand_pos.csv` is not used
by this Humanoid-GPT tracking policy path.

## Safety Follow-Ups

Before serious free-standing real-robot tests, the deployment path should be
reviewed for safety. In particular:

- Add motor temperature monitoring from `LowState.motor_state[*].temperature`.
- Add warning and stop thresholds for hot motors.
- Add tracking-error stop conditions for large root/joint divergence.
- Start new motions at reduced playback speed, for example `--motion-speed 0.5`.
- Test new reference folders in simulation first, then on a suspended robot.

The example SONIC folder contains dynamic motions such as kicks, lunges,
jumping, and dances. Treat those as higher risk than the teleop folder.
