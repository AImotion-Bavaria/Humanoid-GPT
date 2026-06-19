"""Online motion-capture retarget subprocess.

Launches a background process that reads from PNLink, Xsens MVN, or PICO,
runs GMR (General Motion Retargeting), and writes the latest qpos_full
into shared memory for the main deploy loop to consume.  Source
selection is controlled by the ``mocap_type`` string.
"""

from __future__ import annotations

import time
import atexit
import threading
import numpy as np
import multiprocessing as mp
from collections import deque
from multiprocessing.sharedctypes import SynchronizedArray


# Hand detection joint names (for PNLink)
_HAND_JOINTS = {
    "left":  {"wrist": "LeftHand"},
    "right": {"wrist": "RightHand"},
}
_HAND_THRESHOLD = 0.05

# Keep IPC primitives/process handles alive for spawn context.
# If these objects are garbage-collected too early, spawned children may fail
# rebuilding SemLock with FileNotFoundError.
_RETARGET_SESSIONS: list[dict] = []


def _detect_hand_open(frame, wrist: str, threshold: float = _HAND_THRESHOLD):
    """Detect hand open/close from finger-to-thumb distance."""
    try:
        if wrist == "RightHand":
            dist = np.linalg.norm(np.array(frame["RightHandIndex3"][0]) - np.array(frame["RightHandThumb3"][0]))
        else:
            dist = np.linalg.norm(np.array(frame["LeftHandIndex3"][0]) - np.array(frame["LeftHandThumb3"][0]))
        return dist > threshold, dist
    except KeyError:
        return False, 0.0


def _retarget_worker(
    buf, buf_hand, ts, ready_evt, stop_evt,
    robot, actual_human_height, mocap_type,
    buffer_ms, rt_pin,
    xsens_host="0.0.0.0", xsens_port=9763, xsens_protocol="tcp",
):
    """Worker process: mocap -> GMR retarget -> shared memory.

    ``rt_pin``: optional ``(cpu_id, fifo_priority)`` tuple.  When set, pin
    this process to ``cpu_id`` and run it under ``SCHED_FIFO`` at
    ``fifo_priority``.  Intended for resource-constrained on-board targets
    (e.g. Jetson via ``deploy.onboard_deploy.play_track_onboard``) where
    isolating GMR on a dedicated core measurably reduces mocap jitter.
    On general-purpose workstations leave this ``None`` — pinning a single
    core there would only contend with viewer / camera / IDE threads.
    """
    if rt_pin is not None:
        import os
        cpu_id, fifo_prio = rt_pin
        try:
            os.sched_setaffinity(0, {int(cpu_id)})
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(int(fifo_prio)))
        except (OSError, PermissionError):
            pass

    mocap_label = (mocap_type or "").lower()
    if mocap_label in ("pico", "pico_g1_bridge", "pico_bridge"):
        _pico_g1_bridge_worker(buf, buf_hand, ts, ready_evt, stop_evt)
        return
    if mocap_label in ("pico_pose_raw", "pico_raw"):
        _pico_pose_worker(buf, buf_hand, ts, ready_evt, stop_evt)
        return

    from general_motion_retargeting import GeneralMotionRetargeting as GMR

    if mocap_label == "xsens":
        from deploy.xsens.client import XsensClient
        client = XsensClient(
            host=xsens_host, port=xsens_port, protocol=xsens_protocol,
        )
        client.start_thread()
        get_frame = lambda: client.get_frame_data(timeout=0.5)
        src_human = "fbx_xsens"
    else:
        from noitom import NoitomClient
        client = NoitomClient()
        client.start_thread()
        get_frame = lambda: client.get_frame_data(timeout=True)
        src_human = "fbx_noitom"

    retarget = GMR(src_human=src_human, tgt_robot=robot, actual_human_height=actual_human_height)

    qpos_last = None
    ema_alpha = 0.75

    # -- Jitter buffer: absorb Noitom delivery timing jitter --
    _use_jbuf = buffer_ms > 0
    if _use_jbuf:
        _nominal_hz = 90.0
        _target_depth = max(1, round(buffer_ms / 1000.0 * _nominal_hz))
        _jbuf: deque[tuple[np.ndarray, np.ndarray]] = deque()
        _jbuf_lock = threading.Lock()
        _jbuf_filled = threading.Event()

        def _jitter_output():
            dt_out = 1.0 / _nominal_hz
            _out_qpos = None
            _out_hand = np.zeros(4, dtype=np.float32)
            _jbuf_filled.wait()
            if not ready_evt.is_set():
                ready_evt.set()
            while not stop_evt.is_set():
                popped = False
                with _jbuf_lock:
                    if _jbuf:
                        _out_qpos, _out_hand = _jbuf.popleft()
                        depth = len(_jbuf)
                        popped = True
                    else:
                        depth = 0
                if popped and _out_qpos is not None:
                    with buf_hand.get_lock():
                        np.frombuffer(buf_hand.get_obj(), dtype=np.float32)[:] = _out_hand
                    with buf.get_lock(), ts.get_lock():
                        np.frombuffer(buf.get_obj(), dtype=np.float32, count=_out_qpos.size)[:] = _out_qpos
                        ts.value = time.time()
                depth_err = depth - _target_depth
                dt_out = (1.0 / _nominal_hz) * (1.0 - 0.02 * depth_err)
                dt_out = max(0.005, min(0.030, dt_out))
                time.sleep(dt_out)

        threading.Thread(target=_jitter_output, daemon=True).start()
        print(f"[Retarget] Jitter buffer enabled: {buffer_ms:.0f} ms ({_target_depth} frames)")

    try:
        while not stop_evt.is_set():
            frame = get_frame()
            if frame is None:
                continue

            # Hand detection
            l_open, l_dist = _detect_hand_open(frame, **_HAND_JOINTS["left"])
            r_open, r_dist = _detect_hand_open(frame, **_HAND_JOINTS["right"])
            hand_data = np.array([float(l_open), l_dist, float(r_open), r_dist], dtype=np.float32)
            if not _use_jbuf:
                with buf_hand.get_lock():
                    np.frombuffer(buf_hand.get_obj(), dtype=np.float32)[:] = hand_data

            # Retarget
            try:
                qpos = retarget.retarget(frame)
            except Exception as e:
                import traceback
                print(f"[Retarget] error: {e}\n{traceback.format_exc()}")
                continue

            # EMA smoothing
            if qpos_last is not None:
                qpos = qpos_last * ema_alpha + qpos * (1.0 - ema_alpha)
            qpos_last = qpos.copy()
            qpos = np.asarray(qpos, dtype=np.float32)

            if _use_jbuf:
                with _jbuf_lock:
                    _jbuf.append((qpos.copy(), hand_data))
                    if not _jbuf_filled.is_set() and len(_jbuf) >= _target_depth:
                        _jbuf_filled.set()
                    while len(_jbuf) > _target_depth * 3:
                        _jbuf.popleft()
            else:
                with buf.get_lock(), ts.get_lock():
                    mv = np.frombuffer(buf.get_obj(), dtype=np.float32, count=qpos.size)
                    mv[:] = qpos
                    ts.value = time.time()

            if not _use_jbuf and not ready_evt.is_set():
                ready_evt.set()
    finally:
        if hasattr(client, "stop"):
            client.stop()


def _unpack_sonic_pose_message(packed_data: bytes, topic: str = "pose") -> dict:
    import json
    import numpy as np

    topic_bytes = topic.encode("utf-8")
    if not packed_data.startswith(topic_bytes):
        raise ValueError(f"Message does not start with topic '{topic}'")

    offset = len(topic_bytes)
    header_size = 1280
    if len(packed_data) < offset + header_size:
        raise ValueError(f"Packed PICO pose data is too small: {len(packed_data)} bytes")

    header_bytes = packed_data[offset: offset + header_size]
    null_idx = header_bytes.find(b"\x00")
    if null_idx >= 0:
        header_bytes = header_bytes[:null_idx]
    header = json.loads(header_bytes.decode("utf-8"))

    dtype_map = {
        "f32": np.float32,
        "f64": np.float64,
        "i32": np.int32,
        "i64": np.int64,
        "bool": np.bool_,
    }

    result = {"version": header.get("v", 0), "endian": header.get("endian", "le")}
    current_offset = offset + header_size
    for field in header.get("fields", []):
        dtype = dtype_map.get(field["dtype"], np.float32)
        shape = tuple(field["shape"])
        n_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        payload = packed_data[current_offset: current_offset + n_bytes]
        result[field["name"]] = np.frombuffer(payload, dtype=dtype).reshape(shape).copy()
        current_offset += n_bytes

    return result


def _pico_hand_state(data: dict) -> np.ndarray:
    import numpy as np

    left_trigger = float(np.ravel(data.get("left_trigger", [0.0]))[0])
    right_trigger = float(np.ravel(data.get("right_trigger", [0.0]))[0])
    return np.array(
        [
            1.0 if left_trigger < 0.5 else 0.0,
            left_trigger,
            1.0 if right_trigger < 0.5 else 0.0,
            right_trigger,
        ],
        dtype=np.float32,
    )


def _pico_pose_worker(buf, buf_hand, ts, ready_evt, stop_evt):
    import zmq
    import time
    import numpy as np

    print("[Retarget] Connecting to Sonic PICO POSE stream on port 5556...")
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5556")
    socket.setsockopt_string(zmq.SUBSCRIBE, "pose")
    socket.setsockopt(zmq.RCVTIMEO, 1000)

    x, y, yaw = 0.0, 0.0, 0.0
    z = 0.74
    max_linear_vel = 0.5
    last_time = None

    while not stop_evt.is_set():
        try:
            raw_msg = socket.recv()
            data = _unpack_sonic_pose_message(raw_msg, topic="pose")

            if "joint_pos" not in data:
                raise KeyError("PICO POSE packet is missing required field 'joint_pos'")

            curr_time = time.time()
            dt = 0.02 if last_time is None else curr_time - last_time
            last_time = curr_time

            if "heading_increment" in data:
                yaw += float(np.ravel(data["heading_increment"])[0])

            if "joysticks" in data:
                lx, ly = np.ravel(data["joysticks"])[:2].astype(float)
                if abs(lx) < 0.15:
                    lx = 0.0
                if abs(ly) < 0.15:
                    ly = 0.0
                v_fwd = ly * max_linear_vel
                v_strafe = -lx * max_linear_vel
                cos_y = np.cos(yaw)
                sin_y = np.sin(yaw)
                x += (cos_y * v_fwd - sin_y * v_strafe) * dt
                y += (sin_y * v_fwd + cos_y * v_strafe) * dt

            joint_pos = np.asarray(data["joint_pos"][-1], dtype=np.float32).reshape(-1)
            if joint_pos.size < 29:
                joint_pos = np.pad(joint_pos, (0, 29 - joint_pos.size))
            elif joint_pos.size > 29:
                joint_pos = joint_pos[:29]

            qpos_full = np.zeros(36, dtype=np.float32)
            qpos_full[0] = x
            qpos_full[1] = y
            qpos_full[2] = z
            qpos_full[3] = np.cos(yaw / 2.0)
            qpos_full[6] = np.sin(yaw / 2.0)
            qpos_full[7:] = joint_pos
            hand_state = _pico_hand_state(data)

            with buf.get_lock(), ts.get_lock():
                np.frombuffer(buf.get_obj(), dtype=np.float32, count=qpos_full.size)[:] = qpos_full
                ts.value = curr_time

            with buf_hand.get_lock():
                np.frombuffer(buf_hand.get_obj(), dtype=np.float32)[:] = hand_state

            if not ready_evt.is_set():
                ready_evt.set()
                print("[Retarget] Sonic PICO POSE stream active and ready.")
        except zmq.Again:
            continue
        except Exception as e:
            print(f"[Retarget] Error receiving/processing Sonic PICO POSE: {e}")
            time.sleep(0.1)


def _pico_g1_bridge_worker(buf, buf_hand, ts, ready_evt, stop_evt):
    import zmq
    import msgpack
    import time
    import numpy as np

    print("[Retarget] Connecting to PICO G1 ZMQ bridge on port 5558...")
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5558")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.setsockopt(zmq.RCVTIMEO, 1000)

    while not stop_evt.is_set():
        try:
            raw_msg = socket.recv()
            data = msgpack.unpackb(raw_msg)
            
            qpos_full = np.array(data["qpos_full"], dtype=np.float32)
            hand_state = np.array(data["hand_state"], dtype=np.float32)
            
            with buf.get_lock(), ts.get_lock():
                np.frombuffer(buf.get_obj(), dtype=np.float32, count=qpos_full.size)[:] = qpos_full
                ts.value = time.time()
                
            with buf_hand.get_lock():
                np.frombuffer(buf_hand.get_obj(), dtype=np.float32)[:] = hand_state
                
            if not ready_evt.is_set():
                ready_evt.set()
                print("[Retarget] PICO G1 bridge stream active and ready.")
        except zmq.Again:
            continue
        except Exception as e:
            print(f"[Retarget] Error receiving/processing PICO ZMQ: {e}")
            time.sleep(0.1)


def _visualize_worker(buf, stop_evt, robot="unitree_g1"):
    from general_motion_retargeting import RobotMotionViewer
    viewer = RobotMotionViewer(robot_type=robot, motion_fps=120.0)
    while not stop_evt.is_set():
        with buf.get_lock():
            qpos = np.frombuffer(buf.get_obj(), dtype=np.float32).copy()
        viewer.step(root_pos=qpos[:3], root_rot=qpos[3:7], dof_pos=qpos[7:])


def start_realtime_retarget(
    robot: str = "unitree_g1",
    dof_full: int = 36,
    actual_human_height: float = 1.6,
    visualize_retarget: bool = False,
    mocap_type: str = "pnlink",
    buffer_ms: float = 0.0,
    rt_pin: tuple[int, int] | None = None,
    xsens_host: str = "0.0.0.0",
    xsens_port: int = 9763,
    xsens_protocol: str = "tcp",
) -> tuple[SynchronizedArray, ...]:
    """Launch retarget worker and return shared buffers.

    Args:
        rt_pin: Optional ``(cpu_id, fifo_priority)`` for the GMR subprocess.
            When set, the worker pins itself to ``cpu_id`` and runs under
            ``SCHED_FIFO`` at ``fifo_priority``.  Use only on resource-
            constrained on-board targets (e.g. Jetson); leave ``None`` for
            workstation runs (``deploy/play_track.py``, ``collect_data``,
            etc.) to avoid contending with viewer / camera / IDE threads.

    Returns:
        (buf_qpos, ts, buf_hand)
        - buf_qpos: Array('f', dof_full) – latest retargeted full qpos
        - ts: Value('d') – timestamp of last update
        - buf_hand: Array('f', 4) – [left_open, left_dist, right_open, right_dist]
    """
    # Use "spawn" to avoid inheriting X11/GL state from parent process.
    # This prevents xcb thread-sequence crashes when multiple GUI viewers run.
    ctx = mp.get_context("spawn")
    buf = ctx.Array("f", dof_full, lock=True)
    buf_hand = ctx.Array("f", 4, lock=True)
    ts = ctx.Value("d", 0.0)
    ready_evt = ctx.Event()
    stop_evt = ctx.Event()

    p = ctx.Process(
        target=_retarget_worker,
        args=(buf, buf_hand, ts, ready_evt, stop_evt,
              robot, actual_human_height, mocap_type, buffer_ms, rt_pin,
              xsens_host, xsens_port, xsens_protocol),
        daemon=True,
    )
    p.start()

    vis_p = None
    if visualize_retarget and (mocap_type or "").lower() != "pico":
        vis_p = ctx.Process(target=_visualize_worker, args=(buf, stop_evt), daemon=True)
        vis_p.start()
        atexit.register(lambda: vis_p.terminate())

    # Persist references so spawn children can always rebuild synchronization
    # primitives from valid OS handles.
    _RETARGET_SESSIONS.append({
        "proc": p,
        "vis_proc": vis_p,
        "ready_evt": ready_evt,
        "stop_evt": stop_evt,
        "buf": buf,
        "buf_hand": buf_hand,
        "ts": ts,
    })

    return buf, ts, buf_hand


def read_mocap_buffer(buf, ts) -> tuple[np.ndarray, float]:
    """Read the latest qpos_full and timestamp from shared memory."""
    with buf.get_lock(), ts.get_lock():
        qpos_full = np.frombuffer(buf.get_obj(), dtype=np.float32).copy()
        timestamp = ts.value
    if np.all(qpos_full == 0):
        qpos_full[3] = 1.0
    return qpos_full, timestamp


def read_hand_buffer(buf_hand) -> tuple[bool, float, bool, float] | None:
    """Read hand open/close state from shared memory."""
    if buf_hand is None:
        return None
    with buf_hand.get_lock():
        data = np.frombuffer(buf_hand.get_obj(), dtype=np.float32).copy()
    return bool(data[0]), data[1], bool(data[2]), data[3]
