import sys
import os
import numpy as np
import zmq
import json

sys.path.append("/home/aipl/GR00T-WholeBodyControl")
from gear_sonic.scripts.pico_to_g1_bridge import unpack_pose_message

context = zmq.Context()
sub_socket = context.socket(zmq.SUB)
sub_socket.connect("tcp://localhost:5556")
sub_socket.setsockopt_string(zmq.SUBSCRIBE, "pose")

print("Waiting for one ZMQ message...")
raw_msg = sub_socket.recv()
print("Received message, length:", len(raw_msg))
data = unpack_pose_message(raw_msg, topic="pose")

for k, v in data.items():
    if isinstance(v, np.ndarray):
        print(f"Key: {k:20s} Shape: {str(v.shape):20s} Min/Max: {v.min():.4f}/{v.max():.4f}")
    else:
        print(f"Key: {k:20s} Value: {v}")

# Print first few joints
if "smpl_joints" in data:
    print("\nsmpl_joints (latest frame):")
    # shape is (N, 24, 3) where N=5
    latest_joints = data["smpl_joints"][-1]
    for i in range(12):
        print(f"  Joint {i:2d}: {latest_joints[i]}")
