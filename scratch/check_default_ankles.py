import sys
import os
import numpy as np

# Add GR00T-WholeBodyControl root to PYTHONPATH
sys.path.append("/home/aipl/GR00T-WholeBodyControl")

from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model

robot_model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")
robot_model.cache_forward_kinematics(robot_model.q_zero)
T_l = robot_model.frame_placement("left_ankle_roll_link")
T_r = robot_model.frame_placement("right_ankle_roll_link")

print("Left Ankle Default:", T_l.translation)
print("Right Ankle Default:", T_r.translation)
