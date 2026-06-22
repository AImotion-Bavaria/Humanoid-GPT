import numpy as np
from scipy.spatial.transform import Rotation as R

R_smpl2g1 = np.array([
    [0.0,  0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0,  1.0, 0.0]
])

def smpl_to_g1_rot(r_smpl):
    m_smpl = r_smpl.as_matrix()
    m_g1 = R_smpl2g1 @ m_smpl @ R_smpl2g1.T
    return R.from_matrix(m_g1)

# Test with zero rotation
r_zero = R.from_rotvec([0, 0, 0])
r_g1 = smpl_to_g1_rot(r_zero)
euler = r_g1.as_euler('YXZ', degrees=False)
print("Zero rotation output:", euler)

# Test with knee bend (rotation around X in SMPL frame, which is Y in G1 frame)
r_knee = R.from_rotvec([0.5, 0, 0])  # Rotate 0.5 rad around X in SMPL
r_knee_g1 = smpl_to_g1_rot(r_knee)
euler_knee = r_knee_g1.as_euler('YXZ', degrees=False)
print("Knee bend output (Pitch, Roll, Yaw):", euler_knee)
