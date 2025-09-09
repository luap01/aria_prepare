#!/usr/bin/env python3
import numpy as np

# ---- your current Orbbec <- Aria transform (meters) ----
T_oa = np.array([
    [ 0.0013263173,  0.9999822974, -0.0058075166,  0.0466419496],
    [-0.0103721004,  0.0058209659,  0.9999292493,  0.2683310509],
    [ 0.9999453425, -0.0012659986,  0.0103796376,  0.1198298261],
    [ 0.0,           0.0,           0.0,           1.0          ]
], dtype=np.float32)

T_oa = np.array([[-0.39774131774902344, 0.9174908399581909, -0.0035196763928979635, -0.003527136752381921],
 [0.022029375657439232, 0.013384874910116196, 0.9996677041053772, 0.27268728613853455],
 [0.9172331094741821, 0.3975316286087036, -0.025535468012094498, 0.12015814334154129],
 [0, 0, 0, 1]], dtype=np.float32)

R = T_oa[:3,:3].astype(float)
t = T_oa[:3,3].astype(float)

# --- 1) LOWER the axes by 'lower_m' meters along physical Up (i.e., -R[:,2]) ---
lower_m = 0.05  # e.g., 5 cm down toward the table
up_O = R[:,2] / np.linalg.norm(R[:,2])          # ~ (+Y_O)
t = t - lower_m * up_O

# --- 2a) Move 'away from human' by 'away_m' meters using data-driven direction ---
# If you have a bunch of Orbbec 3D hand points (Nx3 in Orbbec world), put them in P_hands_O:
P_hands_O = None  # replace with your Nx3 array if you want a data-driven 'away' direction
away_m = 0.10     # e.g., 10 cm further away

if P_hands_O is not None and len(P_hands_O) >= 5:
    origin_O = T_oa[:3,3]
    dir_to_human = P_hands_O.mean(axis=0) - origin_O
    n = np.linalg.norm(dir_to_human)
    if n > 1e-9:
        dir_to_human /= n
        t = t - away_m * dir_to_human    # subtract to move in the opposite direction (away)
else:
    # --- 2b) Or choose an axis manually in Orbbec world and nudge along it ---
    # Pick ONE of these based on what you see in the overlay:
    ex_O = np.array([1,0,0], float)   # +X_O
    ey_O = np.array([0,1,0], float)   # +Y_O (up)
    ez_O = np.array([0,0,1], float)   # +Z_O
    # Example: move 10 cm along -Z_O (i.e., further from the subject if they sit toward +Z_O):
    t = t + (-away_m) * ez_O          # change sign to match what you see in your image

# --- write back ---
T_oa_new = T_oa.copy()
T_oa_new[:3,3] = t

# If you also keep the inverse around (Aria <- Orbbec), recompute it:
def inv4x4(T):
    R, tt = T[:3,:3], T[:3,3]
    Ti = np.eye(4, dtype=T.dtype)
    Ti[:3,:3] = R.T
    Ti[:3,3]  = -R.T @ tt
    return Ti

T_ao_new = inv4x4(T_oa_new)

print("T_orbbec_world_aria (updated):\n", T_oa_new)
print("T_aria_world_orbbec (updated):\n", T_ao_new)
# np.save("T_orbbec_world_aria_nudged.npy", T_oa_new)
# np.save("T_aria_world_orbbec_nudged.npy", T_ao_new)
