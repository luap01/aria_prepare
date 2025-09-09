#!/usr/bin/env python3
import numpy as np
import cv2
from pathlib import Path
import re
import sys
import glob

sys.path.append(str(Path(__file__).resolve().parents[1]))
from camera import load_cam_infos  # your function

# --------- PATHS / CONFIG ----------
ARIA_HAND_DIR   = Path("../data/20250519_Testing/Aria/export/hand")
ORBBEC_IMG_DIR  = Path("../data/20250519_Testing/Marshall/export/color")
# ORBBEC_3D_DIR   = Path("../HaMuCo/test_set_result_ablation_no_aria/2025-08-23_23:48/3d_kps/20250519")  # dir with <fid>.0_cam1.0_{left,right}_3d_keypoints.npz
ORBBEC_3D_DIR   = Path("../HaMuCo/misc_deprecated/test_set_result/2025-08-20_22:37/3d_kps/20250519")  # dir with <fid>.0_cam1.0_{left,right}_3d_keypoints.npz
# ORBBEC_3D_DIR   = Path("orbbec_3d_points/20250519")  # dir with <fid>.0_cam1.0_{left,right}_3d_keypoints.npz
CAM_TAG         = "camera05"
CALIB_PATH      = Path(f"../HaMuCo/data/OR/calib/20250519/{CAM_TAG}.json")

OUT_DIR         = Path("./output_orbbec_cv_20250519")
IMG_DIR         = OUT_DIR / "frames"
FONT            = cv2.FONT_HERSHEY_SIMPLEX

# Video / frame limit
MAX_FRAMES = 720
FPS        = 30
FOURCC     = "mp4v"
VIDEO_PATH = OUT_DIR / f"overlay_{CAM_TAG}.mp4"

# Units: if Orbbec 3D is in mm, set to 0.001
SCALE_ORBBEC = 1.0

# ---------- World transform ----------
# Orbbec <- Aria (used to move Aria-world 3D into Orbbec world before projecting)
T_orbbec_world_aria = np.array(
    [[-0.4681186142558247, 0.8835560378220539, -0.013917291956442377, 0.06874350185616492],
     [-0.01220315975679898, 0.009284221034147984,  0.9998824361552404, 0.38641361546505054],
     [ 0.8835813747919075, 0.46823341536885565,   0.0064360587986766835, 0.19066101889192952],
     [ 0., 0., 0., 1. ]], dtype=np.float32
)

T_orbbec_world_aria = np.array([[-0.3977413223288592, 0.9174908459438001, -0.00351967632319674, 0.020414896278126787],
 [0.02202937620729426, 0.013384874720551236, 0.9996677206515334, 0.2726872920512715],
 [0.9172330931099286, 0.39753162482756593, -0.025535468782602863, 0.1321804072024521],
 [0, 0, 0, 1]], dtype=np.float32)

# T_orbbec_world_aria = np.array([[-0.40140098572185834, 0.9156079568182105, 0.023223222702779843, 0.08445152223701442],
#  [0.032415273158488904, -0.011138014515009774, 0.999412424727011, 0.3968557186255029],
#  [0.9153286288145814, 0.4019179195356257, -0.0252088720210387, 0.2046510145725901],
#  [0, 0, 0, 1]], dtype=np.float32)

T_orbbec_world_aria = np.array([[-0.39774131774902344, 0.9174908399581909, -0.0035196763928979635, 0.020414896309375763],
 [0.022029375657439232, 0.013384874910116196, 0.9996677041053772, 0.27268728613853455],
 [0.9172331094741821, 0.3975316286087036, -0.025535468012094498, 0.12468025833368301],
 [0, 0, 0, 1]], dtype=np.float32)

T_orbbec_world_aria = np.array([[-0.39774131774902344, 0.9174908399581909, -0.0035196763928979635, -0.002558484673500061],
 [0.022029375657439232, 0.013384874910116196, 0.9996677041053772, 0.27268728613853455],
 [0.9172331094741821, 0.3975316286087036, -0.025535468012094498, 0.12468025833368301],
 [0, 0, 0, 1]], dtype=np.float32)

T_orbbec_world_aria = np.array([[-0.39774131774902344, 0.9174908399581909, -0.0035196763928979635, -0.003527136752381921],
 [0.022029375657439232, 0.013384874910116196, 0.9996677041053772, 0.27268728613853455],
 [0.9172331094741821, 0.3975316286087036, -0.025535468012094498, 0.12015814334154129],
 [0, 0, 0, 1]], dtype=np.float32)

T_orbbec_world_aria = np.array(
 [[-0.39774132,  0.91749084, -0.00351968, -0.00335115],
 [ 0.02202938,  0.01338487,  0.9996677,   0.2227039 ],
 [ 0.9172331,   0.39753163, -0.02553547,  0.02143492],
 [ 0.,          0.,          0.,          1.        ]], dtype=np.float32)

T_orbbec_world_aria = np.array(
 [[-0.39774132,  0.91749084, -0.00351968, 0.035335115],
 [ 0.02202938,  0.01338487,  0.9996677,   0.3727039 ],
 [ 0.9172331,   0.39753163, -0.02553547,  0.23143492],
 [ 0.,          0.,          0.,          1.        ]], dtype=np.float32)

T_orbbec_world_aria = np.array(
 [[-0.39774132,  0.91749084, -0.00351968, -0.00177226],
 [ 0.02202938,  0.01338487,  0.9996677,   0.24243192 ],
 [ 0.9172331,   0.39753163, -0.02553547,  0.05300744],
 [ 0.,          0.,          0.,          1.        ]], dtype=np.float32)


T_orbbec_world_aria = np.array(
 [[-0.39774132,  0.91749084, -0.00351968, -0.00561099],
 [ 0.02202938,  0.01338487,  0.9996677,   0.24455336 ],
 [ 0.9172331,   0.39753163, -0.02553547,  0.06683078],
 [ 0.,          0.,          0.,          1.        ]], dtype=np.float32)

T_orbbec_world_aria = np.array(
 [[-0.39774132,  0.91749084, -0.00351968, 0.07087158],
 [ 0.02202938,  0.01338487,  0.9996677,   0.39764744 ],
 [ 0.9172331,   0.39753163, -0.02553547,  0.20956215],
 [ 0.,          0.,          0.,          1.        ]], dtype=np.float32)


T_orbbec_world_aria = np.array(
 [[-0.39774132,  0.91749084, -0.00351968, -0.08265147 ],
 [ 0.02202938,  0.01338487,  0.9996677,   0.45190677  ],
 [ 0.9172331,   0.39753163, -0.02553547,  0.2571405   ],
 [ 0.,          0.,          0.,          1.        ]], dtype=np.float32)


# T_orbbec_world_aria = np.array(
# [[-0.4362234380106891, 0.8986148873006364, -0.04690838359823188, -0.10419404],
#  [0.02701271884376204, 0.06518370740127519, 0.9975075926077421, 0.45447868],
#  [0.8994328352638692, 0.43386906851156076, -0.05270869224433826, 0.24626365],
#  [0, 0, 0, 1]], dtype=np.float32)

# ---------- Landmark index mapping (Aria index -> Orbbec index) ----------
# If both sides share the same 21‑joint order, identity works:
ARIA_TO_ORBBEC_PAIRS = [(i, i) for i in range(21)]

# ---------- Skeleton edges (MediaPipe-style 21 joints) ----------
EDGES = [
    (0,1),(1,2),(2,3),(3,4),          # thumb
    (0,5),(5,6),(6,7),(7,8),          # index
    (0,9),(9,10),(10,11),(11,12),     # middle
    (0,13),(13,14),(14,15),(15,16),   # ring
    (0,17),(17,18),(18,19),(19,20),   # pinky
    (5,9),(9,13),(13,17)              # palm "fan"
]

# ---------- Colors / sizes ----------
# Orbbec native (solid)
COL_O_L = (0, 255, 0)     # left = green (B,G,R)
COL_O_R = (255, 0, 0)     # right = blue (B,G,R)
# Aria projected (dashed + X)
COL_A_L = (0, 165, 255)   # left = orange
COL_A_R = (255, 0, 255)   # right = magenta

R_PT_O  = 4   # Orbbec point radius
R_PT_A  = 3   # Aria point radius
TH_O    = 3   # Orbbec line thickness
TH_A    = 2   # Aria dashed thickness
DASH    = 9   # dash length in px
GAP     = 6   # gap length in px

# ---------- Helpers ----------
def inv_T(T):
    R = T[:3,:3]; t = T[:3,3]
    Ti = np.eye(4, dtype=T.dtype)
    Ti[:3,:3] = R.T
    Ti[:3,3]  = -R.T @ t
    return Ti

def project_points(M, K, P3):
    """
    M: 4x4 world->camera
    K: 3x3 intrinsics
    P3: (N,3) world points
    returns: (N,2) pixels, mask_z (N,) z>0
    """
    if P3 is None or len(P3) == 0:
        return None, None
    P = np.hstack([P3, np.ones((P3.shape[0],1), dtype=P3.dtype)])
    Xc = (M @ P.T).T[:, :3]
    z = Xc[:,2]
    mask = z > 1e-6
    uv = np.empty((P3.shape[0], 2), dtype=np.float32)
    uv[mask, 0] = K[0,0]*(Xc[mask,0]/z[mask]) + K[0,2]
    uv[mask, 1] = K[1,1]*(Xc[mask,1]/z[mask]) + K[1,2]
    uv[~mask] = np.nan
    return uv, mask

def load_orbbec_for_frame(frame_id: int):
    """
    Load Orbbec 3D left/right for a given numeric frame id.
    Expected files:
      <fid>.0_cam1.0_left_3d_keypoints.npz
      <fid>.0_cam1.0_right_3d_keypoints.npz
    """
    def find(pathpat):
        cands = glob.glob(str(ORBBEC_3D_DIR / pathpat))
        return cands[0] if cands else None

    def load_npz(p):
        if p is None: return None
        with np.load(p) as data:
            if len(data.files) == 0: return None
            X = np.array(data[data.files[0]], float)
        return X.reshape(-1,3).astype(np.float32)

    lp = find(f"{frame_id}.0_cam1.0_left_3d_keypoints.npz")
    rp = find(f"{frame_id}.0_cam1.0_right_3d_keypoints.npz")
    L = load_npz(lp); R = load_npz(rp)
    if L is not None: L *= float(SCALE_ORBBEC)
    if R is not None: R *= float(SCALE_ORBBEC)
    return L, R

def draw_points_solid(img, uv, color, r_px):
    if uv is None: return
    for pt in uv:
        if not np.all(np.isfinite(pt)): continue
        u, v = int(round(pt[0])), int(round(pt[1]))
        if u < 0 or v < 0 or u >= img.shape[1] or v >= img.shape[0]: continue
        cv2.circle(img, (u, v), r_px, color, -1, lineType=cv2.LINE_AA)

def draw_points_cross(img, uv, color, size=4, thickness=2):
    if uv is None: return
    for pt in uv:
        if not np.all(np.isfinite(pt)): continue
        u, v = int(round(pt[0])), int(round(pt[1]))
        if u < 0 or v < 0 or u >= img.shape[1] or v >= img.shape[0]: continue
        cv2.line(img, (u-size, v-size), (u+size, v+size), color, thickness, cv2.LINE_AA)
        cv2.line(img, (u-size, v+size), (u+size, v-size), color, thickness, cv2.LINE_AA)

def draw_line_dashed(img, p1, p2, color, thickness=2, dash=DASH, gap=GAP):
    # p1, p2 are tuples of ints
    x1, y1 = p1; x2, y2 = p2
    dx, dy = x2-x1, y2-y1
    dist = int(np.hypot(dx, dy))
    if dist <= 0: return
    vx, vy = dx / dist, dy / dist
    n = 0
    while n < dist:
        a = n
        b = min(n + dash, dist)
        pt1 = (int(round(x1 + vx*a)), int(round(y1 + vy*a)))
        pt2 = (int(round(x1 + vx*b)), int(round(y1 + vy*b)))
        cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)
        n += dash + gap

def draw_skeleton(img, uv, edges, color, thickness=2, dashed=False):
    if uv is None: return
    H, W = img.shape[:2]
    for i, j in edges:
        if i >= len(uv) or j >= len(uv): continue
        pi, pj = uv[i], uv[j]
        if not (np.all(np.isfinite(pi)) and np.all(np.isfinite(pj))): continue
        ui, vi = int(round(pi[0])), int(round(pi[1]))
        uj, vj = int(round(pj[0])), int(round(pj[1]))
        if not (0 <= ui < W and 0 <= vi < H and 0 <= uj < W and 0 <= vj < H): continue
        if dashed:
            draw_line_dashed(img, (ui,vi), (uj,vj), color, thickness)
        else:
            cv2.line(img, (ui,vi), (uj,vj), color, thickness, cv2.LINE_AA)

def parse_frame_token(name: str) -> int:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else None

def palm_point_from_21(X21):
    if X21 is None or X21.shape[0] < 1: return None
    idxs = [1,5,9,13,17]
    idxs = [i for i in idxs if i < X21.shape[0]]
    return np.mean(X21[idxs], axis=0) if idxs else None

def compute_rms2d(uvA, uvO, pairs):
    if uvA is None or uvO is None: return None
    H = min(len(uvA), len(uvO))
    valid_pairs = [(ia,io) for (ia,io) in pairs if ia < len(uvA) and io < len(uvO)]
    if not valid_pairs: return None
    a = np.stack([uvA[ia] for (ia,io) in valid_pairs])
    o = np.stack([uvO[io] for (ia,io) in valid_pairs])
    m = np.isfinite(a).all(1) & np.isfinite(o).all(1)
    if np.count_nonzero(m) < 4: return None
    return float(np.sqrt(np.mean(np.sum((a[m]-o[m])**2, axis=1))))


# ---------- Translation estimation (NEW) ----------
USE_PALM_ONLY   = False   # set True to use only a palm center per frame/hand
TRIM_FRAC       = 0.15    # drop the worst 15% residuals for robustness
MAX_FRAMES_FOR_T = 100000  # reuse your frame cap

def _select_pairs(A21, O21, pairs=ARIA_TO_ORBBEC_PAIRS):
    """Return matched Nx3 arrays using joint index pairs; filter non-finite."""
    if A21 is None or O21 is None:
        return None, None
    idx = [(ia, io) for (ia, io) in pairs if ia < len(A21) and io < len(O21)]
    if not idx:
        return None, None
    A = np.stack([A21[ia] for (ia, io) in idx]).astype(np.float32)
    O = np.stack([O21[io] for (ia, io) in idx]).astype(np.float32)
    m = np.isfinite(A).all(1) & np.isfinite(O).all(1)
    if not np.any(m):
        return None, None
    return A[m], O[m]

def estimate_translation_global(R_orb_from_aria, max_frames=MAX_FRAMES_FOR_T, 
                                trim_frac=TRIM_FRAC, use_palm_only=USE_PALM_ONLY):
    """
    Estimate translation t in O ≈ R A + t using robust aggregation over many frames.
    Returns (t_est [3,], stats dict) or (None, stats) if insufficient data.
    """
    diffs = []
    frames_seen = 0

    for hand_file in sorted(ARIA_HAND_DIR.glob("hand_*.npy")):
        fid = parse_frame_token(hand_file.name)
        if fid is None:
            continue

        # Load per-frame 3D
        hand = np.load(hand_file, allow_pickle=True).item()
        A_L = np.asarray(hand["left_landmarks_world"],  np.float32).reshape(-1,3)  if "left_landmarks_world"  in hand else None
        A_R = np.asarray(hand["right_landmarks_world"], np.float32).reshape(-1,3)  if "right_landmarks_world" in hand else None
        O_L, O_R = load_orbbec_for_frame(fid)

        for A, O in ((A_L, O_L), (A_R, O_R)):
            if A is None or O is None:
                continue

            if use_palm_only:
                pA = palm_point_from_21(A)
                pO = palm_point_from_21(O)
                if pA is None or pO is None:
                    continue
                # d = O - R*A  (vector form)
                d = pO - (R_orb_from_aria @ pA)
                diffs.append(d[None, :])
            else:
                A_sel, O_sel = _select_pairs(A, O)
                if A_sel is None:
                    continue
                # (R @ A_sel.T).T = A_sel @ R.T  for speed/shape convenience
                d = O_sel - (A_sel @ R_orb_from_aria.T)
                diffs.append(d)

        frames_seen += 1
        if frames_seen >= max_frames:
            break

    if not diffs:
        return None, {"used_pairs": 0, "frames": frames_seen}

    D = np.concatenate(diffs, axis=0)  # [N,3] differences

    # Robust trimming by 3D residual norm
    r = np.linalg.norm(D, axis=1)
    keep = np.ones(len(D), dtype=bool)
    if 0.0 < trim_frac < 0.5 and len(D) > 10:
        thr = np.quantile(r, 1.0 - trim_frac)
        keep = r <= thr
        D = D[keep]

    # Robust center (median) — switches easily to mean if desired
    t_est = np.median(D, axis=0).astype(np.float32)
    # Quality numbers (w.r.t. the chosen t_est)
    rmse_after = float(np.sqrt(np.mean(np.sum((D - t_est)**2, axis=1))))
    spread = D.std(axis=0)

    stats = {
        "used_pairs": int(D.shape[0]),
        "frames": frames_seen,
        "rmse_after": rmse_after,
        "std_xyz": spread,
        "trim_frac": trim_frac,
        "palm_only": use_palm_only,
        "kept_ratio": float(np.count_nonzero(keep) / max(1, len(keep)))
    }
    return t_est, stats


# ---------- Main ----------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # Intrinsics/extrinsics (we'll auto-pick direction)
    cam = load_cam_infos(CALIB_PATH)
    K_orbbec = np.asarray(cam["intrinsics"], dtype=np.float32)
    T_extr   = np.asarray(cam["extrinsics"], dtype=np.float32)  # unknown direction

    video_writer = None
    frame_count  = 0
    M_choice     = None   # Aria->Orbbec camera
    T_cw_used    = None   # Orbbec camera<-world

    # >>> NEW: estimate translation from 3D correspondences with fixed rotation
    R_fixed = T_orbbec_world_aria[:3,:3].copy()
    old_t = T_orbbec_world_aria[:3,3].copy()
    t_est, stats = estimate_translation_global(R_fixed)
    if t_est is not None:
        T_orbbec_world_aria[:3,3] = t_est
        print("[calib] Estimated translation t =", t_est, 
              f"(was {old_t}); pairs used: {stats['used_pairs']}, "
              f"frames: {stats['frames']}, kept_ratio: {stats['kept_ratio']:.2f}, "
              f"3D RMSE after: {stats['rmse_after']:.4f} (units of your data).")
    else:
        print("[calib] Not enough valid 3D pairs to estimate translation. Keeping old t:", old_t)

    hand_files = sorted(ARIA_HAND_DIR.glob("hand_*.npy"))
    for hand_file in hand_files:
        fid = parse_frame_token(hand_file.name)
        if fid is None: continue

        # Orbbec image path
        img_path = ORBBEC_IMG_DIR / f"color_{fid:06d}_{CAM_TAG.replace('05', '01').replace('06', '02')}.jpg"
        if not img_path.exists():
            found = False
            for ext in (".png", ".jpeg", ".jpg", ".JPG"):
                p2 = ORBBEC_IMG_DIR / f"color_{fid:06d}_{CAM_TAG}{ext}"
                if p2.exists(): img_path = p2; found = True; break
            if not found:
                print(f"[skip] missing image for {fid}")
                continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[skip] failed to read {img_path}")
            continue
        H, W = img.shape[:2]

        # Load Aria hand dict
        hand = np.load(hand_file, allow_pickle=True).item()
        lconf = float(hand.get("left_conf",  1.0)) if "left_conf"  in hand else 1.0
        rconf = float(hand.get("right_conf", 1.0)) if "right_conf" in hand else 1.0

        A_left_lm  = np.asarray(hand["left_landmarks_world"],  np.float32).reshape(-1,3)  if lconf>0 and "left_landmarks_world"  in hand else None
        A_right_lm = np.asarray(hand["right_landmarks_world"], np.float32).reshape(-1,3)  if rconf>0 and "right_landmarks_world" in hand else None

        # Orbbec 3D for same frame
        O_left, O_right = load_orbbec_for_frame(fid)

        # Need at least some points to decide direction
        if A_left_lm is None and A_right_lm is None:
            continue

        # Decide extrinsics direction once (pick the one with more points in front)
        if M_choice is None:
            P_any = A_left_lm if A_left_lm is not None else A_right_lm
            T_cw_1 = T_extr.copy()     # assume camera<-world
            T_cw_2 = inv_T(T_extr)     # assume world<-camera -> invert
            M1 = T_cw_1 @ T_orbbec_world_aria
            M2 = T_cw_2 @ T_orbbec_world_aria
            uv1,_ = project_points(M1, K_orbbec, P_any)
            uv2,_ = project_points(M2, K_orbbec, P_any)
            npos1 = int(np.isfinite(uv1).all(1).sum()) if uv1 is not None else 0
            npos2 = int(np.isfinite(uv2).all(1).sum()) if uv2 is not None else 0
            if npos2 > npos1:
                M_choice, T_cw_used, choice = M2, T_cw_2, "invert(extr)"
            else:
                M_choice, T_cw_used, choice = M1, T_cw_1, "as-is(extr)"
            print(f"[INFO] Extrinsics dir = {choice}  (front pts: {npos1} vs {npos2})")

        # Project: ARIA->ORBBEC camera
        uv_A_L = uv_A_R = None
        if A_left_lm is not None:
            uv_A_L,_ = project_points(M_choice, K_orbbec, A_left_lm)
        if A_right_lm is not None:
            uv_A_R,_ = project_points(M_choice, K_orbbec, A_right_lm)

        # Project: ORBBEC-native -> camera
        uv_O_L = uv_O_R = None
        if O_left is not None:
            uv_O_L,_ = project_points(T_cw_used, K_orbbec, O_left)
        if O_right is not None:
            uv_O_R,_ = project_points(T_cw_used, K_orbbec, O_right)

        # Draw skeletons
        # Orbbec solid
        # draw_skeleton(img, uv_O_L, EDGES, COL_O_L, thickness=TH_O, dashed=False)
        # draw_skeleton(img, uv_O_R, EDGES, COL_O_R, thickness=TH_O, dashed=False)
        # draw_points_solid(img, uv_O_L, COL_O_L, R_PT_O)
        # draw_points_solid(img, uv_O_R, COL_O_R, R_PT_O)

        # Aria dashed + X markers
        draw_skeleton(img, uv_A_L, EDGES, COL_A_L, thickness=TH_A, dashed=True)
        draw_skeleton(img, uv_A_R, EDGES, COL_A_R, thickness=TH_A, dashed=True)
        draw_points_cross(img, uv_A_L, COL_A_L, size=5, thickness=2)
        draw_points_cross(img, uv_A_R, COL_A_R, size=5, thickness=2)

        # Wrist/palm (optional emphasis)
        # for side, uvLM, col in (("L", uv_O_L, COL_O_L), ("R", uv_O_R, COL_O_R)):
        #     if uvLM is None or len(uvLM) < 1: continue
        #     # wrist = 0
        #     wpt = uvLM[0]
        #     if np.all(np.isfinite(wpt)):
        #         cv2.circle(img, (int(wpt[0]), int(wpt[1])), R_PT_O+2, (255,255,255), 2, cv2.LINE_AA)

        # Metrics per frame (2D RMS px) for matched indices
        rms2d_L = compute_rms2d(uv_A_L, uv_O_L, ARIA_TO_ORBBEC_PAIRS)
        rms2d_R = compute_rms2d(uv_A_R, uv_O_R, ARIA_TO_ORBBEC_PAIRS)

        y0 = 22
        legend = [
            "Orbbec = SOLID (L=green, R=blue) | Aria proj = DASHED+X (L=orange, R=magenta)"
        ]
        for line in legend:
            cv2.putText(img, line, (12, y0), FONT, 0.55, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(img, line, (12, y0), FONT, 0.55, (255,255,255), 1, cv2.LINE_AA)
            y0 += 20

        mline = []
        if rms2d_L is not None: mline.append(f"L RMS: {rms2d_L:.1f}px")
        if rms2d_R is not None: mline.append(f"R RMS: {rms2d_R:.1f}px")
        if mline:
            s = " | ".join(mline)
            cv2.putText(img, s, (12, y0), FONT, 0.65, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(img, s, (12, y0), FONT, 0.65, (0,255,255), 2, cv2.LINE_AA)

        # Save frame
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        IMG_DIR.mkdir(parents=True, exist_ok=True)
        out_path = IMG_DIR / f"overlay_orbbec_{fid:06d}_{CAM_TAG}.jpg"
        cv2.imwrite(str(out_path), img)

        # Video writer init
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*FOURCC)
            video_writer = cv2.VideoWriter(str(VIDEO_PATH), fourcc, FPS, (W, H))
            if not video_writer.isOpened():
                print(f"[error] Could not open video writer at {VIDEO_PATH}")
                return

        video_writer.write(img)
        frame_count += 1
        print(f"[ok] {img_path.name} -> {out_path}  (frame {frame_count}/{MAX_FRAMES})")

        if frame_count >= MAX_FRAMES:
            break

    if video_writer is not None:
        video_writer.release()

    if frame_count == 0:
        print("[done] No frames written; video not created.")
    else:
        print(f"[done] Saved {frame_count} frame(s). Video written to: {VIDEO_PATH}")

if __name__ == "__main__":
    main()


# python refine_se3_with_small_rotation_prior.py \
#     --X_oa_prior ./T_orbbec_world_aria_refined.npy \
#     --aria3d_dir ../data/20250519_Testing/Aria/export/hand \
#     --orbbec3d_dir orbbec_joints3d/20250519 \
#     --scale_orbbec 0.001 \
#     --rot_sigma_deg 1.0 \
#     --iters 1000 \
#     --out ./T_orbbec_world_aria_refined.npy --dof_rot 0,0,0 --dof_trans 0,0,1