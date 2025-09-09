#!/usr/bin/env python3
"""
Calibrate the rigid transform between Orbbec world and Aria world via:
  1) Per-frame PnP:   {}^O T_C(t)  (Aria camera pose in Orbbec world)
  2) Hand-eye solve:  X = {}^A T_O minimizing {}^O T_C(t) ≈ X^{-1} {}^A T_C(t)

Latest file layout & rules:
- 3D (Orbbec world, meters): j3d_dir / "<fid>.0_cam1.0_left_3d_keypoints.npz"
                             j3d_dir / "<fid>.0_cam1.0_right_3d_keypoints.npz"
  Each .npz holds a single array [21,3] (we load the first/only array key).
- 2D (Aria UNDISTORTED image; often rotated 90°): kp2d_dir / "<fid>.json"
    {
      "people":[{
        "hand_left_keypoints_2d":[x0,y0,z0, ... len=63],
        "hand_left_shift":[sx,sy],
        "hand_left_conf":[float],
        "hand_right_keypoints_2d":[...],
        "hand_right_shift":[sx,sy],
        "hand_right_conf":[float]
      }]
    }
Rules:
- Add the per-hand shift to each (x,y).
- Use a hand only if its conf[0] > 0. (If the other hand is 0, we still keep the frame if one hand is valid.)
- Not all 2D files have the corresponding 3D files; those hands are ignored. If both hands end up unusable, skip the frame.

We UNROTATE keypoints (if needed) back to the UNROTATED rectified plane and use K from 'destination_intrinsics' (pinhole).

Outputs:
- <out_prefix>_T_aria_orbbec.npy   ({}^A T_O : Orbbec -> Aria)
- <out_prefix>_T_orbbec_aria.npy   ({}^O T_A : Aria -> Orbbec)
- <out_prefix>_report.json         (diagnostics)
"""

import os, re, json, glob, argparse
import numpy as np
import cv2
from scipy.optimize import least_squares

JOINTS_PER_HAND = 21

# ------------------ SE(3) utilities ------------------

def hat3(w):
    wx, wy, wz = w
    return np.array([[0, -wz,  wy],
                     [wz,  0, -wx],
                     [-wy, wx,  0]], dtype=float)

def so3_log(R):
    tr = np.trace(R)
    cos_th = (tr - 1.0) * 0.5
    cos_th = np.clip(cos_th, -1.0, 1.0)
    th = np.arccos(cos_th)
    if th < 1e-8:
        return np.zeros(3)
    w_hat = (R - R.T) * (0.5 / np.sin(th))
    return np.array([w_hat[2,1], w_hat[0,2], w_hat[1,0]]) * th

def se3_exp(xi):
    """Exponential map with normalized rotation axis.
       xi = [wx, wy, wz, tx, ty, tz]"""
    w = xi[:3]; t = xi[3:]
    T = np.eye(4)
    th = np.linalg.norm(w)
    if th < 1e-12:
        R = np.eye(3); V = np.eye(3)
    else:
        K = hat3(w / th)
        R = np.eye(3) + np.sin(th)*K + (1-np.cos(th))*(K@K)
        V = np.eye(3) + (1-np.cos(th))*K + (th-np.sin(th))*(K@K)
    T[:3,:3] = R
    T[:3,3]  = V @ t
    return T

def se3_log(T):
    R = T[:3,:3]
    t = T[:3,3]
    phi = so3_log(R)
    th = np.linalg.norm(phi)
    if th < 1e-12:
        Vinv = np.eye(3)
    else:
        K = hat3(phi / th)
        Vinv = (np.eye(3) - 0.5*K
                + (1/(th**2) - (1 + np.cos(th))/(2*th*np.sin(th))) * (K@K))
    rho = Vinv @ t
    return np.r_[phi, rho]

def inv_T(T):
    R = T[:3,:3]; t = T[:3,3]
    Ti = np.eye(4); Ti[:3,:3] = R.T; Ti[:3,3] = -R.T @ t
    return Ti

# --- extras for prior-guided PnP ---
def rot_geodesic_deg(R1, R2):
    R = R1.T @ R2
    tr = np.clip((np.trace(R)-1)*0.5, -1.0, 1.0)
    return np.degrees(np.arccos(tr))

def cluster_X_list(X_list, rot_thresh_deg=8.0, trans_thresh_m=0.10):
    """
    Simple RANSAC/medoid cluster: choose each X_i as a seed, count inliers s.t.
    rotation/translation differences are below thresholds; pick best; compute
    robust mean on the inliers (Lie algebra average).
    """
    n = len(X_list)
    if n == 0:
        return [], None
    best_inliers = []
    for i in range(n):
        Xi = X_list[i]
        Ri = Xi[:3,:3]; ti = Xi[:3,3]
        inliers = []
        for j in range(n):
            Xj = X_list[j]
            Rj = Xj[:3,:3]; tj = Xj[:3,3]
            dR = rot_geodesic_deg(Ri, Rj)
            dt = np.linalg.norm(tj - ti)
            if dR <= rot_thresh_deg and dt <= trans_thresh_m:
                inliers.append(j)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    if not best_inliers:
        return [], None

    # Robust mean on inliers via iterative log/exp average
    X_ref = X_list[best_inliers[0]].copy()
    for _ in range(10):
        xi_sum = np.zeros(6)
        for idx in best_inliers:
            E = np.linalg.inv(X_ref) @ X_list[idx]
            # se3_log
            def hat3(w):
                return np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]], float)
            # SO(3) log
            R = E[:3,:3]; t = E[:3,3]
            tr = np.clip((np.trace(R)-1)/2, -1, 1)
            th = np.arccos(tr)
            if th < 1e-12:
                phi = np.zeros(3); Vinv = np.eye(3)
            else:
                K = (R - R.T)*(0.5/np.sin(th))
                phi = np.array([K[2,1], K[0,2], K[1,0]])*th
                k = phi/np.linalg.norm(phi)
                Kk = hat3(k)
                Vinv = (np.eye(3) - 0.5*Kk
                        + (1/(th**2) - (1+np.cos(th))/(2*th*np.sin(th))) * (Kk@Kk))
            rho = Vinv @ t
            xi = np.r_[phi, rho]
            xi_sum += xi
        xi_mean = xi_sum / len(best_inliers)
        # se3_exp
        w = xi_mean[:3]; v = xi_mean[3:]
        th = np.linalg.norm(w)
        if th < 1e-12:
            R = np.eye(3); V = np.eye(3)
        else:
            k = w/th
            Kk = hat3(k)
            R = np.eye(3) + np.sin(th)*Kk + (1-np.cos(th))*(Kk@Kk)
            V = np.eye(3) + (1-np.cos(th))*Kk + (th-np.sin(th))*(Kk@Kk)
        X_ref = X_ref @ np.block([[R, (V@v).reshape(3,1)], [np.zeros((1,3)), np.ones((1,1))]])
    return best_inliers, X_ref

def pnp_pose_in_orbbec_world_prior(
    Pw_orbbec, uv, K, valid,
    Twc_A_t,            # {}^A T_C at this frame
    X_ao_prior,         # {}^A T_O prior (Aria <- Orbbec)
    pre_px=8.0,         # pixels for prior-based inlier selection
    min_inliers=8,
    accept_deg=45.0,    # reject if > this rotation diff vs prior-predicted pose
    accept_m=0.5        # reject if > this translation diff (meters)
):
    """
    Prior-guided single-frame PnP.
    1) Predict Aria camera pose in Orbbec world from prior: Twc_O_pred = X_oa_prior @ Twc_A_t
    2) Build initial rvec/tvec from Tcw_O_pred and select inliers within pre_px.
    3) Refine with LM. Reject if far from prediction.
    Fallback: RANSAC EPNP + LM, still rejected if far from prediction.
    """
    X_oa_prior = inv_T(X_ao_prior)
    Twc_O_pred = X_oa_prior @ Twc_A_t
    Tcw_O_pred = inv_T(Twc_O_pred)
    Rcw_pred = Tcw_O_pred[:3,:3]; tcw_pred = Tcw_O_pred[:3,3]
    rvec_pred, _ = cv2.Rodrigues(Rcw_pred)

    valid2 = valid & np.isfinite(uv).all(axis=1) & np.isfinite(Pw_orbbec).all(axis=1)
    obj = Pw_orbbec[valid2].astype(np.float64)
    img = uv[valid2].astype(np.float64)
    if obj.shape[0] < 4:
        return None, np.inf, np.zeros_like(valid, bool)

    dist = np.zeros(4)
    proj, _ = cv2.projectPoints(obj, rvec_pred, tcw_pred.reshape(3,1), K, dist)
    err = np.linalg.norm(proj.reshape(-1,2) - img, axis=1)
    inl = np.where(err < pre_px)[0]

    if len(inl) >= min_inliers:
        # refine from prior with prior-selected inliers
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], K, dist, rvec_pred, tcw_pred.reshape(3,1))
        Rcw,_ = cv2.Rodrigues(rvec); tcw = tvec.reshape(3)
        Tcw_O = np.eye(4); Tcw_O[:3,:3]=Rcw; Tcw_O[:3,3]=tcw
        Twc_O = inv_T(Tcw_O)

        # accept only if close to predicted pose
        dR = rot_geodesic_deg(Twc_O_pred[:3,:3], Twc_O[:3,:3])
        dt = np.linalg.norm(Twc_O[:3,3] - Twc_O_pred[:3,3])
        if dR > accept_deg or dt > accept_m:
            return None, np.inf, np.zeros_like(valid, bool)

        proj2, _ = cv2.projectPoints(obj[inl], rvec, tvec, K, dist)
        rms = float(np.sqrt(np.mean(np.sum((proj2.reshape(-1,2) - img[inl])**2, axis=1))))
        full_inlier_mask = np.zeros_like(valid, bool)
        idx_valid = np.where(valid2)[0]; full_inlier_mask[idx_valid[inl]] = True
        return Twc_O, rms, full_inlier_mask

    # fallback: RANSAC -> LM, then still gate by prior proximity
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=obj, imagePoints=img, cameraMatrix=K, distCoeffs=dist,
        flags=cv2.SOLVEPNP_EPNP, reprojectionError=4.0, iterationsCount=200, confidence=0.999
    )
    if not ok or inliers is None or len(inliers) < min_inliers:
        return None, np.inf, np.zeros_like(valid, bool)

    rvec, tvec = cv2.solvePnPRefineLM(obj[inliers[:,0]], img[inliers[:,0]], K, dist, rvec, tvec)
    Rcw,_ = cv2.Rodrigues(rvec); tcw = tvec.reshape(3)
    Tcw_O = np.eye(4); Tcw_O[:3,:3]=Rcw; Tcw_O[:3,3]=tcw
    Twc_O = inv_T(Tcw_O)

    dR = rot_geodesic_deg(Twc_O_pred[:3,:3], Twc_O[:3,:3])
    dt = np.linalg.norm(Twc_O[:3,3] - Twc_O_pred[:3,3])
    if dR > accept_deg or dt > accept_m:
        return None, np.inf, np.zeros_like(valid, bool)

    proj3, _ = cv2.projectPoints(obj[inliers[:,0]], rvec, tvec, K, dist)
    rms = float(np.sqrt(np.mean(np.sum((proj3.reshape(-1,2) - img[inliers[:,0]])**2, axis=1))))
    full_inlier_mask = np.zeros_like(valid, bool)
    idx_valid = np.where(valid2)[0]; full_inlier_mask[idx_valid[inliers[:,0]]] = True
    return Twc_O, rms, full_inlier_mask

# ------------------ Unrotation helpers ------------------

def unrotate_uv_90cw_batch(kp2d, W_rot, H_rot):
    """
    kp2d: [T,K,2] on a **90° CW rotated** rectified image with size (W_rot,H_rot).
    Mapping (rotated -> original):
        u = v_rot
        v = (W_rot - 1) - u_rot
    Original (unrotated) size is (W=H_rot, H=W_rot).
    """
    kp = kp2d.copy()
    u_r = kp[...,0]; v_r = kp[...,1]
    kp[...,0] = v_r
    kp[...,1] = (W_rot - 1) - u_r
    return kp

def unrotate_uv_90ccw_batch(kp2d, W_rot, H_rot):
    """
    kp2d: [T,K,2] on a **90° CCW rotated** rectified image with size (W_rot,H_rot).
    Mapping (rotated -> original):
        u = (H_rot - 1) - v_rot
        v = u_rot
    Original (unrotated) size is (W=H_rot, H=W_rot).
    """
    kp = kp2d.copy()
    u_r = kp[...,0]; v_r = kp[...,1]
    kp[...,0] = (H_rot - 1) - v_r
    kp[...,1] = u_r
    return kp

# ------------------ I/O helpers for your layout ------------------

JSON2D_RX = re.compile(r"^(\d+)\.json$")
NPZ3D_RX  = re.compile(r"^(\d+)\..*?(_left_3d_keypoints|_right_3d_keypoints)\.npz$")

def _load_json_allow_nan(path):
    with open(path, "r") as f:
        txt = f.read()
    txt = txt.replace("NaN", "null").replace("Infinity", "null").replace("-Infinity", "null")
    return json.loads(txt)

def _parse_2d_with_shift(arr_flat, shift_xy):
    """
    arr_flat: length 63 = 21*(x,y,third)
    shift_xy: [sx, sy] to be ADDED to every (x,y)
    Returns: uv [21,2], valid [21]
    """
    a = np.asarray(arr_flat, float).reshape(-1, 3)  # [21,3]
    uv = a[:, :2] + np.asarray(shift_xy, float).reshape(1,2)
    valid = np.isfinite(uv).all(axis=1)
    return uv, valid

def _maybe_get_conf(singleton_list, default=1.0):
    try:
        if isinstance(singleton_list, (list, tuple)) and len(singleton_list) > 0:
            return float(singleton_list[0])
        return float(singleton_list)
    except Exception:
        return float(default)

def _load_2d_frame_json_conf_gated(path_json, use_left=True, use_right=True):
    """
    Reads <fid>.json and returns dict with per-hand content ONLY IF conf>0:
      { "left": (uv[21,2], valid[21]) or None,
        "right": (uv[21,2], valid[21]) or None }
    """
    d = _load_json_allow_nan(path_json)
    people = d.get("people", [])
    if not people:
        return {"left": None, "right": None}
    p0 = people[0]

    out = {"left": None, "right": None}

    if use_left and ("hand_left_keypoints_2d" in p0) and ("hand_left_shift" in p0):
        confL = _maybe_get_conf(p0.get("hand_left_conf", [1.0]), default=1.0)
        if confL > 0.0:
            uvL, vL = _parse_2d_with_shift(p0["hand_left_keypoints_2d"], p0["hand_left_shift"])
            out["left"] = (uvL, vL)

    if use_right and ("hand_right_keypoints_2d" in p0) and ("hand_right_shift" in p0):
        confR = _maybe_get_conf(p0.get("hand_right_conf", [1.0]), default=1.0)
        if confR > 0.0:
            uvR, vR = _parse_2d_with_shift(p0["hand_right_keypoints_2d"], p0["hand_right_shift"])
            out["right"] = (uvR, vR)

    return out

def _load_npz_points(path_npz):
    """Loads the first/only array from .npz and returns as float [N,3]."""
    with np.load(path_npz) as data:
        if len(data.files) == 0:
            raise ValueError(f"{path_npz} contains no arrays.")
        key = data.files[0]
        X = np.array(data[key], dtype=float)
    X = np.asarray(X, float)
    if X.ndim == 1 and X.size % 3 == 0:
        X = X.reshape(-1,3)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError(f"{path_npz}: expected shape [N,3], got {X.shape}")
    return X

def collect_maps_npz(kp2d_dir, j3d_dir):
    """
    Returns:
      map2d[fid] -> /path/to/<fid>.json  (2D)
      map3d_left[fid]  -> /path/to/<fid>..._left_3d_keypoints.npz
      map3d_right[fid] -> /path/to/<fid>..._right_3d_keypoints.npz
    """
    map2d = {}
    for p in glob.glob(os.path.join(kp2d_dir, "*.json")):
        m = JSON2D_RX.match(os.path.basename(p))
        if m:
            fid = int(m.group(1))
            map2d[fid] = p

    map3d_left, map3d_right = {}, {}
    for p in glob.glob(os.path.join(j3d_dir, "*.npz")):
        base = os.path.basename(p)
        m = NPZ3D_RX.match(base)
        if not m:
            continue
        fid = int(m.group(1))
        if "left_3d_keypoints" in base:
            map3d_left[fid] = p
        elif "right_3d_keypoints" in base:
            map3d_right[fid] = p

    return map2d, map3d_left, map3d_right

def build_arrays_custom_npz_fixedK(kp2d_dir, j3d_dir, use_left=True, use_right=True, scale3d=1.0):
    """
    Builds fixed-shape arrays per frame:
      frame_ids: [T]
      kp2d:      [T, K, 2]   (K = JOINTS_PER_HAND * (#hands selected), left then right)
      P3D_O:     [T, K, 3]
      valid:     [T, K]      (bool)
    A hand contributes only if 2D present with conf>0 AND corresponding 3D .npz present.
    Missing hands/joints are padded with NaNs / False so shapes match across frames.
    Frames with <4 valid points are skipped.
    """
    assert use_left or use_right, "At least one of use_left/use_right must be True."

    map2d, map3d_left, map3d_right = collect_maps_npz(kp2d_dir, j3d_dir)
    fids_sorted = sorted(map2d.keys())

    hands_order = []
    if use_left:  hands_order.append("left")
    if use_right: hands_order.append("right")

    Kj = JOINTS_PER_HAND * len(hands_order)

    kp_all, X_all, valid_all, kept = [], [], [], []
    skipped_no_points = 0

    for fid in fids_sorted:
        sides2d = _load_2d_frame_json_conf_gated(map2d[fid], use_left=use_left, use_right=use_right)

        # Initialize padded containers
        uv_frame  = np.full((Kj, 2), np.nan, dtype=float)
        X_frame   = np.full((Kj, 3), np.nan, dtype=float)
        v_frame   = np.zeros((Kj,), dtype=bool)

        for h_idx, hand in enumerate(hands_order):
            idx0 = h_idx * JOINTS_PER_HAND
            idx1 = idx0 + JOINTS_PER_HAND

            if hand == "left":
                has2d = sides2d.get("left") is not None
                has3d = fid in map3d_left
                if not (has2d and has3d):
                    continue
                uv, v2d = sides2d["left"]
                try:
                    X = _load_npz_points(map3d_left[fid])
                except Exception:
                    continue
            else:  # "right"
                has2d = sides2d.get("right") is not None
                has3d = fid in map3d_right
                if not (has2d and has3d):
                    continue
                uv, v2d = sides2d["right"]
                try:
                    X = _load_npz_points(map3d_right[fid])
                except Exception:
                    continue

            num = min(JOINTS_PER_HAND, uv.shape[0], X.shape[0])
            if num < 1:
                continue

            uv_slice = uv[:num]
            X_slice  = X[:num] * float(scale3d)
            v_slice  = v2d[:num] & np.isfinite(X_slice).all(axis=1) & np.isfinite(uv_slice).all(axis=1)

            uv_frame[idx0:idx0+num] = uv_slice
            X_frame[idx0:idx0+num]  = X_slice
            v_frame[idx0:idx0+num]  = v_slice

        if np.count_nonzero(v_frame) < 4:
            skipped_no_points += 1
            continue

        kp_all.append(uv_frame)
        X_all.append(X_frame)
        valid_all.append(v_frame)
        kept.append(fid)

    if not kp_all:
        raise RuntimeError("No usable frames after applying 2D conf gating / 3D matching. "
                           "Check your inputs or relax gating.")
    if skipped_no_points:
        print(f"[INFO] Skipped {skipped_no_points} frames that had <4 valid correspondences.")

    return kept, np.stack(kp_all, 0), np.stack(X_all, 0), np.stack(valid_all, 0)

# ------------------ Calib loader (Aria) ------------------

def load_cam_T_world_and_K_from_calibs(calib_dir, frame_ids, K_key="destination_intrinsics"):
    """Returns cam_T_world [T,4,4] and a constant K (3x3) from the selected intrinsics block."""
    Ts, K_ref = [], None
    for fid in frame_ids:
        path = os.path.join(calib_dir, f"calib_{fid:06d}.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {path}. Ensure fid matches your calib filenames.")
        d = _load_json_allow_nan(path)
        Twc = np.array(d["extrinsics"]["T_world_cam"], float)  # {}^W T_C
        Ts.append(Twc)
        if K_key not in d:
            raise KeyError(f"{K_key} missing in {path} (has {list(d.keys())}).")
        K_here = np.array(d[K_key]["K"], float)
        if K_ref is None: K_ref = K_here
        else:
            if not np.allclose(K_ref, K_here, atol=1e-6):
                raise ValueError(f"Intrinsics changed at frame {fid}.")
    return np.stack(Ts, 0), K_ref

# ------------------ PnP & Hand-eye ------------------

def pnp_pose_in_orbbec_world(Pw_orbbec, uv, K, valid, min_inliers=8):
    """
    Per-frame PnP on rectified/unrotated pixels.
    Returns:
      Twc_O: {}^O T_C (camera-to-world in Orbbec world), reproj_rms_px (float), inlier_mask [N]
      or (None, inf, zeros) if fails.
    """
    valid = valid & np.isfinite(uv).all(axis=1)
    obj = Pw_orbbec[valid].astype(np.float64)
    img = uv[valid].astype(np.float64)
    if obj.shape[0] < 4:
        return None, np.inf, np.zeros_like(valid, bool)

    dist = np.zeros(4)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=obj, imagePoints=img, cameraMatrix=K, distCoeffs=dist,
        flags=cv2.SOLVEPNP_EPNP, reprojectionError=4.0,
        iterationsCount=200, confidence=0.999
    )
    if not ok or inliers is None or len(inliers) < min_inliers:
        return None, np.inf, np.zeros_like(valid, bool)

    rvec, tvec = cv2.solvePnPRefineLM(obj[inliers[:,0]], img[inliers[:,0]], K, dist, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    Tco = np.eye(4); Tco[:3,:3]=R; Tco[:3,3]=t       # {}^C T_O
    Twc_O = inv_T(Tco)                               # {}^O T_C

    proj, _ = cv2.projectPoints(obj[inliers[:,0]], rvec, tvec, K, dist)
    proj = proj.reshape(-1,2)
    rms = float(np.sqrt(np.mean(np.sum((proj - img[inliers[:,0]])**2, axis=1))))

    inlier_mask = np.zeros_like(valid, bool)
    inlier_mask[np.where(valid)[0][inliers[:,0]]] = True
    return Twc_O, rms, inlier_mask

def align_worlds_orbbec_to_aria(Twc_O_seq, Twc_A_seq, huber=3.0, max_nfev=200):
    """
    Solve for X = {}^A T_O (Orbbec -> Aria) minimizing:
        E_t = {}^O T_C(t) * {}^A T_C(t)^{-1} * X   (identity at optimum)
    """
    assert len(Twc_O_seq) == len(Twc_A_seq) and len(Twc_O_seq) > 0
    A = np.stack(Twc_O_seq, 0)   # {}^O T_C(t)
    B = np.stack(Twc_A_seq, 0)   # {}^A T_C(t)

    def residuals(x):
        X = se3_exp(x)           # {}^A T_O
        res = []
        for t in range(A.shape[0]):
            Et = A[t] @ inv_T(B[t]) @ X
            r = se3_log(Et)
            s = np.abs(r)
            w = np.ones_like(s); m = s > huber; w[m] = huber / s[m]
            res.append(r * w)
        return np.concatenate(res)

    x0 = np.zeros(6)
    sol = least_squares(residuals, x0, method="lm", max_nfev=max_nfev)
    X_ao = se3_exp(sol.x)        # {}^A T_O
    X_oa = inv_T(X_ao)           # {}^O T_A
    return X_ao, X_oa, sol

def build_motion_pairs(Twc_seq, gap=1):
    """From a list of Twc (camera-to-world), build relative motions A_i = inv(Twc_t) @ Twc_{t+gap}."""
    A = []
    for i in range(len(Twc_seq)-gap):
        T1 = Twc_seq[i]
        T2 = Twc_seq[i+gap]
        A.append(np.linalg.inv(T1) @ T2)
    return A

def handeye_AX_XB_motions(A_list, B_list, huber=3.0, max_nfev=200):
    """
    Nonlinear LS on motions: minimize sum log( A_i X B_i^{-1} X^{-1} ).
    Returns X ({}^A T_O) and the optimizer result.
    """
    assert len(A_list) == len(B_list) and len(A_list) > 0
    def hat3(w):
        return np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]], float)
    def so3_log(R):
        tr = np.trace(R); c = np.clip((tr-1)*0.5, -1, 1); th = np.arccos(c)
        if th < 1e-12: return np.zeros(3)
        w_hat = (R - R.T) * (0.5/np.sin(th))
        return np.array([w_hat[2,1], w_hat[0,2], w_hat[1,0]]) * th
    def se3_log(T):
        R = T[:3,:3]; t = T[:3,3]; phi = so3_log(R); th = np.linalg.norm(phi)
        if th < 1e-12: Vinv = np.eye(3)
        else:
            k = phi/th; K = hat3(k)
            Vinv = (np.eye(3) - 0.5*K + (1/(th**2) - (1+np.cos(th))/(2*th*np.sin(th))) * (K@K))
        return np.r_[phi, Vinv @ t]
    def se3_exp(xi):
        w = xi[:3]; v = xi[3:]; th = np.linalg.norm(w)
        if th < 1e-12:
            R = np.eye(3); V = np.eye(3)
        else:
            k = w/th; K = hat3(k)
            R = np.eye(3) + np.sin(th)*K + (1-np.cos(th))*(K@K)
            V = np.eye(3) + (1-np.cos(th))*K + (th-np.sin(th))*(K@K)
        T = np.eye(4); T[:3,:3]=R; T[:3,3]=(V@v); return T
    def invT(T):
        R = T[:3,:3]; t = T[:3,3]
        Ti = np.eye(4); Ti[:3,:3]=R.T; Ti[:3,3]= -R.T@t; return Ti

    def residuals(x):
        X = se3_exp(x); Xi = invT(X)
        res = []
        for A,B in zip(A_list,B_list):
            E = A @ X @ invT(B) @ Xi
            r = se3_log(E)
            s = np.abs(r); w = np.ones_like(s); m = s > huber; w[m] = huber/s[m]
            res.append(r*w)
        return np.concatenate(res)

    x0 = np.zeros(6)
    sol = least_squares(residuals, x0, method="lm", max_nfev=max_nfev)
    # Reconstruct X
    # (reuse se3_exp from above closure)
    X = se3_exp(sol.x)
    return X, sol


# ------------------ QA: reprojection stats ------------------

def project_pinhole(K, Xc):
    u = K[0,0]*(Xc[:,0]/Xc[:,2]) + K[0,2]
    v = K[1,1]*(Xc[:,1]/Xc[:,2]) + K[1,2]
    return np.stack([u,v], axis=1)

def reprojection_stats_across_frames(X_ao, Twc_A, K, P3D_O, kp2d, valid):
    rms_list = []
    for t in range(P3D_O.shape[0]):
        m = valid[t] & np.isfinite(P3D_O[t]).all(axis=1) & np.isfinite(kp2d[t]).all(axis=1)
        if np.count_nonzero(m) < 4:
            continue
        Pw = P3D_O[t, m]
        # Orbbec -> Aria
        Pw_A = Pw @ X_ao[:3,:3].T + X_ao[:3,3]
        Tcw = np.linalg.inv(Twc_A[t])
        Xc = (np.c_[Pw_A, np.ones(len(Pw_A))] @ Tcw.T)[:, :3]
        z_ok = Xc[:,2] > 1e-6
        if np.count_nonzero(z_ok) < 4:
            continue
        uv = np.column_stack([K[0,0]*(Xc[z_ok,0]/Xc[z_ok,2]) + K[0,2],
                              K[1,1]*(Xc[z_ok,1]/Xc[z_ok,2]) + K[1,2]])
        err = uv - kp2d[t, m][z_ok]
        rms = float(np.sqrt(np.mean(np.sum(err*err, axis=1))))
        rms_list.append(rms)
    if not rms_list:
        return np.inf, np.inf, []
    return float(np.mean(rms_list)), float(np.median(rms_list)), rms_list


def se3_to_rt(T):
    R = T[:3,:3]; t = T[:3,3]
    return R, t

def rot_geodesic_deg(R1, R2):
    R = R1.T @ R2
    tr = np.clip((np.trace(R)-1)/2, -1, 1)
    return np.degrees(np.arccos(tr))

def diagnostics_per_frame_transforms(Twc_O_list, Twc_A_list, save_path=None):
    """
    Given per-frame camera-to-world in Orbbec (Twc_O_list) and Aria (Twc_A_list),
    compute X_t = Twc_A @ inv(Twc_O) and summarize.
    """
    X_list = []
    for A, B in zip(Twc_A_list, Twc_O_list):
        X_t = A @ inv_T(B)           # {}^A T_O at frame t
        X_list.append(X_t)

    # Rotation/translation scatter vs the first frame
    R0, t0 = se3_to_rt(X_list[0])
    rot_d = []
    trans_cm = []
    for X in X_list:
        R, t = se3_to_rt(X)
        rot_d.append(rot_geodesic_deg(R0, R))
        trans_cm.append(np.linalg.norm(t - t0) * 100.0)

    print(f"[DIAG] X_t rotation spread vs first frame: mean={np.mean(rot_d):.2f}°, median={np.median(rot_d):.2f}°, max={np.max(rot_d):.2f}°")
    print(f"[DIAG] X_t translation spread vs first frame: mean={np.mean(trans_cm):.2f} cm, median={np.median(trans_cm):.2f} cm, max={np.max(trans_cm):.2f} cm")

    if save_path:
        np.save(save_path, np.stack(X_list,0))
        print(f"[DIAG] Saved per-frame X_t to {save_path}")

    return X_list, np.array(rot_d), np.array(trans_cm)

# ------------------ Main ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kp2d_dir", required=True, help="Dir with <fid>.json (2D keypoints on UNDISTORTED image; maybe rotated 90°).")
    ap.add_argument("--j3d_dir",  required=True, help='Dir with <fid>.0_cam1.0_left/right_3d_keypoints.npz files.')
    ap.add_argument("--calib_dir", required=True, help="Dir with calib_XXXXXX.json (contains T_world_cam and intrinsics blocks).")
    ap.add_argument("--use_left",  action="store_true", help="Use LEFT hand if available and conf>0.")
    ap.add_argument("--use_right", action="store_true", help="Use RIGHT hand if available and conf>0.")
    ap.add_argument("--K_key", default="destination_intrinsics",
                    choices=["destination_intrinsics"],  # pinhole (unrotated) only
                    help="Intrinsics matching the UNROTATED rectified image. Use 'destination_intrinsics'.")
    ap.add_argument("--unrotate", default="cw90", choices=["none","cw90","ccw90"],
                    help="How to map annotated pixels back to UNROTATED rectified plane. Default 'cw90'.")
    ap.add_argument("--rot_w", type=int, default=1024, help="Width of the ROTATED rectified image.")
    ap.add_argument("--rot_h", type=int, default=1024, help="Height of the ROTATED rectified image.")
    ap.add_argument("--scale3d", type=float, default=1.0, help="Scale for 3D (use 0.001 if your 3D is in mm).")
    ap.add_argument("--min_inliers", type=int, default=8, help="Min PnP inliers to accept a frame.")
    ap.add_argument("--rms_px_thresh", type=float, default=6.0, help="Drop frames with PnP RMS above this (px).")
    ap.add_argument("--max_frames", type=int, default=0, help="Optional limit on frames used (0 = all).")
    ap.add_argument("--stride", type=int, default=1, help="Use every Nth usable frame to speed up.")
    ap.add_argument("--out_prefix", default="./handeye", help="Prefix for saved outputs.")
    ap.add_argument("--x_prior", type=str, default=None,
                    help="Path to 4x4 .npy with initial transform. Usually T_aria_orbbec (Aria <- Orbbec).")
    ap.add_argument("--x_prior_type", choices=["ao","oa"], default="ao",
                    help="'ao' = file is T_aria_orbbec (Aria <- Orbbec). 'oa' = file is T_orbbec_aria (Orbbec <- Aria).")
    ap.add_argument("--pnp_prior_px", type=float, default=8.0,
                    help="Pixels for prior-based inlier selection around the prior projection.")
    ap.add_argument("--accept_deg", type=float, default=45.0,
                    help="Reject frame if PnP pose differs from prior prediction by more than this (degrees).")
    ap.add_argument("--accept_m", type=float, default=0.5,
                    help="Reject frame if PnP pose differs from prior prediction by more than this (meters).")
    ap.add_argument("--fid_min", type=int, default=None, help="Keep only frames with fid >= this.")
    ap.add_argument("--fid_max", type=int, default=None, help="Keep only frames with fid <= this.")


    args = ap.parse_args()

    if not (args.use_left or args.use_right):
        # Default to both if neither flag is provided
        args.use_left  = True
        args.use_right = True

    # 1) Build arrays (fixed shape per frame)
    frame_ids, kp2d_rot, P3D_O, valid = build_arrays_custom_npz_fixedK(
        args.kp2d_dir, args.j3d_dir,
        use_left=args.use_left, use_right=args.use_right,
        scale3d=args.scale3d
    )

    # Optionally subsample
    if args.stride > 1:
        sel = np.arange(0, len(frame_ids), args.stride)
        frame_ids = [frame_ids[i] for i in sel]
        kp2d_rot, P3D_O, valid = kp2d_rot[sel], P3D_O[sel], valid[sel]
    if args.max_frames and len(frame_ids) > args.max_frames:
        sel = np.linspace(0, len(frame_ids)-1, args.max_frames).astype(int)
        frame_ids = [frame_ids[i] for i in sel]
        kp2d_rot, P3D_O, valid = kp2d_rot[sel], P3D_O[sel], valid[sel]

    Kj = kp2d_rot.shape[1]
    print(f"[INFO] Loaded {len(frame_ids)} frames; Kj={Kj} (fixed per frame).")

    # 2) Unrotate keypoints (if needed)
    if args.unrotate != "none":
        if args.unrotate == "cw90":
            kp2d = unrotate_uv_90cw_batch(kp2d_rot, W_rot=args.rot_w, H_rot=args.rot_h)
            print("[INFO] Unrotated keypoints from 90° CW to original rectified plane.")
        else:
            kp2d = unrotate_uv_90ccw_batch(kp2d_rot, W_rot=args.rot_w, H_rot=args.rot_h)
            print("[INFO] Unrotated keypoints from 90° CCW to original rectified plane.")
    else:
        kp2d = kp2d_rot
        
    # 3) Load UNROTATED rectified intrinsics and camera poses (Aria world)
    Twc_A, K = load_cam_T_world_and_K_from_calibs(args.calib_dir, frame_ids, K_key=args.K_key)
    print(f"[INFO] Using K from '{args.K_key}':\n{K}")
    if abs(K[1,0]) > 1e-6:
        print("[WARN] K[1,0] is not ~0; ensure you are using the UNROTATED pinhole K (destination_intrinsics).")
    
    X_ao_prior = None
    if args.x_prior:
        # X_prior_loaded = np.load(args.x_prior).astype(float)
        T_orbbec_aria_may_rot = np.array([
            [-0.46918556,  0.8819397,   0.0452469,   0.069228  ],
            [-0.0348995,  -0.06971398,  0.99695635,  0.52026004],
            [ 0.88240975,  0.46617845,  0.06348804,  0.30815   ],
            [ 0.,          0.,          0.,          1.       ]
        ], dtype=np.float32)
        X_prior_loaded = T_orbbec_aria_may_rot
        X_ao_prior = X_prior_loaded if args.x_prior_type == "ao" else inv_T(X_prior_loaded)
        print("[INFO] Loaded X prior ({}):\n{}".format(args.x_prior_type, X_ao_prior))

    # 4) Per-frame PnP in Orbbec world
    Twc_O_list, Twc_A_list, rms_list, kept_ids = [], [], [], []
    for t, fid in enumerate(frame_ids):
        if X_ao_prior is not None:
            Twc_O, rms, _inl = pnp_pose_in_orbbec_world_prior(
                P3D_O[t], kp2d[t], K, valid[t],
                Twc_A[t], X_ao_prior,
                pre_px=args.pnp_prior_px,
                min_inliers=args.min_inliers,
                accept_deg=args.accept_deg,
                accept_m=args.accept_m
            )
        else:
            Twc_O, rms, _inl = pnp_pose_in_orbbec_world(
                P3D_O[t], kp2d[t], K, valid[t],
                min_inliers=args.min_inliers
            )
        if Twc_O is None:
            continue
        if rms > args.rms_px_thresh:
            continue
        Twc_O_list.append(Twc_O)
        Twc_A_list.append(Twc_A[t])
        rms_list.append(rms)
        kept_ids.append(fid)

    if not Twc_O_list:
        raise RuntimeError("PnP failed for all frames (or all filtered). Check K/unrotation/shift/conf and adjust thresholds.")
    
    print(f"[INFO] PnP accepted {len(Twc_O_list)}/{len(frame_ids)} frames.")
    print(f"[INFO] PnP RMS (px): mean={np.mean(rms_list):.2f}, median={np.median(rms_list):.2f}")
    X_list, rot_d, trans_cm = diagnostics_per_frame_transforms(
        Twc_O_list, Twc_A_list, save_path=f"{args.out_prefix}_X_per_frame.npy"
    )

    # X_list from diagnostics_per_frame_transforms(...)
    inliers, X_cluster = cluster_X_list(X_list, rot_thresh_deg=8.0, trans_thresh_m=0.10)
    print(f"[CLUSTER] Chosen cluster size: {len(inliers)} / {len(X_list)}")
    if X_cluster is not None and len(inliers) >= 10:
        np.save(f"{args.out_prefix}_T_aria_orbbec_cluster.npy", X_cluster)
        print(f"[CLUSTER] Saved cluster-mean T_aria_orbbec to {args.out_prefix}_T_aria_orbbec_cluster.npy")

    # Build motions from accepted sequences (ensure same ordering)
    A_m = build_motion_pairs(Twc_A_list, gap=1)
    B_m = build_motion_pairs(Twc_O_list, gap=1)
    min_len = min(len(A_m), len(B_m))
    A_m = A_m[:min_len]; B_m = B_m[:min_len]

    X_motions, sol_m = handeye_AX_XB_motions(A_m, B_m, huber=3.0, max_nfev=200)
    np.save(f"{args.out_prefix}_T_aria_orbbec_motions.npy", X_motions)
    print(f"[MOT] LM status: {sol_m.status} | cost={sol_m.cost:.3f}")
    print("[MOT] T_aria_orbbec from motions:\n", X_motions)

    # 5) Hand-eye world alignment
    X_ao, X_oa, sol = align_worlds_orbbec_to_aria(Twc_O_list, Twc_A_list, huber=3.0, max_nfev=200)
    print(f"[INFO] LM status: {sol.status} | cost={sol.cost:.3f}")
    print("[INFO] T_aria_orbbec (Orbbec -> Aria):\n", X_ao)
    print("[INFO] T_orbbec_aria (Aria -> Orbbec):\n", X_oa)

    # 6) Global 2D reprojection QA with X_ao across all frames
    mean_rms, med_rms, _ = reprojection_stats_across_frames(X_ao, Twc_A, K, P3D_O, kp2d, valid)
    print(f"[INFO] Global 2D reprojection RMS with T_aria_orbbec: mean={mean_rms:.2f}px, median={med_rms:.2f}px")

    # 7) Save outputs
    np.save(f"{args.out_prefix}_T_aria_orbbec.npy", X_ao)
    np.save(f"{args.out_prefix}_T_orbbec_aria.npy", X_oa)

    report = {
        "frames_total": int(len(frame_ids)),
        "frames_used": int(len(Twc_O_list)),
        "kept_frame_ids": kept_ids,
        "K_key": args.K_key,
        "K": K.tolist(),
        "pnp_rms_px_mean": float(np.mean(rms_list)),
        "pnp_rms_px_median": float(np.median(rms_list)),
        "handeye_cost": float(sol.cost),
        "global_reproj_rms_mean_px": float(mean_rms),
        "global_reproj_rms_median_px": float(med_rms),
        "unrotate": args.unrotate,
        "rotated_size": [int(args.rot_w), int(args.rot_h)],
        "use_left": bool(args.use_left),
        "use_right": bool(args.use_right),
        "Kj_per_frame": int(Kj),
    }
    with open(f"{args.out_prefix}_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Saved:\n  {args.out_prefix}_T_aria_orbbec.npy\n  {args.out_prefix}_T_orbbec_aria.npy\n  {args.out_prefix}_report.json")

if __name__ == "__main__":
    main()

# python align_worlds_from_3d.py \
#   --aria3d_dir ../data/20250519_Testing/Aria/export/hand \
#   --orbbec3d_dir ../HaMuCo/test_set_result/2025-08-20_22:37/3d_kps/20250519 \
#   --use_left --use_right \
#   --use_landmarks --palm_mode avg_mcp \
#   --huber_m 0.05 \
#   --out_prefix ./X_from3D --iters 100 --scale3d 0.001