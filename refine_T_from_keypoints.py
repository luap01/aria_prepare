import os, re, json, glob, warnings
import numpy as np
from scipy.optimize import least_squares

# ---------- SE(3) utilities ----------
def hat(w):
    wx, wy, wz = w
    return np.array([[0,-wz,wy],[wz,0,-wx],[-wy,wx,0]], float)

def se3_exp(xi):
    """xi = [wx, wy, wz, tx, ty, tz] -> 4x4"""
    w = xi[:3]; t = xi[3:]
    th = np.linalg.norm(w)
    if th < 1e-12:
        R = np.eye(3)
        V = np.eye(3)
    else:
        K = hat(w/th)
        R = np.eye(3) + np.sin(th)*K + (1-np.cos(th))*(K@K)
        # V = (np.eye(3) + (1-np.cos(th))*K + (th-np.sin(th))*(K@K)) / th
        V = (np.eye(3)
             + ((1-np.cos(th))/th)*K
             + ((th-np.sin(th))/th)*(K@K))
    T = np.eye(4)
    T[:3, :3] = R 
    T[:3, 3]  = V @ t
    return T

# ---------- Projection models ----------
def project_pinhole(K, Xc):
    """Xc: Nx3 in camera frame (no distortion)"""
    x = Xc[:,0]/Xc[:,2]; y = Xc[:,1]/Xc[:,2]
    u = K[0,0]*x + K[0,2]; v = K[1,1]*y + K[1,2]
    return np.stack([u,v], axis=1)

def project_fisheye_equidistant(K, D, Xc):
    """
    OpenCV fisheye (equidistant): r = f * theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
    K: 3x3, D: (k1,k2,k3,k4)
    """
    x = Xc[:,0] / Xc[:,2]
    y = Xc[:,1] / Xc[:,2]
    r = np.sqrt(x*x + y*y) + 1e-12
    theta = np.arctan(r)
    k1,k2,k3,k4 = D
    theta_d = theta*(1 + k1*theta**2 + k2*theta**4 + k3*theta**6 + k4*theta**8)
    scale = theta_d / r
    xd = x*scale; yd = y*scale
    u = K[0,0]*xd + K[0,2]; v = K[1,1]*yd + K[1,2]
    return np.stack([u,v], axis=1)

# ---------- Robust solver ----------
def refine_T_from_keypoints(
    T_prior,              # 4x4 Orbbec->Aria (good initial guess)
    K,                    # 3x3 intrinsics of Aria camera
    cam_T_world,          # [T] list/array of 4x4 camera-to-world in Aria frame
    P3D_orbbec,           # [T,Kj,3] hand joints in Orbbec world
    kp2d,                 # [T,Kj,2] measured pixels in Aria image (raw if fisheye)
    valid=None,           # [T,Kj] bool mask (True = use)
    model="fisheye",      # "fisheye" or "pinhole"
    D=(0,0,0,0),          # fisheye coeffs if model="fisheye"
    dof_mask=(1,1,1,1,1,1),  # lock some DOFs, e.g. (0,0,0,1,1,1) for translation-only
    huber_px=3.0,         # robust cutoff in pixels
    prior_sigma=(np.deg2rad(1), np.deg2rad(1), np.deg2rad(1), 0.01, 0.01, 0.01)  # small prior
):
    T_prior = np.asarray(T_prior, float).copy()
    cam_T_world = np.asarray(cam_T_world, float)
    P3D_orbbec = np.asarray(P3D_orbbec, float)
    kp2d = np.asarray(kp2d, float)
    T_frames, Kj, _ = P3D_orbbec.shape
    if valid is None: valid = np.ones((T_frames,Kj), bool)

    dof_mask = np.array(dof_mask, float)
    free_idx = np.where(dof_mask>0.5)[0]

    def pack(x_free):
        xi = np.zeros(6); xi[free_idx] = x_free
        return xi

    proj = (lambda K, Xc: project_fisheye_equidistant(K, D, Xc)) if model == "fisheye" else project_pinhole

    # Prebuild homogeneous points per frame
    Pw_o = np.concatenate([P3D_orbbec, np.ones((T_frames,Kj,1))], axis=2)  # [T,K,4]

    # Residuals
    def residuals(x_free):
        xi = pack(x_free)
        T = se3_exp(xi) @ T_prior
        res = []
        for t in range(T_frames):
            # Orbbec world -> Aria world
            Pw_a = (Pw_o[t] @ T.T)[:, :3]  # [K,3]
            # Aria world -> camera
            Twc = cam_T_world[t]                    # camera-to-world
            Tcw = np.linalg.inv(Twc)                # world-to-camera
            Xc = (np.c_[Pw_a, np.ones((Kj,1))] @ Tcw.T)[:, :3]
            # drop points that ended up behind the camera
            z_ok = Xc[:,2] > 1e-6
            uv = proj(K, Xc[z_ok])
            m = valid[t] & z_ok
            r = (uv - kp2d[t,m]).reshape(-1)
            res.append(r)

        # Small quadratic prior on xi (keeps update tiny)
        w = 1.0/np.array(prior_sigma, float)
        r_prior = (w*xi).astype(float)
        return np.concatenate(res + [r_prior])

    # Huber on residuals (manually; LM does Gauss-Newton)
    def residuals_huber(x_free):
        r = residuals(x_free)
        s = np.abs(r)
        w = np.ones_like(s)
        mask = s > huber_px
        w[mask] = huber_px / s[mask]
        return r * w

    x0 = np.zeros(len(free_idx))
    sol = least_squares(residuals_huber, x0, method="lm", max_nfev=50)
    T_ref = se3_exp(pack(sol.x)) @ T_prior
    return T_ref, sol

# ---------- File I/O helpers ----------
FRAME2D_RX = re.compile(r"frame_(\d+)\.json$")
FRAME3D_RX = re.compile(r"frame_(\d+)_joints3d\.json$")

def _load_json_allow_nan(path):
    """Load JSON, replacing bare NaN/Infinity with null so the std decoder won’t choke."""
    with open(path, "r") as f:
        txt = f.read()
    txt = txt.replace("NaN", "null").replace("Infinity", "null").replace("-Infinity", "null")
    return json.loads(txt)

def _parse_2d_side(arr_flat, vis_min=1):
    """
    arr_flat: length 63 = 21*(x,y,v).
    Returns (uv [21,2], valid [21]) with valid = (visibility >= vis_min) & finite(uv).
    """
    a = np.asarray(arr_flat, float).reshape(-1, 3)  # [21,3]
    uv = a[:, :2]
    vis = a[:, 2]
    valid = (vis >= vis_min) & np.isfinite(uv).all(axis=1)
    return uv, valid

def _parse_3d_side(x3d, mask=None):
    """
    x3d: list of 21 [x,y,z], mask: optional 21 ints.
    Returns (X [21,3], valid3d [21]) where valid3d True if mask>0 (or all True if None).
    """
    X = np.asarray(x3d, float)
    if mask is None:
        valid3d = np.ones(X.shape[0], dtype=bool)
    else:
        valid3d = np.asarray(mask, int) > 0
    return X, valid3d

def _collect_pairs(kp2d_dir, j3d_dir):
    """Find frames present in both dirs and return sorted frame ids + paths."""
    twod = {int(FRAME2D_RX.search(os.path.basename(p)).group(1)): p
            for p in glob.glob(os.path.join(kp2d_dir, "frame_*.json"))
            if FRAME2D_RX.search(os.path.basename(p))}
    thrd = {int(FRAME3D_RX.search(os.path.basename(p)).group(1)): p
            for p in glob.glob(os.path.join(j3d_dir, "frame_*_joints3d.json"))
            if FRAME3D_RX.search(os.path.basename(p))}
    common = sorted(set(twod).intersection(thrd))
    missing_2d = sorted(set(thrd).difference(twod))
    missing_3d = sorted(set(twod).difference(thrd))
    if missing_2d:
        warnings.warn(f"{len(missing_2d)} frames missing 2D: {missing_2d[:10]}...")
    if missing_3d:
        warnings.warn(f"{len(missing_3d)} frames missing 3D: {missing_3d[:10]}...")
    return common, twod, thrd

def load_cam_T_world_and_K_from_calibs(calib_dir, frame_ids):
    """
    Reads calib_<fid>.json files and returns:
      cam_T_world: [T,4,4] with T_world_cam from each file (camera-to-world)
      K_raw:       3x3 from 'intrinsics.K' (verified constant across frames)
    """
    Ts = []
    K_ref = None
    for fid in frame_ids:
        path = os.path.join(calib_dir, f"calib_{fid:06d}.json")
        d = _load_json_allow_nan(path)
        Twc = np.array(d["extrinsics"]["T_world_cam"], dtype=float)
        assert Twc.shape == (4,4)
        Ts.append(Twc)

        K_here = np.array(d["intrinsics"]["K"], dtype=float)
        if K_ref is None:
            K_ref = K_here
        else:
            if not np.allclose(K_ref, K_here, atol=1e-6):
                raise ValueError(f"Intrinsics changed at frame {fid}.")

        # (Optional sanity check) T_world_device @ T_device_cam == T_world_cam
        if "T_world_device" in d["extrinsics"] and "T_device_cam" in d["extrinsics"]:
            Twd = np.array(d["extrinsics"]["T_world_device"], float)
            Tdc = np.array(d["extrinsics"]["T_device_cam"], float)
            if not np.allclose(Twd @ Tdc, Twc, atol=1e-5):
                print(f"Warning: T_world_device @ T_device_cam != T_world_cam at frame {fid}")

    cam_T_world = np.stack(Ts, axis=0)
    return cam_T_world, K_ref

import numpy as np
from scipy.optimize import least_squares
import cv2, json, os, glob

# ---------- SO(3)/SE(3) utils ----------
def hat3(w):
    wx, wy, wz = w
    return np.array([[0, -wz, wy],
                     [wz, 0, -wx],
                     [-wy, wx, 0]], float)

def so3_log(R):
    # numerically stable log for SO(3)
    cos_th = (np.trace(R) - 1.0) * 0.5
    cos_th = np.clip(cos_th, -1.0, 1.0)
    th = np.arccos(cos_th)
    if th < 1e-8:
        return np.zeros(3)
    w_hat = (R - R.T) * (0.5 / np.sin(th))
    return np.array([w_hat[2,1], w_hat[0,2], w_hat[1,0]]) * th

def se3_exp(xi):
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
        # Left Jacobian inverse for SO(3)
        Vinv = (np.eye(3) - 0.5*K
                + (1/(th**2) - (1 + np.cos(th))/(2*th*np.sin(th))) * (K@K))
    rho = Vinv @ t
    return np.r_[phi, rho]

def inv_T(T):
    R = T[:3,:3]; t = T[:3,3]
    Ti = np.eye(4); Ti[:3,:3] = R.T; Ti[:3,3] = -R.T @ t
    return Ti

# ---------- I/O ----------
def _load_json(path):
    with open(path, "r") as f:
        txt = f.read().replace("NaN", "null").replace("Infinity", "null").replace("-Infinity", "null")
    return json.loads(txt)

def load_cam_T_world_and_K_from_calibs(calib_dir, frame_ids, K_key="destination_intrinsics_rotated"):
    """
    Returns:
      Twc_list: [T,4,4]  (camera-to-world, in Aria 'world')
      K:        3x3      from the selected key:
                'intrinsics' (raw fisheye),
                'destination_intrinsics' (rectified),
                'destination_intrinsics_rotated' (rectified + 90° CW)
    """
    Ts = []
    K_ref = None
    for fid in frame_ids:
        d = _load_json(os.path.join(calib_dir, f"calib_{fid:06d}.json"))
        Twc = np.array(d["extrinsics"]["T_world_cam"], float)  # {}^W T_C (camera-to-world)
        Ts.append(Twc)
        if K_key not in d:
            raise KeyError(f"{K_key} missing in calib_{fid:06d}.json (has keys: {list(d.keys())})")
        K_here = np.array(d[K_key]["K"], float)
        if K_ref is None:
            K_ref = K_here
        else:
            if not np.allclose(K_ref, K_here, atol=1e-6):
                raise ValueError(f"Intrinsics changed at frame {fid}.")
    return np.stack(Ts, 0), K_ref

# ---------- PnP per frame (Orbbec world -> Aria camera) ----------
def pnp_pose_in_orbbec_world(Pw_orbbec, uv, K, valid=None):
    """
    Inputs per frame:
      Pw_orbbec: [N,3] 3D in Orbbec world
      uv:        [N,2] pixels in the (rectified!) Aria image
      K:         3x3 pinhole intrinsics matching those pixels
      valid:     [N] optional bool mask
    Returns:
      Twc_O: 4x4 camera-to-world in Orbbec world  ({}^O T_C)
      reproj_rms: float
      inlier_mask: [N] bool
    """
    if valid is None: valid = np.isfinite(uv).all(1)
    obj = Pw_orbbec[valid].astype(np.float64)
    img = uv[valid].astype(np.float64)
    if obj.shape[0] < 4:
        return None, np.inf, np.zeros_like(valid, bool)

    dist = np.zeros(4)  # rectified → zero distortion
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=obj, imagePoints=img, cameraMatrix=K, distCoeffs=dist,
        flags=cv2.SOLVEPNP_EPNP, reprojectionError=4.0, iterationsCount=200, confidence=0.999
    )
    if not success:
        return None, np.inf, np.zeros_like(valid, bool)
    # optional LM refinement
    rvec, tvec = cv2.solvePnPRefineLM(obj[inliers[:,0]], img[inliers[:,0]], K, dist, rvec, tvec)

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    Tco = np.eye(4); Tco[:3,:3]=R; Tco[:3,3]=t       # {}^C T_O
    Twc_O = inv_T(Tco)                               # {}^O T_C

    # reprojection RMS over inliers
    proj, _ = cv2.projectPoints(obj[inliers[:,0]], rvec, tvec, K, dist)
    proj = proj.reshape(-1,2)
    rms = np.sqrt(np.mean(np.sum((proj - img[inliers[:,0]])**2, axis=1)))
    inlier_mask = np.zeros_like(valid, bool); inlier_mask[np.where(valid)[0][inliers[:,0]]] = True
    return Twc_O, float(rms), inlier_mask

# ---------- Hand-eye world alignment (AX ≈ B) ----------
def align_worlds_orbbec_to_aria(Twc_O_seq, Twc_A_seq, X0=None, huber=3.0):
    """
    Solve for a single X = {}^A T_O (Orbbec->Aria) that best aligns camera poses:
        {}^O T_C(t) ≈ X^{-1} {}^A T_C(t)
    We return X_oa = {}^A T_O and its inverse X_ao = {}^O T_A for convenience.
    """
    assert len(Twc_O_seq) == len(Twc_A_seq) and len(Twc_O_seq) > 0
    if X0 is None:
        X0 = np.eye(4)

    def pack(x): return x
    def unpack(x): return se3_exp(x)

    A = np.stack(Twc_O_seq, 0)   # camera-to-world in O
    B = np.stack(Twc_A_seq, 0)   # camera-to-world in A

    def residuals(x):
        X = unpack(x)            # candidate {}^A T_O
        Xinv = inv_T(X)
        res = []
        for t in range(A.shape[0]):
            # error pose in SE(3):
            #  E_t = A_t * (X^{-1} * B_t)^{-1}  =  A_t * B_t^{-1} * X
            Et = A[t] @ inv_T(B[t]) @ X
            r = se3_log(Et)
            # simple Huber on each component
            s = np.abs(r); w = np.ones_like(s); m = s > huber; w[m] = huber / s[m]
            res.append(r * w)
        return np.concatenate(res)

    x0 = np.zeros(6)
    sol = least_squares(residuals, x0, method="lm", max_nfev=200)
    X_ao = unpack(sol.x)          # {}^A T_O  (Aria <- Orbbec)
    X_oa = inv_T(X_ao)            # {}^O T_A  (Orbbec <- Aria)
    return X_ao, X_oa, sol

# ---------- Driver (per your file layout) ----------
def run_pnp_handeye(
    frame_ids, kp2d, P3D_orbbec, valid,
    calib_dir, K_key="destination_intrinsics_rotated",
    cam_label="camera-rgb"
):
    """
    Inputs:
      frame_ids, kp2d, P3D_orbbec, valid: like in your builder
      calib_dir: folder containing calib_<fid>.json written by export.py
      K_key: which intrinsics to use. Use "destination_intrinsics_rotated" for your labeled images.
    Returns:
      X_ao: {}^A T_O (Orbbec -> Aria)
      X_oa: {}^O T_A (Aria -> Orbbec)
      diagnostics dict
    """
    Twc_A, K = load_cam_T_world_and_K_from_calibs(calib_dir, frame_ids, K_key=K_key)

    Twc_O_list = []
    rms_list   = []
    kept       = []

    for t, fid in enumerate(frame_ids):
        Tco, rms, inl = pnp_pose_in_orbbec_world(P3D_orbbec[t], kp2d[t], K, valid[t])
        if Tco is None: continue
        Twc_O_list.append(Tco); rms_list.append(rms); kept.append((t,fid))

    if not Twc_O_list:
        raise RuntimeError("PnP failed for all frames.")

    # align worlds
    Twc_O = np.stack(Twc_O_list, 0)
    Twc_A_sel = np.stack([Twc_A[t] for (t,_) in kept], 0)
    X_ao, X_oa, sol = align_worlds_orbbec_to_aria(Twc_O, Twc_A_sel)

    diags = {
        "num_frames_used": len(Twc_O_list),
        "mean_pnp_rms_px": float(np.mean(rms_list)),
        "median_pnp_rms_px": float(np.median(rms_list)),
        "lm_status": sol.status,
        "lm_message": sol.message,
        "lm_cost": sol.cost,
    }
    return X_ao, X_oa, diags


# ---------- Dataset builder ----------
def build_arrays(kp2d_dir, j3d_dir, use_left=True, use_right=True, vis_min=1):
    """
    Returns:
      frames: [T] frame ids
      kp2d:   [T, Kj, 2]
      P3D:    [T, Kj, 3]
      valid:  [T, Kj] bool
    Where Kj = 21*(use_left + use_right), ordered as left then right.
    """
    frames, twod, thrd = _collect_pairs(kp2d_dir, j3d_dir)
    kp_all, X_all, valid_all = [], [], []
    kept_frames = []
    for fid in frames:
        d2 = _load_json_allow_nan(twod[fid])
        d3 = _load_json_allow_nan(thrd[fid])

        kp_frame = []
        X_frame = []
        v_frame = []

        if use_left and ("left_hand" in d2) and ("left_hand" in d3):
            uvL, vL2d = _parse_2d_side(d2["left_hand"], vis_min=vis_min)
            XL, vL3d = _parse_3d_side(d3["left_hand"]["X3d"], d3["left_hand"].get("mask"))
            if uvL.shape[0] != XL.shape[0]:
                warnings.warn(f"Left count mismatch at frame {fid}: 2D {uvL.shape[0]} vs 3D {XL.shape[0]}")
            Kc = min(uvL.shape[0], XL.shape[0])
            kp_frame.append(uvL[:Kc]); X_frame.append(XL[:Kc]); v_frame.append((vL2d[:Kc] & vL3d[:Kc]))

        if use_right and ("right_hand" in d2) and ("right_hand" in d3):
            uvR, vR2d = _parse_2d_side(d2["right_hand"], vis_min=vis_min)
            XR, vR3d = _parse_3d_side(d3["right_hand"]["X3d"], d3["right_hand"].get("mask"))
            if uvR.shape[0] != XR.shape[0]:
                warnings.warn(f"Right count mismatch at frame {fid}: 2D {uvR.shape[0]} vs 3D {XR.shape[0]}")
            Kc = min(uvR.shape[0], XR.shape[0])
            kp_frame.append(uvR[:Kc]); X_frame.append(XR[:Kc]); v_frame.append((vR2d[:Kc] & vR3d[:Kc]))

        if not kp_frame:
            continue  # nothing usable in this frame

        kp_all.append(np.concatenate(kp_frame, axis=0))
        X_all.append(np.concatenate(X_frame, axis=0))
        valid_all.append(np.concatenate(v_frame, axis=0))
        kept_frames.append(fid)

    if not kp_all:
        raise RuntimeError("No overlapping frames with usable 2D+3D hands found.")

    kp2d = np.stack(kp_all, axis=0).astype(float)     # [T,Kj,2]
    P3D  = np.stack(X_all, axis=0).astype(float)      # [T,Kj,3]
    valid = np.stack(valid_all, axis=0).astype(bool)  # [T,Kj]
    return kept_frames, kp2d, P3D, valid

# ---------- Example usage ----------
if __name__ == "__main__":
    # ---- User paths (edit these) ----
    KP2D_DIR = "./aria_keypoints_2d"       # contains frame_XXXX.json
    J3D_DIR  = "./orbbec_joints3d"         # contains frame_XXXX_joints3d.json
    CAM_T_WORLD_SRC = "./cam_T_world.npy"  # either .npy (T×4×4 or 4×4) or a directory with per-frame .npy
    T_INIT_PATH = "./T_init.npy"           # 4×4 initial Orbbec->Aria
    OUT_PATH = "./T_refined_from_keypoints.npy"
    CALIB_DIR = "../data/20250519_Testing/Aria/export/calib"
    # ---- Camera intrinsics (raw Aria) ----
    # Replace with your actual K (fx, 0, cx; 0, fy, cy; 0,0,1)
    K = np.array([
        [   0.,  -300.,   511.5],   # <-- Replace with correct K for your setup
        [ 300.,     0.,   511.5],
        [   0.,     0.,     1. ]], dtype=float)

    # Fisheye distortion coeffs (OpenCV fisheye, equidistant)
    D = (0.0, 0.0, 0.0, 0.0)  # <-- Replace with actual (k1,k2,k3,k4) for best accuracy


    T_orbbec_aria_may = np.array([
        [-0.499695413510, 0.866106831943, 0.012784732288, 0.029228000000],
        [0.865497844508, 0.498640616331, 0.047655187522, 0.288147000000],
        [0.034899496703, 0.034878236872, -0.998782025130, -0.430261000000],
        [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]
        ], dtype=np.float32)
    
    T_aria_orbbec = np.linalg.inv(T_orbbec_aria_may)

    # ---- Build data arrays from files ----
    frame_ids, kp2d, P3D_orbbec, valid = build_arrays(
        KP2D_DIR, J3D_DIR, use_left=True, use_right=True, vis_min=1
    )
    print(f"Loaded {len(frame_ids)} frames; Kj={kp2d.shape[1]} joints per frame.")

    # ---- Load camera poses for those frames (Aria camera-to-world) ----
    cam_T_world, K = load_cam_T_world_and_K_from_calibs(CALIB_DIR, frame_ids)
    if cam_T_world.shape[0] != len(frame_ids):
        raise RuntimeError(f"cam_T_world length {cam_T_world.shape[0]} != #frames {len(frame_ids)}")
    assert cam_T_world.shape[1:] == (4,4), f"cam_T_world wrong shape: {cam_T_world.shape}"

    # ---- Load prior transform ----
    # T_prior = np.load(T_INIT_PATH).astype(float)
    T_prior = T_aria_orbbec
    assert T_prior.shape == (4,4), "T_init.npy must be a 4x4 matrix"

    # ---- Run refinement (raw fisheye) ----
    T_ref, _ = refine_T_from_keypoints(
        T_prior=T_prior,
        K=K,
        cam_T_world=cam_T_world,
        P3D_orbbec=P3D_orbbec,
        kp2d=kp2d,
        valid=valid,
        model="fisheye",
        D=D,
        dof_mask=(0,0,0,1,1,1),   # start with translation-only
        huber_px=3.0
    )

    T_ref, sol = refine_T_from_keypoints(
        T_prior=T_ref, K=K, cam_T_world=cam_T_world,
        P3D_orbbec=P3D_orbbec, kp2d=kp2d, valid=valid,
        model="fisheye", D=D,
        dof_mask=(1,1,1,1,1,1), huber_px=3.0
    )
    np.save(OUT_PATH, T_ref)
    print("Saved refined T to:", OUT_PATH)
    print("LM status:", sol.status, sol.message)
    print("Final cost (sum of squared, incl. prior):", sol.cost)
