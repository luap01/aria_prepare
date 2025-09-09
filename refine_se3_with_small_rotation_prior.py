#!/usr/bin/env python3
import os, glob, argparse, re, json
import numpy as np

ARIA_RX  = re.compile(r"^hand_([-\d]+)\.npy$")
JOINTS   = 21

def load_orbbec_npz(path):
    if not os.path.exists(path): return None
    with np.load(path) as data:
        if not data.files: return None
        X = np.array(data[data.files[0]], float).reshape(-1,3)
    return X

def palm_point(X21, idxs=(1,5,9,13,17)):
    idxs = [i for i in idxs if i < X21.shape[0]]
    return np.mean(X21[idxs], axis=0) if idxs else None

def collect_pairs(aria_dir, orbbec_dir, scale_orbbec=1.0, use_landmarks=True):
    """Return P_O (Orbbec world), P_A (Aria world). Uses 21-lm if present, else wrist+palm."""
    P_A, P_O = [], []
    for hp in sorted(glob.glob(os.path.join(aria_dir, "hand_*.npy"))):
        m = ARIA_RX.match(os.path.basename(hp))
        if not m: continue
        token = m.group(1)
        try:
            fid = int(token)
        except Exception:
            fid = int(token.lstrip("0") or "0")

        d = np.load(hp, allow_pickle=True).item()
        lconf = float(d.get("left_conf", 1.0)) if "left_conf" in d else 1.0
        rconf = float(d.get("right_conf",1.0)) if "right_conf" in d else 1.0

        # Aria 21-lm (world)
        A_L = np.asarray(d["left_landmarks_world"],  np.float32).reshape(-1,3)  if lconf>0 and "left_landmarks_world"  in d else None
        A_R = np.asarray(d["right_landmarks_world"], np.float32).reshape(-1,3)  if rconf>0 and "right_landmarks_world" in d else None

        # Fallback to wrist/palm
        if A_L is None:
            lw, lp = d.get("left_wrist",None), d.get("left_palm",None)
            if lw is not None and lp is not None:
                A_L = np.vstack([lw, lp]).astype(np.float32)
        if A_R is None:
            rw, rp = d.get("right_wrist",None), d.get("right_palm",None)
            if rw is not None and rp is not None:
                A_R = np.vstack([rw, rp]).astype(np.float32)

        # Orbbec 3D
        Lp = os.path.join(orbbec_dir, f"{fid}.0_cam1.0_left_3d_keypoints.npz")
        Rp = os.path.join(orbbec_dir, f"{fid}.0_cam1.0_right_3d_keypoints.npz")
        O_L = load_orbbec_npz(Lp)
        O_R = load_orbbec_npz(Rp)
        if O_L is not None: O_L = (O_L * scale_orbbec).astype(np.float32)
        if O_R is not None: O_R = (O_R * scale_orbbec).astype(np.float32)

        # Fallback to wrist+palm on Orbbec side when Aria has only 2
        if O_L is not None and A_L is not None and O_L.shape[0] >= 1 and A_L.shape[0] < 21:
            pp = palm_point(O_L);  O_L = np.vstack([O_L[0], pp]).astype(np.float32) if pp is not None else None
        if O_R is not None and A_R is not None and O_R.shape[0] >= 1 and A_R.shape[0] < 21:
            pp = palm_point(O_R);  O_R = np.vstack([O_R[0], pp]).astype(np.float32) if pp is not None else None

        # Pair by identity (adjust here if orders differ)
        if A_L is not None and O_L is not None:
            n = min(len(A_L), len(O_L));  P_A.append(A_L[:n]);  P_O.append(O_L[:n])
        if A_R is not None and O_R is not None:
            n = min(len(A_R), len(O_R));  P_A.append(A_R[:n]);  P_O.append(O_R[:n])

    if not P_A:
        raise RuntimeError("No pairs found. Check dirs, confidences, and filenames.")
    return np.vstack(P_A), np.vstack(P_O)

# --------- SE(3) utilities (small update) ----------
def hat_3(w):
    wx, wy, wz = w
    return np.array([[0,-wz,wy],[wz,0,-wx],[-wy,wx,0]], float)

def se3_exp(xi):
    """xi=[wx,wy,wz, tx,ty,tz]; returns 4x4."""
    w = xi[:3]; t = xi[3:]
    th = np.linalg.norm(w)
    T = np.eye(4); I = np.eye(3)
    if th < 1e-12:
        R = I; V = I
    else:
        K = hat_3(w/th)
        R = I + np.sin(th)*K + (1-np.cos(th))*(K@K)
        V = I + (1-np.cos(th))*K + (th-np.sin(th))*(K@K)
        V = V / th
    T[:3,:3] = R
    T[:3,3]  = (V @ t)
    return T

def refine_with_priors(X_oa_prior, P_A, P_O, huber=0.05, iters=15,
                       rot_sigma_deg=2.0, trans_sigma_m=None,
                       dof_rot=(1,1,1), dof_trans=(1,1,1)):
    """
    Solve for small delta xi so that X_new = exp(delta) @ X_oa_prior.
    Cost = Huber(|| (R*A + t) - O ||) + priors on rotation (and optionally translation).
    """
    R0 = X_oa_prior[:3,:3].copy()
    t0 = X_oa_prior[:3,3].copy()

    # Precompute A' = A @ R0^T  and O' = O - t0
    A = np.asarray(P_A, float);  O = np.asarray(P_O, float)
    A0 = (A @ R0.T)  # N x 3
    O0 = (O - t0)    # N x 3

    # dof masks
    mR = np.array(dof_rot, float)   > 0.5
    mT = np.array(dof_trans, float) > 0.5
    idx_free = np.r_[ np.where(mR)[0], 3+np.where(mT)[0] ]  # indices in xi that are free
    if len(idx_free)==0:  # nothing to optimize
        return X_oa_prior, np.zeros(6), 0.0

    # priors
    rot_w = 0.0 if rot_sigma_deg is None or rot_sigma_deg<=0 else 1.0 / np.deg2rad(rot_sigma_deg)
    trans_w = 0.0 if trans_sigma_m is None or trans_sigma_m<=0 else 1.0 / trans_sigma_m

    xi = np.zeros(6)  # small update
    for _ in range(iters):
        # Current update
        T = se3_exp(xi)
        R = T[:3,:3] @ R0
        t = T[:3,3] + t0

        # residuals r_i = (R*A_i + t) - O_i
        RA = (A @ R.T)
        r = (RA + t) - O      # N x 3
        d = np.linalg.norm(r, axis=1) + 1e-12
        w = np.ones_like(d)
        mask = d > huber
        w[mask] = huber / d[mask]
        W = w[:,None]

        # Jacobian (approx): dr/dxi at xi ~ 0 (valid for small)
        # d/dw (R*A) ~ -[RA]_x; d/dt = I
        # We build normal equations for the free DOFs only
        Jrot = -np.stack([hat_3(RA[i]) for i in range(RA.shape[0])], axis=0)  # N x 3 x 3
        J = np.concatenate([Jrot, np.tile(np.eye(3),(RA.shape[0],1,1))], axis=2)  # N x 3 x 6
        # mask columns not free
        J = J[:,:,idx_free]                                           # N x 3 x nf
        r_vec = (W * r).reshape(-1,3)                                 # weighted residuals
        Jw = (W[:,:,None] * J).reshape(-1, J.shape[2])                # weighted Jacobian

        # Add priors
        H = Jw.T @ Jw
        b = -Jw.T @ r_vec.reshape(-1)

        if rot_w > 0:
            # small-angle prior on rotation delta (first 3 comps)
            for k,ax in enumerate([0,1,2]):
                if ax in np.where(mR)[0]:
                    j = np.where(idx_free==ax)[0][0]
                    H[j,j] += rot_w**2
                    b[j]   += -rot_w**2 * xi[ax]
        if trans_w > 0:
            for k,ax in enumerate([0,1,2]):
                if 3+ax in idx_free:
                    j = np.where(idx_free==(3+ax))[0][0]
                    H[j,j] += trans_w**2
                    b[j]   += -trans_w**2 * xi[3+ax]

        # Solve
        try:
            dx_free = np.linalg.solve(H, b)
        except np.linalg.LinAlgError:
            dx_free = np.linalg.lstsq(H, b, rcond=None)[0]

        # unpack back into xi
        for k,j in enumerate(idx_free):
            xi[j] += dx_free[k]

        # stop if tiny
        if np.linalg.norm(dx_free) < 1e-9:
            break

    # Final X
    X_new = se3_exp(xi) @ X_oa_prior
    # Report numbers
    r_final = (A @ X_new[:3,:3].T + X_new[:3,3]) - O
    rms = float(np.sqrt(np.mean(np.sum(r_final*r_final, axis=1))))
    rot_deg = np.degrees(np.linalg.norm(xi[:3]))
    return X_new, xi, rms

def main():
    ap = argparse.ArgumentParser(description="Small-angle SE(3) refinement with rotation prior (Aria->Orbbec world).")
    ap.add_argument("--X_oa_prior", required=True, help="4x4 .npy Orbbec<-Aria prior (your current best).")
    ap.add_argument("--aria3d_dir", required=True, help="Aria hand_*.npy dir (world 3D).")
    ap.add_argument("--orbbec3d_dir", required=True, help="Orbbec 3D npz dir.")
    ap.add_argument("--scale_orbbec", type=float, default=1.0, help="Scale for Orbbec units (e.g., 0.001 if mm).")
    ap.add_argument("--huber", type=float, default=0.05, help="Huber delta (meters).")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--rot_sigma_deg", type=float, default=2.0, help="Std‑dev prior on rotation delta (deg). Set larger to allow more change.")
    ap.add_argument("--trans_sigma_m", type=float, default=None, help="Optional std‑dev prior on translation delta (m).")
    ap.add_argument("--dof_rot", type=str, default="1,1,1", help="Which rotation axes to free (e.g., '0,1,1' to lock wx).")
    ap.add_argument("--dof_trans", type=str, default="1,1,1", help="Which translation axes to free.")
    ap.add_argument("--out", required=True, help="Output 4x4 .npy for refined Orbbec<-Aria.")
    args = ap.parse_args()

    dof_rot   = tuple(int(x) for x in args.dof_rot.split(","))
    dof_trans = tuple(int(x) for x in args.dof_trans.split(","))

    P_A, P_O = collect_pairs(args.aria3d_dir, args.orbbec3d_dir, scale_orbbec=args.scale_orbbec)
    X_prior  = np.load(args.X_oa_prior).astype(np.float64)

    X_new, xi, rms = refine_with_priors(
        X_prior, P_A, P_O,
        huber=args.huber, iters=args.iters,
        rot_sigma_deg=args.rot_sigma_deg, trans_sigma_m=args.trans_sigma_m,
        dof_rot=dof_rot, dof_trans=dof_trans
    )
    print(f"[INFO] Δrot (deg) ~ |δw| = {np.degrees(np.linalg.norm(xi[:3])):.3f}")
    print(f"[INFO] Δt (m)     = {xi[3:]}  (in Orbbec world axes)")
    print(f"[INFO] Final RMS  = {rms*100:.1f} cm")
    np.save(args.out, X_new.astype(np.float32))
    print(f"[OK] wrote: {args.out}")

if __name__ == "__main__":
    main()

# python3 refine_se3_with_small_rotation_prior.py \  
#         --X_oa_prior ./X_from3D_T_orbbec_aria_save.npy \  
#         --aria3d_dir ../data/20250519_Testing/Aria/export/hand \  
#         --orbbec3d_dir orbbec_joints3d/20250519 \  
#         --scale_orbbec 0.001 \ 
#         --rot_sigma_deg 25.0 \ 
#         --iters 1000  \  
#         --out ./T_orbbec_world_aria_refined.npy --dof_rot 0,0,1 --dof_trans 0,0,0
