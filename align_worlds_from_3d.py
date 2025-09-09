#!/usr/bin/env python3
import os, re, glob, argparse, json
import numpy as np

# ---------- filename patterns ----------
ARIA3D_RX = re.compile(r"^hand_([-\d]+)\.npy$")
NPZ3D_RX  = re.compile(r"^(\d+)\..*?_(left|right)_3d_keypoints\.npz$")

JOINTS_PER_HAND = 21

# ---------- I/O ----------
def load_aria_hand(path):
    """
    hand_<fid>.npy -> structured dict per side with:
      side_dict = {
        'conf' : float (if present; default 1.0),
        'wrist': (3,) or None,
        'palm' : (3,) or None,
        'lm'   : (21,3) or (N,3) or None  (left/right_landmarks_world)
      }
    Returns: { 'left': side_dict or None, 'right': side_dict or None }
    """
    d = np.load(path, allow_pickle=True).item()

    def get_side(side):
        out = {
            'conf': float(d.get(f"{side}_conf", 1.0)) if f"{side}_conf" in d else 1.0,
            'wrist': None,
            'palm': None,
            'lm': None
        }
        w = d.get(f"{side}_wrist", None)
        c = d.get(f"{side}_palm",  None)
        lm = d.get(f"{side}_landmarks_world", None)

        if w is not None:
            out['wrist'] = np.asarray(w, float).reshape(3)
        if c is not None:
            out['palm']  = np.asarray(c, float).reshape(3)
        if lm is not None:
            lm = np.asarray(lm, float)
            lm = lm.reshape((-1,3)) if lm.ndim == 2 else lm.reshape((JOINTS_PER_HAND, 3))
            out['lm'] = lm
        # Drop if nothing present
        if out['wrist'] is None and out['palm'] is None and out['lm'] is None:
            return None
        return out

    return {'left': get_side('left'), 'right': get_side('right')}

def load_orbbec_npz(dir_):
    """Returns maps: left[fid]->X21, right[fid]->X21 (each Nx3)."""
    left, right = {}, {}
    for p in glob.glob(os.path.join(dir_, "*.npz")):
        m = NPZ3D_RX.match(os.path.basename(p))
        if not m:
            continue
        fid = int(m.group(1)); side = m.group(2)
        with np.load(p) as data:
            if len(data.files) == 0:
                continue
            X = np.array(data[data.files[0]], float)
        if X.ndim == 1 and X.size % 3 == 0:
            X = X.reshape(-1,3)
        if X.ndim != 2 or X.shape[1] != 3:
            continue
        if side == "left":
            left[fid]  = X
        else:
            right[fid] = X
    return left, right

def palm_point(X21, mode="avg_mcp", idx=9):
    """Compute a palm proxy from 21-joint hand. Default: mean of MCP knuckles."""
    if mode == "avg_mcp":
        idxs = [1,5,9,13,17]
        idxs = [i for i in idxs if i < X21.shape[0]]
        return np.mean(X21[idxs], axis=0)
    elif mode == "index":
        if idx < 0 or idx >= X21.shape[0]:
            raise ValueError(f"palm_idx={idx} out of range for shape {X21.shape}")
        return X21[idx]
    else:
        raise ValueError("palm_mode must be 'avg_mcp' or 'index'")

# ---------- mapping helpers for landmarks ----------
def load_index_pairs(map_json, N_a, N_o):
    """
    Load landmark index mapping pairs (aria_idx, orbbec_idx).
    If map_json is None -> identity up to min(N_a, N_o).
    Accepted JSON formats:
      1) {"pairs": [[ia,io], ...]}
      2) {"map": {"ia":"io", ...}} or {"ia": io, ...}
      3) [[ia, io], ...]
    """
    if map_json is None:
        upto = int(min(N_a, N_o))
        return [(i, i) for i in range(upto)]

    with open(map_json, "r") as f:
        d = json.load(f)

    pairs = None
    if isinstance(d, list):
        if all(isinstance(x, list) and len(x) == 2 for x in d):
            pairs = [(int(a), int(b)) for a,b in d]
    elif isinstance(d, dict):
        if "pairs" in d and isinstance(d["pairs"], list):
            pairs = [(int(a), int(b)) for a,b in d["pairs"]]
        elif "map" in d and isinstance(d["map"], dict):
            pairs = [(int(a), int(b)) for a,b in d["map"].items()]
        else:
            # treat dict itself as map
            pairs = [(int(a), int(b)) for a,b in d.items()]

    if pairs is None:
        raise ValueError(f"Unrecognized mapping format in {map_json}")

    # Filter to valid ranges
    return [(ia,io) for ia,io in pairs if 0 <= ia < N_a and 0 <= io < N_o]

def parse_indices_list(s):
    """Parse '0,1,5,9' -> [0,1,5,9]. Empty/None -> None (use all)."""
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [int(p) for p in parts]

# ---------- pairing ----------
def build_pairs(aria_dir, orbbec_dir, use_left=True, use_right=True,
                palm_mode="avg_mcp", palm_idx=9,
                fid_min=None, fid_max=None, dist_tol_m=0.15,
                use_landmarks=True, lm_indices=None, map_json=None,
                scale_orbbec=1.0):
    """
    Returns:
        P_O: [N,3] (Orbbec world)
        P_A: [N,3] (Aria world)
        used: list of fids used at least once
    Also prints stats on how many pairs came from wrist/palm vs landmarks.
    """
    left_map, right_map = load_orbbec_npz(orbbec_dir)

    P_O, P_A = [], []
    used_fids = []
    cnt_wp, cnt_lm = 0, 0

    for path in sorted(glob.glob(os.path.join(aria_dir, "hand_*.npy"))):
        m = ARIA3D_RX.match(os.path.basename(path))
        if not m:
            continue
        token = m.group(1)
        try:
            fid = int(token)
        except Exception:
            fid = int(token.lstrip("0") or "0")

        if (fid_min is not None and fid < fid_min) or (fid_max is not None and fid > fid_max):
            continue

        sides = load_aria_hand(path)
        for side, maps, use_flag in (("left",  left_map,  use_left),
                                     ("right", right_map, use_right)):
            if not use_flag:
                continue
            side_dict = sides.get(side, None)
            if side_dict is None:
                continue
            if side_dict.get('conf', 1.0) <= 0:
                continue
            if fid not in maps:
                continue

            Xo = np.asarray(maps[fid], float) * float(scale_orbbec)
            if Xo.ndim != 2 or Xo.shape[1] != 3:
                continue

            used_this_frame = False

            # 1) Wrist + palm (if both present on Aria and we can compute palm on Orbbec)
            if side_dict.get('wrist', None) is not None and side_dict.get('palm', None) is not None:
                wrist_o = Xo[0]
                palm_o  = palm_point(Xo, mode=palm_mode, idx=palm_idx)

                wrist_a = side_dict['wrist']
                palm_a  = side_dict['palm']

                # sanity gate on wrist-palm lengths
                dO = np.linalg.norm(wrist_o - palm_o)
                dA = np.linalg.norm(wrist_a - palm_a)
                ok = (np.isfinite(dO) and np.isfinite(dA) and dO >= 1e-6 and dA >= 1e-6 and abs(dO - dA) <= dist_tol_m)
                if ok and np.isfinite(wrist_a).all() and np.isfinite(palm_a).all() \
                      and np.isfinite(wrist_o).all() and np.isfinite(palm_o).all():
                    P_O.extend([wrist_o, palm_o])
                    P_A.extend([wrist_a, palm_a])
                    cnt_wp += 2
                    used_this_frame = True

            # 2) 21-landmarks (if requested and available on both sides)
            if use_landmarks and (side_dict.get('lm', None) is not None) and (Xo.shape[0] >= 1):
                La = np.asarray(side_dict['lm'], float)
                if La.ndim != 2 or La.shape[1] != 3:
                    La = La.reshape(-1,3)

                # filter out non-finite rows on both sides
                valid_a = np.isfinite(La).all(axis=1)
                valid_o = np.isfinite(Xo).all(axis=1)

                # build index pairs
                pairs = load_index_pairs(map_json, La.shape[0], Xo.shape[0])  # identity up to min() if map_json=None

                # optional subset of aria lm indices
                if lm_indices is not None:
                    lm_set = set(lm_indices)
                    pairs = [(ia, io) for (ia, io) in pairs if ia in lm_set]

                # keep only pairs with valid rows
                pairs = [(ia, io) for (ia, io) in pairs if valid_a[ia] and valid_o[io]]

                if pairs:
                    Po = np.stack([Xo[io] for (ia, io) in pairs], axis=0)
                    Pa = np.stack([La[ia] for (ia, io) in pairs], axis=0)
                    P_O.extend(Po.tolist())
                    P_A.extend(Pa.tolist())
                    cnt_lm += len(pairs)
                    used_this_frame = True

            if used_this_frame:
                used_fids.append(fid)

    if not P_O:
        raise RuntimeError("No paired 3D points collected. "
                           "Check paths, fid naming, conf>0, palm_mode, use_landmarks, and mapping/index settings.")

    print(f"[INFO] Added pairs: wrist/palm={cnt_wp}, landmarks={cnt_lm}, total={cnt_wp + cnt_lm}")
    return np.asarray(P_O, float), np.asarray(P_A, float), used_fids

# ---------- rigid alignment (Umeyama + robust IRLS) ----------
def umeyama_rigid(P, Q, w=None):
    """Find R,t so that R*P + t ≈ Q. No scale. P,Q: Nx3."""
    assert P.shape == Q.shape and P.shape[1] == 3
    N = P.shape[0]
    if w is None:
        w = np.ones(N)
    w = w.astype(float)
    w = w / (w.sum() + 1e-12)
    muP = (w[:,None] * P).sum(axis=0)
    muQ = (w[:,None] * Q).sum(axis=0)
    X = P - muP
    Y = Q - muQ
    Sigma = X.T @ (w[:,None] * Y)
    U, S, Vt = np.linalg.svd(Sigma)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
    t = muQ - R @ muP
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3]  = t
    return T

def robust_align(P, Q, iters=10, huber=0.05, R_fixed=None):
    """
    Robust IRLS with Huber (delta=huber meters).
    If R_fixed is provided (3x3), solve only translation t.
    """
    if R_fixed is None:
        T = umeyama_rigid(P, Q)
        R = T[:3,:3]; t = T[:3,3]
    else:
        R = R_fixed.copy()
        t = (Q - P @ R.T).mean(axis=0)

    for _ in range(iters):
        r = (P @ R.T + t) - Q  # [N,3]
        d = np.linalg.norm(r, axis=1) + 1e-12
        w = np.ones_like(d)
        mask = d > huber
        w[mask] = huber / d[mask]

        if R_fixed is None:
            T = umeyama_rigid(P, Q, w=w)
            R = T[:3,:3]; t = T[:3,3]
        else:
            # weighted mean for translation with fixed rotation
            t = (w[:,None] * (Q - P @ R.T)).sum(axis=0) / (w.sum() + 1e-12)

    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3]  = t
    return T

# ---------- utilities ----------
def so3_geodesic_deg(R1, R2):
    R = R1.T @ R2
    tr = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Align Orbbec world to Aria world using wrist/palm + 21-landmark 3D correspondences.")
    ap.add_argument("--aria3d_dir", required=True, help="Dir with Aria hand_*.npy (contains wrist/palm and optionally 21-landmarks in Aria world).")
    ap.add_argument("--orbbec3d_dir", required=True, help="Dir with <fid>.0_cam1.0_left/right_3d_keypoints.npz (Nx3 in Orbbec world).")
    ap.add_argument("--use_left", action="store_true", help="Use left hand if available.")
    ap.add_argument("--use_right", action="store_true", help="Use right hand if available.")
    ap.set_defaults(use_left=True, use_right=True)

    # Palm construction on Orbbec side (if 'palm' not explicit)
    ap.add_argument("--palm_mode", choices=["avg_mcp","index"], default="avg_mcp", help="How to get palm point from Orbbec 21-joint hand.")
    ap.add_argument("--palm_idx", type=int, default=9, help="If palm_mode=index, which Orbbec joint index to treat as palm.")

    # Landmark usage / mapping
    ap.add_argument("--use_landmarks", action="store_true", help="Include 21-landmark correspondences when present.")
    ap.set_defaults(use_landmarks=True)
    ap.add_argument("--lm_indices", type=str, default="", help="Comma-separated Aria landmark indices to use (default: all).")
    ap.add_argument("--map_json", type=str, default=None, help="Optional JSON with landmark index mapping pairs (aria_idx, orbbec_idx).")

    # Gating & scaling
    ap.add_argument("--fid_min", type=int, default=None)
    ap.add_argument("--fid_max", type=int, default=None)
    ap.add_argument("--dist_tol_m", type=float, default=0.15, help="Gate pairs where |d_wrist-palm^O - d_wrist-palm^A| > tol.")
    ap.add_argument("--scale_orbbec", type=float, default=1.0, help="Multiply Orbbec 3D by this (e.g., 0.001 if mm).")

    # Solver controls
    ap.add_argument("--huber_m", type=float, default=0.05, help="Huber delta for robust IRLS (meters).")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--R_prior", type=str, default=None, help="Optional .npy 4x4: fix rotation to this (solve translation only).")
    ap.add_argument("--out_prefix", default="./X_from_3d", help="Prefix for saved outputs.")
    args = ap.parse_args()

    lm_idx_list = parse_indices_list(args.lm_indices)

    # 1) Build 3D↔3D pairs (wrist/palm + landmarks)
    P_O, P_A, used = build_pairs(
        args.aria3d_dir, args.orbbec3d_dir,
        use_left=args.use_left, use_right=args.use_right,
        palm_mode=args.palm_mode, palm_idx=args.palm_idx,
        fid_min=args.fid_min, fid_max=args.fid_max,
        dist_tol_m=args.dist_tol_m,
        use_landmarks=args.use_landmarks,
        lm_indices=lm_idx_list,
        map_json=args.map_json,
        scale_orbbec=args.scale_orbbec
    )
    print(f"[INFO] Collected {P_O.shape[0]} 3D correspondences from {len(set(used))} frames.")

    # 2) Optional rotation prior (fix R, solve only t)
    R_fix = None
    if args.R_prior:
        Xp = np.load(args.R_prior).astype(float)
        Xp = np.linalg.inv(Xp)
        if Xp.shape != (4,4):
            raise ValueError(f"R_prior {args.R_prior} is not 4x4.")
        R_fix = Xp[:3,:3]
        print("[INFO] Using rotation prior from:", args.R_prior)

    # 3) Robust alignment
    X_ao = robust_align(P_O, P_A, iters=args.iters, huber=args.huber_m, R_fixed=R_fix)  # Aria <- Orbbec
    X_oa = np.linalg.inv(X_ao)

    # 4) Report residuals
    resid = (P_O @ X_ao[:3,:3].T + X_ao[:3,3]) - P_A
    d = np.linalg.norm(resid, axis=1)
    print(f"[INFO] Residual (meters): mean={d.mean():.4f}, median={np.median(d):.4f}, "
          f"95p={np.percentile(d,95):.4f}, max={d.max():.4f}")
    if args.R_prior is not None:
        deg = so3_geodesic_deg(Xp[:3,:3], X_ao[:3,:3])
        print(f"[INFO] Rotation change vs prior: {deg:.2f}°")

    # 5) Save
    np.save(f"{args.out_prefix}_T_aria_orbbec.npy", X_ao)
    np.save(f"{args.out_prefix}_T_orbbec_aria.npy", X_oa)
    print(f"[OK] Saved:\n  {args.out_prefix}_T_aria_orbbec.npy\n  {args.out_prefix}_T_orbbec_aria.npy")

if __name__ == "__main__":
    main()


# python align_worlds_from_3d.py \
#   --aria3d_dir ../data/20250519_Testing/Aria/export/hand \
#   --orbbec3d_dir ../HaMuCo/test_set_result/2025-08-20_22:37/3d_kps/20250519 \
#   --use_left --use_right \
#   --use_landmarks --palm_mode avg_mcp \
#   --huber_m 0.05 \
#   --out_prefix ./X_from3D --iters 100 --scale_orbbec 0.001