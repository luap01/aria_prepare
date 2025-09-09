
"""
Two-hand 3D keypoint propagation from fused point clouds (template)
==================================================================

What this does
--------------
- Loads fused point clouds (*.ply) for selected frames.
- Uses your seed 3D keypoints (first N seed frames) to compute per-bone lengths and
  initialize a constant-velocity (CV) tracker with irregular dt (when you skip frames).
- For each processed frame:
  * Predicts joint positions from the last two processed frames (CV model).
  * Crops the point cloud around each hand's predicted center.
  * For each joint, finds neighbors in a small radius and computes a robust (Huber) mean.
  * Enforces bone-length constraints (from your seeds) to keep the skeleton plausible.
  * Optionally runs a short Kalman smoothing pass (CV, variable dt).
- Saves left/right 3D keypoints and per-joint quality metrics (neighbor count, residual).

Assumptions
-----------
- 21-keypoint MediaPipe-style order per hand (0..20):
  0: wrist
  Thumb: 1-4  (CMC, MCP, IP, tip)
  Index: 5-8  (MCP, PIP, DIP, tip)
  Middle: 9-12
  Ring: 13-16
  Little: 17-20
- Your seed GT exists as .npy files with shape (21,3) in meters.

This is a template: adapt the I/O to your dataset structure.
Requires: numpy, scipy

Author: (your name)
"""

import os
import math
import json
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from scipy.spatial import cKDTree

# ------------------------
# Lightweight PLY loader
# ------------------------

def _parse_ply_header(f) -> dict:
    f.seek(0)
    hdr_bytes = b''
    while True:
        line = f.readline()
        if not line:
            raise ValueError("Unexpected EOF in header")
        hdr_bytes += line
        if line.strip() == b'end_header':
            break
    lines = hdr_bytes.decode('ascii', errors='ignore').splitlines()
    fmt, version = None, None
    elements = []
    cur_elem = None
    for s in lines:
        if s.startswith('format '):
            parts = s.split()
            fmt = parts[1]
            version = parts[2] if len(parts) > 2 else None
        elif s.startswith('comment') or s.startswith('obj_info'):
            continue
        elif s.startswith('element '):
            parts = s.split()
            name = parts[1]
            count = int(parts[2])
            cur_elem = {"name": name, "count": count, "properties": []}
            elements.append(cur_elem)
        elif s.startswith('property '):
            parts = s.split()
            if parts[1] == 'list':
                list_count_type = parts[2]; list_type = parts[3]; name = parts[4]
                cur_elem["properties"].append({"name": name, "type": list_type, "is_list": True, "list_count_type": list_count_type})
            else:
                ptype = parts[1]; name = parts[2]
                cur_elem["properties"].append({"name": name, "type": ptype, "is_list": False})
        elif s == 'end_header':
            break
    return {"format": fmt, "version": version, "header_length": len(hdr_bytes), "elements": elements}

def _ply_type_to_np(ptype: str, endian: str) -> np.dtype:
    mapping = {
        'char': 'i1', 'int8': 'i1',
        'uchar': 'u1', 'uint8': 'u1',
        'short': 'i2', 'int16': 'i2',
        'ushort': 'u2', 'uint16': 'u2',
        'int': 'i4', 'int32': 'i4',
        'uint': 'u4', 'uint32': 'u4',
        'float': 'f4', 'float32': 'f4',
        'double': 'f8', 'float64': 'f8',
    }
    if ptype not in mapping:
        raise ValueError(f"Unsupported PLY type: {ptype}")
    return np.dtype(endian + mapping[ptype])

def load_ply_vertices(path: str, want_normals: bool=True, want_rgb: bool=False, max_points: Optional[int]=None):
    with open(path, 'rb') as f:
        hdr = _parse_ply_header(f)
        fmt = hdr['format']
        if fmt == 'binary_little_endian':
            endian = '<'
        elif fmt == 'binary_big_endian':
            endian = '>'
        else:
            raise NotImplementedError("ASCII PLY not supported in this minimal loader")
        assert hdr['elements'][0]['name'] == 'vertex', "This loader expects 'vertex' to be the first element"
        vert_elem = hdr['elements'][0]
        N = vert_elem['count']
        props = vert_elem['properties']
        fields = []
        for p in props:
            if p['is_list']:
                raise NotImplementedError("List properties in 'vertex' not supported")
            fields.append((p['name'], _ply_type_to_np(p['type'], endian)))
        dtype = np.dtype(fields)
        f.seek(hdr['header_length'])
        data = np.fromfile(f, dtype=dtype, count=N)
    xyz = np.stack([data['x'], data['y'], data['z']], axis=1)
    out = {'xyz': xyz}
    if want_normals and all(n in data.dtype.names for n in ['nx','ny','nz']):
        nrm = np.stack([data['nx'], data['ny'], data['nz']], axis=1)
        out['normals'] = nrm
    if want_rgb and all(n in data.dtype.names for n in ['red','green','blue']):
        rgb = np.stack([data['red'], data['green'], data['blue']], axis=1)
        out['rgb'] = rgb
    if max_points is not None and xyz.shape[0] > max_points:
        rng = np.random.RandomState(42)
        idx = rng.choice(xyz.shape[0], size=max_points, replace=False)
        for k in list(out.keys()):
            out[k] = out[k][idx]
    return out

# ------------------------
# Hand skeleton utilities
# ------------------------

HAND_EDGES = [
    (0,1),(1,2),(2,3),(3,4),        # thumb
    (0,5),(5,6),(6,7),(7,8),        # index
    (0,9),(9,10),(10,11),(11,12),   # middle
    (0,13),(13,14),(14,15),(15,16), # ring
    (0,17),(17,18),(18,19),(19,20), # little
]

def compute_bone_lengths(joints_list: List[np.ndarray]) -> np.ndarray:
    """Average bone lengths over seed frames. joints: (21,3) each."""
    Ls = []
    for J in joints_list:
        lens = []
        for p,c in HAND_EDGES:
            v = J[c] - J[p]
            lens.append(np.linalg.norm(v))
        Ls.append(lens)
    return np.asarray(Ls).mean(axis=0)

def hand_center(J: np.ndarray) -> np.ndarray:
    """Center as mean of wrist+MCPs (wrist 0; MCPs 1,5,9,13,17)."""
    idx = [0,1,5,9,13,17]
    return J[idx].mean(axis=0)

# ------------------------
# Geometry helpers
# ------------------------

def voxel_downsample(xyz: np.ndarray, voxel: float) -> Tuple[np.ndarray, np.ndarray]:
    """Simple grid downsample by rounding coordinates to voxel size. Returns (downsampled_xyz, kept_indices)."""
    q = np.floor(xyz / voxel).astype(np.int64)
    # unique rows
    uq, idx = np.unique(q, axis=0, return_index=True)
    return xyz[idx], idx

def huber_weights(r: np.ndarray, delta: float) -> np.ndarray:
    w = np.ones_like(r)
    mask = r > delta
    w[mask] = delta / (r[mask] + 1e-9)
    return w

def robust_mean(points: np.ndarray, init: Optional[np.ndarray]=None, delta: float=0.01, iters: int=5) -> Optional[np.ndarray]:
    if points.shape[0] == 0:
        return None
    x = points.mean(axis=0) if init is None else init.copy()
    for _ in range(iters):
        r = np.linalg.norm(points - x, axis=1)
        w = huber_weights(r, delta)
        x = (points * w[:,None]).sum(axis=0) / (w.sum() + 1e-9)
    return x

def enforce_bone_lengths(P: np.ndarray, edges: List[Tuple[int,int]], target_lengths: np.ndarray, iters: int=2) -> np.ndarray:
    """Project child joints along their parent direction to match target lengths; keeps parents fixed."""
    Q = P.copy()
    for _ in range(iters):
        for e_i,(p,c) in enumerate(edges):
            v = Q[c] - Q[p]
            n = np.linalg.norm(v)
            if n < 1e-8:
                continue
            d = v / n
            Q[c] = Q[p] + d * target_lengths[e_i]
    return Q

# ------------------------
# Tracking / Propagation
# ------------------------

@dataclass
class PropagationParams:
    voxel_size: float = 0.003    # 3 mm
    roi_radius_base: float = 0.12  # 12 cm base crop radius per hand
    roi_radius_vel_scale: float = 0.4  # extra meters per 1 m/s * dt
    search_radius_base: float = 0.018  # 18 mm
    search_radius_tip: float = 0.014   # 14 mm for fingertips
    search_radius_wrist: float = 0.022 # 22 mm
    huber_delta: float = 0.012  # 12 mm
    min_neighbors: int = 35
    max_neighbors: int = 2000

def per_joint_search_radius(j: int, P: PropagationParams) -> float:
    if j == 0:
        return P.search_radius_wrist
    if j in [4,8,12,16,20]:
        return P.search_radius_tip
    return P.search_radius_base

def refine_hand_joints_in_cloud(
    cloud_xyz: np.ndarray,
    prev_J: np.ndarray,
    prev_prev_J: Optional[np.ndarray],
    dt_seconds: float,
    params: PropagationParams,
    normals: Optional[np.ndarray]=None
) -> Tuple[np.ndarray, Dict]:
    """
    Refine joint positions for one hand from the current cloud and previous frames.
    Returns (J, info) where J is (21,3) and info carries per-joint quality metrics.
    """
    # Constant-velocity prediction
    if prev_prev_J is not None:
        v = (prev_J - prev_prev_J) / max(dt_seconds, 1e-6)
        J_pred = prev_J + v * dt_seconds
    else:
        J_pred = prev_J.copy()

    center = hand_center(J_pred)
    # ROI crop radius grows with estimated speed
    speed = 0.0 if prev_prev_J is None else float(np.linalg.norm(v.mean(axis=0)))
    roi_r = params.roi_radius_base + params.roi_radius_vel_scale * speed * dt_seconds + 0.03  # +3 cm margin
    dists = np.linalg.norm(cloud_xyz - center[None,:], axis=1)
    roi_mask = dists <= roi_r
    roi = cloud_xyz[roi_mask]
    if roi.shape[0] == 0:
        # Nothing in ROI -> fall back to prediction
        return J_pred, {"roi_points": 0, "per_joint": []}

    # Downsample ROI for speed
    roi_ds, keep_idx = voxel_downsample(roi, voxel=params.voxel_size)
    tree = cKDTree(roi_ds)

    J = J_pred.copy()
    per_joint_info = []
    for j in range(21):
        r = per_joint_search_radius(j, params)
        idxs = tree.query_ball_point(J_pred[j], r=r)
        # If too many, subsample nearest by distance
        if len(idxs) > params.max_neighbors:
            # Query k nearest large K, then cut by radius
            dists, nn_idx = tree.query(J_pred[j], k=params.max_neighbors)
            idxs = nn_idx.tolist() if np.ndim(nn_idx)==0 else nn_idx.tolist()
        pts = roi_ds[idxs] if len(idxs)>0 else np.empty((0,3))
        if pts.shape[0] < params.min_neighbors:
            # Low support -> keep predicted
            per_joint_info.append({"j": j, "support": int(pts.shape[0]), "residual": None})
            J[j] = J_pred[j]
            continue
        mu = robust_mean(pts, init=J_pred[j], delta=params.huber_delta, iters=5)
        J[j] = mu if mu is not None else J_pred[j]
        resid = float(np.mean(np.linalg.norm(pts - J[j], axis=1)))
        per_joint_info.append({"j": j, "support": int(pts.shape[0]), "residual": resid})

    # Enforce bone lengths
    info = {"roi_points": int(roi.shape[0])}
    info["per_joint"] = per_joint_info
    return J, info

# ------------------------
# Kalman smoothing (CV with variable dt)
# ------------------------

class CVKalman:
    def __init__(self, q: float=1e-4, r: float=2e-4):
        self.q = q  # process noise base
        self.r = r  # measurement noise base
        self.x = None  # (6,) pos+vel
        self.P = None  # (6,6)

    def init(self, p0: np.ndarray):
        self.x = np.zeros((6,))
        self.x[:3] = p0
        self.P = np.eye(6) * 1e-2

    def step(self, z: np.ndarray, dt: float, meas_var_scale: float=1.0) -> np.ndarray:
        F = np.eye(6); F[:3,3:] = np.eye(3)*dt
        H = np.zeros((3,6)); H[:,:3] = np.eye(3)
        Q = np.eye(6) * self.q * max(dt,1e-3)**2
        R = np.eye(3) * self.r * meas_var_scale
        # predict
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        # update
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        return self.x[:3]

def smooth_trajectory(joints_seq: List[np.ndarray], dts: List[float], per_joint_meas_var: Optional[List[np.ndarray]]=None) -> List[np.ndarray]:
    """
    Apply independent CV Kalman per joint over the sequence.
    joints_seq: list of (21,3) arrays (measurements)
    dts: list of dt between consecutive entries (seconds), len = len(seq)-1
    per_joint_meas_var: optional list of (21,) variance scales from quality metrics.
    """
    J = len(joints_seq[0])
    K = len(joints_seq)
    out = [None]*K
    # forward pass
    filters = [CVKalman() for _ in range(J)]
    for j in range(J):
        filters[j].init(joints_seq[0][j])
    out[0] = joints_seq[0].copy()
    for t in range(1, K):
        dt = dts[t-1]
        Z = joints_seq[t]
        prev = out[t-1].copy()
        cur = np.zeros_like(Z)
        for j in range(J):
            mv = 1.0
            if per_joint_meas_var is not None and per_joint_meas_var[t] is not None:
                mv = per_joint_meas_var[t][j]
            cur[j] = filters[j].step(Z[j], dt, meas_var_scale=mv)
        out[t] = cur
    return out

# ------------------------
# Driver (example)
# ------------------------

def load_seed_gt(seed_dir: str, frames: List[int], hand: str) -> List[np.ndarray]:
    # Expect files like '{seed_dir}/{hand}_{frame:06d}.npy' with shape (21,3)
    seq = []
    for f in frames:
        p = os.path.join(seed_dir, f"{hand}_{f:06d}.npy")
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        seq.append(np.load(p))
    return seq

def process_sequence(
    cloud_dir: str,
    out_dir: str,
    frames: List[int],
    step: int,
    seed_frames: List[int],
    seed_dir: str,
    params: PropagationParams=PropagationParams()
):
    os.makedirs(out_dir, exist_ok=True)
    # Load seeds and compute bone lengths (per hand)
    left_seeds = load_seed_gt(seed_dir, seed_frames, 'L')
    right_seeds = load_seed_gt(seed_dir, seed_frames, 'R')
    L_len = compute_bone_lengths(left_seeds)
    R_len = compute_bone_lengths(right_seeds)

    # Initialize previous joints from last seed frame
    prev_L = left_seeds[-1]
    prev_prev_L = left_seeds[-2] if len(left_seeds) >= 2 else None
    prev_R = right_seeds[-1]
    prev_prev_R = right_seeds[-2] if len(right_seeds) >= 2 else None

    last_frame = seed_frames[-1]
    seq_frames = [f for f in frames if f > last_frame and ((f - last_frame) % step == 0)]
    all_L, all_R = [], []
    all_info = {}

    prev_time = last_frame
    for f in seq_frames:
        ply_path = os.path.join(cloud_dir, f"pointcloud_{f:06d}.ply")
        cloud = load_ply_vertices(ply_path, want_normals=False, want_rgb=False, max_points=None)
        xyz = cloud['xyz']
        # Estimate dt in seconds; if you know FPS, dt = (f - prev_time)/FPS. Set FPS here:
        FPS = 30.0
        dt = (f - prev_time) / FPS

        # Left hand
        JL_pred, infoL = refine_hand_joints_in_cloud(xyz, prev_L, prev_prev_L, dt, params)
        # Enforce bone lengths left
        JL = enforce_bone_lengths(JL_pred, HAND_EDGES, L_len, iters=2)

        # Right hand
        JR_pred, infoR = refine_hand_joints_in_cloud(xyz, prev_R, prev_prev_R, dt, params)
        JR = enforce_bone_lengths(JR_pred, HAND_EDGES, R_len, iters=2)

        all_L.append(JL); all_R.append(JR)
        all_info[f] = {"L": infoL, "R": infoR, "dt": dt}

        # update prev
        prev_prev_L, prev_L = prev_L, JL
        prev_prev_R, prev_R = prev_R, JR
        prev_time = f

        # Save per-frame
        np.savez_compressed(os.path.join(out_dir, f"keys_{f:06d}.npz"), L=JL, R=JR, info=all_info[f])

    # Optional smoothing (requires dts list)
    if len(all_L) >= 2:
        dts = [all_info[seq_frames[i+1]]['dt'] for i in range(len(seq_frames)-1)]
        # You can derive per-joint measurement variance from residuals if you like:
        per_joint_var = []
        for i, f in enumerate(seq_frames):
            per_joint_var.append(None)  # simple default
        L_sm = smooth_trajectory(all_L, dts, per_joint_meas_var=None)
        R_sm = smooth_trajectory(all_R, dts, per_joint_meas_var=None)
        # Overwrite smoothed outputs
        for i, f in enumerate(seq_frames):
            np.savez_compressed(os.path.join(out_dir, f"keys_{f:06d}.npz"), L=L_sm[i], R=R_sm[i], info=all_info[f])

    # Save an index json
    with open(os.path.join(out_dir, "index.json"), "w") as fp:
        json.dump({"frames": seq_frames, "step": step}, fp, indent=2)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud_dir", type=str, required=True, help="Folder containing pointcloud_XXXXXX.ply files")
    ap.add_argument("--seed_dir", type=str, required=True, help="Folder with seed 3D GT .npy files: L_XXXXXX.npy, R_XXXXXX.npy")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--step", type=int, default=10, help="Frame step (if you skip 10-20 frames)")
    args = ap.parse_args()

    frames = list(range(args.start, args.end+1))
    seed_frames = [f for f in frames if os.path.exists(os.path.join(args.seed_dir, f"L_{f:06d}.npy"))]
    if len(seed_frames) < 1:
        raise RuntimeError("No seed frames found in seed_dir")

    process_sequence(
        cloud_dir=args.cloud_dir,
        out_dir=args.out_dir,
        frames=frames,
        step=args.step,
        seed_frames=seed_frames,
        seed_dir=args.seed_dir,
        params=PropagationParams()
    )
