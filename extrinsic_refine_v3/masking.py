
import numpy as np
import open3d as o3d
from collections import defaultdict

def build_static_cloud(pcd_list, voxel=0.01, min_frames_ratio=0.6):
    if not pcd_list or len(pcd_list)==0:
        raise ValueError("pcd_list is empty.")
    n = len(pcd_list)
    counts = defaultdict(int)
    for pcd in pcd_list:
        pts = np.asarray(pcd.points)
        if pts.size == 0: 
            continue
        vox = np.floor(pts / voxel).astype(np.int64)
        uniq = np.unique(vox, axis=0)
        for v in map(tuple, uniq):
            counts[v] += 1
    min_count = int(np.ceil(min_frames_ratio * n))
    static_keys = {k for k, c in counts.items() if c >= min_count}
    if len(static_keys) == 0:
        raise RuntimeError("No static voxels found. Try lowering min_frames_ratio or increasing voxel.")
    sums = {k: np.zeros(3, dtype=np.float64) for k in static_keys}
    freq = {k: 0 for k in static_keys}
    for pcd in pcd_list:
        pts = np.asarray(pcd.points)
        if pts.size == 0: 
            continue
        vox = np.floor(pts / voxel).astype(np.int64)
        for pt, key in zip(pts, map(tuple, vox)):
            if key in sums:
                sums[key] += pt
                freq[key] += 1
    out_pts = []
    for k in static_keys:
        if freq[k] > 0:
            out_pts.append(sums[k] / freq[k])
    out = o3d.geometry.PointCloud()
    if len(out_pts) == 0:
        raise RuntimeError("Static voxels had no contributing points. Adjust parameters.")
    out.points = o3d.utility.Vector3dVector(np.stack(out_pts, axis=0))
    return out
