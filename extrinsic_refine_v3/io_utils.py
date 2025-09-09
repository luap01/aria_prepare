
import os
import numpy as np
import open3d as o3d
from glob import glob

def _np_to_o3d(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd

def read_point_cloud_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".ply", ".pcd", ".xyz"]:
        pcd = o3d.io.read_point_cloud(path)
        if pcd.is_empty():
            raise ValueError(f"Loaded empty point cloud from {path}")
        return pcd
    elif ext in [".npy", ".npz"]:
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.lib.npyio.NpzFile):
            if "points" in arr:
                pts = arr["points"]
                cols = arr["colors"] if "colors" in arr else None
            elif "xyz" in arr:
                pts = arr["xyz"]
                cols = arr["rgb"] if "rgb" in arr else None
            else:
                keys = list(arr.keys())
                if len(keys)==1:
                    pts = arr[keys[0]]
                    cols = None
                else:
                    raise ValueError(f"NPZ file {path} missing 'points'/'xyz' keys.")
        else:
            pts = arr
            cols = None
        pts = np.asarray(pts)
        if pts.ndim != 2 or pts.shape[1] not in [3,6]:
            raise ValueError(f"Unsupported array shape {pts.shape} in {path}")
        if pts.shape[1] == 6:
            xyz, rgb = pts[:,:3], pts[:,3:]
            return _np_to_o3d(xyz, rgb)
        else:
            return _np_to_o3d(pts)
    else:
        raise ValueError(f"Unsupported point cloud format: {ext}")

def load_clouds_from_dir(directory, pattern="*.ply", limit=None, sort=True):
    files = glob(os.path.join(directory, pattern))
    if sort:
        files.sort()
    if limit is not None:
        files = files[:limit]
    if len(files) == 0:
        raise ValueError(f"No files found in {directory} matching {pattern}")
    clouds = []
    for f in files:
        clouds.append(read_point_cloud_any(f))
    return clouds, files
