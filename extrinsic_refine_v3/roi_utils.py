
import numpy as np
import open3d as o3d

def _to_np(pcd):
    return np.asarray(pcd.points)

def crop_aabb(pcd, min_corner, max_corner):
    aabb = o3d.geometry.AxisAlignedBoundingBox(min_corner, max_corner)
    return pcd.crop(aabb)

def crop_sphere(pcd, center, radius):
    pts = _to_np(pcd)
    m = np.linalg.norm(pts - np.asarray(center), axis=1) <= radius
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts[m])
    if pcd.has_colors():
        out.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[m])
    if pcd.has_normals():
        out.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[m])
    return out

def crop_cylinder(pcd, center, axis, radius, z_min=None, z_max=None):
    pts = _to_np(pcd)
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    rel = pts - np.asarray(center, dtype=np.float64)
    z = rel @ axis
    proj = np.outer(z, axis)
    radial = rel - proj
    r = np.linalg.norm(radial, axis=1)
    m = r <= radius
    if z_min is not None:
        m = np.logical_and(m, z >= z_min)
    if z_max is not None:
        m = np.logical_and(m, z <= z_max)
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts[m])
    if pcd.has_colors():
        out.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[m])
    if pcd.has_normals():
        out.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[m])
    return out

def crop_plane_band(pcd, normal, offset_d, half_thickness):
    pts = _to_np(pcd)
    n = np.asarray(normal, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    d = float(offset_d)
    val = np.abs(pts @ n + d)
    m = val <= half_thickness
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts[m])
    if pcd.has_colors():
        out.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[m])
    if pcd.has_normals():
        out.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[m])
    return out

def apply_roi(pcd, roi_cfg: dict):
    if not roi_cfg:
        return pcd
    mode = roi_cfg.get("mode", None)
    if mode is None:
        return pcd
    if mode == "box":
        return crop_aabb(pcd, roi_cfg["min_corner"], roi_cfg["max_corner"])
    if mode == "sphere":
        return crop_sphere(pcd, roi_cfg["center"], roi_cfg["radius"])
    if mode == "cylinder":
        return crop_cylinder(pcd, roi_cfg["center"], roi_cfg["axis"], roi_cfg["radius"],
                             roi_cfg.get("z_min"), roi_cfg.get("z_max"))
    if mode == "plane_band":
        return crop_plane_band(pcd, roi_cfg["normal"], roi_cfg["d"], roi_cfg["half_thickness"])
    raise ValueError(f"Unknown ROI mode: {mode}")
