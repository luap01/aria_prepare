
import numpy as np
import open3d as o3d
from typing import List

def preprocess(pcd: o3d.geometry.PointCloud, voxel: float, nb_neighbors=20, std_ratio=1.5, normal_radius=None):
    q = pcd.voxel_down_sample(voxel)
    if len(q.points) == 0:
        return q
    q, _ = q.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    if normal_radius is None:
        normal_radius = 3.0 * voxel
    if len(q.points) > 0:
        q.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=50))
        q.orient_normals_consistent_tangent_plane(50)
    return q

def multiscale_icp(source: o3d.geometry.PointCloud,
                   target: o3d.geometry.PointCloud,
                   T_init: np.ndarray,
                   voxel_schedule: List[float],
                   corr_dist_schedule: List[float],
                   max_iters: List[int] = None,
                   method: str = "point_to_plane"):
    assert len(voxel_schedule) == len(corr_dist_schedule)
    if max_iters is None:
        max_iters = [100] * len(voxel_schedule)
    T = T_init.copy()
    last = None
    for voxel, corr, iters in zip(voxel_schedule, corr_dist_schedule, max_iters):
        s = preprocess(source, voxel)
        t = preprocess(target, voxel)
        if len(s.points) == 0 or len(t.points) == 0:
            raise RuntimeError(f"Empty point clouds at voxel={voxel}.")
        try:
            loss = o3d.pipelines.registration.TukeyLoss(k=corr)
        except Exception:
            loss = None
        if method == "point_to_plane":
            est = (o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
                   if loss is not None else
                   o3d.pipelines.registration.TransformationEstimationPointToPlane())
        elif method == "generalized":
            est = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
        else:
            est = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        last = o3d.pipelines.registration.registration_icp(
            s, t, corr, T, est,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters)
        )
        T = last.transformation
    return T, last

def fit_plane(pcd: o3d.geometry.PointCloud, distance_threshold=0.01, ransac_n=3, num_iterations=2000):
    if len(pcd.points) == 0:
        raise RuntimeError("Cannot segment plane from empty cloud")
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    a, b, c, d = plane_model
    n = np.array([a, b, c], dtype=np.float64)
    n /= np.linalg.norm(n) + 1e-12
    return plane_model, n, inliers

def rotation_aligning_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]], dtype=np.float64)
    R = np.eye(3) + K + K @ K * ((1 - c) / (s**2))
    return R

def plane_snap_rotation(T_init: np.ndarray,
                        src: o3d.geometry.PointCloud,
                        tgt: o3d.geometry.PointCloud,
                        plane_dist_thresh=0.01):
    _, n_src, _ = fit_plane(src, distance_threshold=plane_dist_thresh)
    _, n_tgt, _ = fit_plane(tgt, distance_threshold=plane_dist_thresh)
    R = rotation_aligning_vectors(n_src, n_tgt)
    T = T_init.copy()
    T[:3,:3] = R @ T[:3,:3]
    return T

def chamfer_distance(a: o3d.geometry.PointCloud, b: o3d.geometry.PointCloud, truncate=None) -> float:
    from scipy.spatial import cKDTree
    pa = np.asarray(a.points)
    pb = np.asarray(b.points)
    if len(pa)==0 or len(pb)==0:
        return float("nan")
    ta = cKDTree(pa); tb = cKDTree(pb)
    d1, _ = tb.query(pa, k=1)
    d2, _ = ta.query(pb, k=1)
    if truncate is not None:
        d1 = np.minimum(d1, truncate); d2 = np.minimum(d2, truncate)
    return float(0.5 * (np.mean(d1) + np.mean(d2)))

def transform_pcd(pcd: o3d.geometry.PointCloud, T: np.ndarray) -> o3d.geometry.PointCloud:
    q = o3d.geometry.PointCloud(pcd)
    q.transform(T)
    return q
