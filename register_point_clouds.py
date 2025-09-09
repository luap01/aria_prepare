#!/usr/bin/env python3
"""
register_point_clouds.py

Rigidly register two point clouds and output the 4x4 transformation matrix
that maps the SOURCE into the TARGET frame.

Pipeline:
1) Preprocess: voxel downsample, estimate normals, compute FPFH features.
2) Global alignment: RANSAC on FPFH correspondences.
3) Local refinement: ICP (point-to-plane).
4) Save/print transformation and (optionally) the transformed source.

Requirements:
- Python 3.8+
- open3d >= 0.17.0  (pip install open3d)
- numpy

Example:
python register_point_clouds.py source.ply target.ply \
  --voxel-size 0.02 --save-matrix transform.txt --save-transformed aligned_source.ply

The transformation printed/saved is a 4x4 matrix T such that:
    target ≈ T @ source
i.e., points in the SOURCE frame are mapped to the TARGET frame.
"""

import argparse
import sys
from pathlib import Path
import numpy as np

try:
    import open3d as o3d
except ImportError as e:
    print("Error: open3d is required. Install with: pip install open3d", file=sys.stderr)
    raise

def load_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"Loaded empty point cloud: {path}")
    # Remove NaNs/inf if present
    pts = np.asarray(pcd.points)
    mask = np.isfinite(pts).all(axis=1)
    if mask.sum() != len(pts):
        pcd = pcd.select_by_index(np.where(mask)[0])
    return pcd

def estimate_normals(pcd: o3d.geometry.PointCloud, radius: float, max_nn: int = 30):
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    pcd.normalize_normals()
    return pcd

def preprocess_point_cloud(pcd: o3d.geometry.PointCloud, voxel_size: float):
    # Downsample
    pcd_down = pcd.voxel_down_sample(voxel_size)
    if pcd_down.is_empty():
        raise ValueError("Downsampled point cloud is empty. Try a smaller --voxel-size.")
    # Estimate normals for features/ICP
    normal_radius = voxel_size * 2.0
    estimate_normals(pcd_down, radius=normal_radius, max_nn=50)
    # Compute FPFH
    fpfh_radius = voxel_size * 5.0
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius, max_nn=100),
    )
    return pcd_down, fpfh

def execute_global_registration(src_down, tgt_down, src_fpfh, tgt_fpfh, voxel_size: float, ransac_n: int = 4):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=ransac_n,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    return result

def refine_registration(source, target, init_trans: np.ndarray, voxel_size: float):
    # Point-to-plane ICP usually converges better on real scans
    distance_threshold = voxel_size * 0.6
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    return result

def save_matrix(matrix: np.ndarray, path: Path):
    # Save both human-readable and machine-friendly formats
    if path.suffix.lower() in {".npy"}:
        np.save(path, matrix)
    else:
        np.savetxt(path, matrix, fmt="%.8f")

def main():
    parser = argparse.ArgumentParser(description="Register two point clouds and output the 4x4 transform (SOURCE -> TARGET).")
    parser.add_argument("--source", type=Path, help="Path to source point cloud (e.g., .ply/.pcd/.xyz)")
    parser.add_argument("--target", type=Path, help="Path to target point cloud (e.g., .ply/.pcd/.xyz)")
    parser.add_argument("--voxel-size", type=float, default=0.02, help="Downsampling voxel size (in same units as the point clouds). Default: 0.02")
    parser.add_argument("--save-matrix", type=Path, default=None, help="Where to save the 4x4 transform (txt or npy).")
    parser.add_argument("--save-transformed", type=Path, default=None, help="Save the source transformed into the target frame to this path (e.g., aligned_source.ply).")
    parser.add_argument("--no-refine", action="store_true", help="Skip ICP refinement; only use global alignment.")
    parser.add_argument("--verbose", action="store_true", help="Print extra info.")
    args = parser.parse_args()

    if args.verbose:
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    else:
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    source = load_point_cloud(args.source)
    target = load_point_cloud(args.target)

    # Preprocess
    src_down, src_fpfh = preprocess_point_cloud(source, args.voxel_size)
    tgt_down, tgt_fpfh = preprocess_point_cloud(target, args.voxel_size)

    # Global alignment
    if args.verbose:
        print("[Global] Running RANSAC feature-based alignment...")
    global_result = execute_global_registration(src_down, tgt_down, src_fpfh, tgt_fpfh, args.voxel_size)
    if args.verbose:
        print("[Global] Fitness: %.4f, Inlier RMSE: %.6f" % (global_result.fitness, global_result.inlier_rmse))

    T = global_result.transformation

    # Optional refinement
    if not args.no_refine:
        if args.verbose:
            print("[Refine] Running ICP (point-to-plane)...")
        # For ICP we want reasonably dense data with normals
        estimate_normals(source, radius=args.voxel_size * 2.0, max_nn=60)
        estimate_normals(target, radius=args.voxel_size * 2.0, max_nn=60)
        refine_result = refine_registration(source, target, T, args.voxel_size)
        if args.verbose:
            print("[Refine] Fitness: %.4f, Inlier RMSE: %.6f" % (refine_result.fitness, refine_result.inlier_rmse))
        if refine_result.fitness > 0:
            T = refine_result.transformation

    # Output
    np.set_printoptions(precision=6, suppress=True)
    print("\n=== Transformation (SOURCE -> TARGET) ===")
    print(T)

    if args.save_matrix is not None:
        save_matrix(T, args.save_matrix)
        print(f"\nSaved matrix to: {args.save_matrix}")

    if args.save_transformed is not None:
        src_aligned = source.transform(T.copy())
        ok = o3d.io.write_point_cloud(str(args.save_transformed), source)
        # O3D transforms in-place; 'source' is now aligned
        if ok:
            print(f"Saved transformed source to: {args.save_transformed}")
        else:
            print(f"Failed to save transformed source to: {args.save_transformed}", file=sys.stderr)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
