#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
import numpy as np
import open3d as o3d

def load_pcd(path: Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        raise ValueError(f"Empty point cloud: {path}")
    return pcd

def bbox_diag(pcd):
    aabb = pcd.get_axis_aligned_bounding_box()
    ext = np.asarray(aabb.get_extent(), dtype=float)
    return float(np.linalg.norm(ext)), ext

def rescale_if_mm(pcd, force_mm=None):
    """If diag > 1000 assume mm -> convert to meters. Override with force_mm=True/False."""
    if force_mm is not None:
        if force_mm:
            pcd.scale(1.0/1000.0, center=pcd.get_center())
        return force_mm
    diag, _ = bbox_diag(pcd)
    if diag > 1000.0:  # likely millimeters
        pcd.scale(1.0/1000.0, center=pcd.get_center())
        return True
    return False

def down_normal(pcd, voxel, nn=60, rad_mult=3.0):
    q = pcd.voxel_down_sample(voxel)
    q.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=rad_mult*voxel, max_nn=nn
        )
    )
    q.normalize_normals()
    return q

def compute_fpfh(pcd, voxel, rad_mult=2.5):
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=rad_mult*voxel, max_nn=100)
    )

def ransac_with_prior(src_Tinit, tgt, voxel):
    f_src = compute_fpfh(src_Tinit, voxel)
    f_tgt = compute_fpfh(tgt, voxel)
    dist = 1.5 * voxel
    checker_edge   = o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.95)
    checker_dist   = o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist)
    checker_normal = o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(np.deg2rad(30))

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_Tinit, tgt, f_src, f_tgt,
        mutual_filter=True,
        max_correspondence_distance=dist,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[checker_edge, checker_normal, checker_dist],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 1000)
    )
    return result

def fgr_with_prior(src_Tinit, tgt, voxel):
    f_src = compute_fpfh(src_Tinit, voxel)
    f_tgt = compute_fpfh(tgt, voxel)
    opt = o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=1.0*voxel
    )
    return o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        src_Tinit, tgt, f_src, f_tgt, opt
    )

def quick_icp_score(src, tgt, T, voxel):
    s = down_normal(src, voxel)
    t = down_normal(tgt, voxel)
    reg = o3d.pipelines.registration.registration_icp(
        s, t, max_correspondence_distance=2.5*voxel, init=T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20)
    )
    return reg.fitness, reg.inlier_rmse

def multiscale_icp(src, tgt, T_start, voxels=(0.10, 0.05, 0.02)):
    T = T_start.copy()
    huber = o3d.pipelines.registration.HuberLoss(k=1.0)
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(kernel=huber)
    hist = []
    for v in voxels:
        s = down_normal(src, v)
        t = down_normal(tgt, v)
        reg = o3d.pipelines.registration.registration_icp(
            s, t, max_correspondence_distance=2.5*v, init=T,
            estimation_method=estimation,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=80, relative_fitness=1e-6, relative_rmse=1e-6
            )
        )
        T = reg.transformation
        hist.append({"voxel_m": float(v), "fitness": float(reg.fitness), "rmse_m": float(reg.inlier_rmse)})
        print(f"[ICP @ {v*100:.0f} mm] fitness={reg.fitness:.3f} rmse={reg.inlier_rmse:.4f}")
    return T, hist

def main():
    ap = argparse.ArgumentParser(description="Register two room point clouds (Open3D): RANSAC/FGR + robust point-to-plane ICP")
    ap.add_argument("--src", required=True, type=Path, help="Source PLY (will be moved into target frame)")
    ap.add_argument("--tgt", required=True, type=Path, help="Target PLY (reference frame)")
    ap.add_argument("--t_init", type=Path, default=None, help="Optional 4x4 NumPy .npy initial guess (src->tgt)")
    ap.add_argument("--assume-mm", action="store_true", help="Treat coordinates as millimeters (convert to meters)")
    ap.add_argument("--out_prefix", type=str, default="registration",
                    help="Prefix for outputs (PLY/NPY/JSON)")
    ap.add_argument("--min_voxel", type=float, default=None, help="Override finest ICP voxel (m); default auto")
    args = ap.parse_args()

    src = load_pcd(args.src)
    tgt = load_pcd(args.tgt)

    # Optional unit conversion
    scaled_src = rescale_if_mm(src, force_mm=True if args.assume_mm else None)
    scaled_tgt = rescale_if_mm(tgt, force_mm=True if args.assume_mm else None)

    # Scene scale -> choose voxels
    diag_src, ext_src = bbox_diag(src)
    diag_tgt, ext_tgt = bbox_diag(tgt)
    scene_diag = max(diag_src, diag_tgt)
    base_voxel = np.clip(scene_diag/100.0, 0.01, 0.20)  # ~1/100 of scene, clamped
    if args.min_voxel is not None:
        base_voxel = float(args.min_voxel)
    voxels = [float(base_voxel*5.0), float(base_voxel*2.5), float(base_voxel)]

    print(f"Source points: {len(src.points):,}   Target points: {len(tgt.points):,}")
    print(f"Extents src (m): {ext_src}   tgt (m): {ext_tgt}")
    print(f"Auto voxel pyramid (m): {voxels}")

    T_init = np.array([
        [-0.46918556,  0.8819397,   0.0452469,   0.069228  ],
        [-0.0348995,  -0.06971398,  0.99695635,  0.52026004],
        [ 0.88240975,  0.46617845,  0.06348804,  0.30815   ],
        [ 0.,          0.,          0.,          1.       ]
    ], dtype=np.float32)
    T_init = np.linalg.inv(T_init)  # Orbbec-world -> Aria-world
    # Prior
    # if args.t_init and args.t_init.exists():
    #     T_init = np.load(args.t_init)
    #     if T_init.shape != (4,4): raise ValueError("t_init must be a 4x4 matrix .npy")
    # else:
    #     T_init = np.eye(4)

    # Global: transform source by prior so thresholds can be tight
    src_Tinit = o3d.geometry.PointCloud(src)  # shallow copy
    src_Tinit.transform(T_init.copy())

    # Downsample for global stage
    coarse_voxel = voxels[1]
    src_coarse = down_normal(src_Tinit, coarse_voxel)
    tgt_coarse = down_normal(tgt,       coarse_voxel)

    print("=== Global registration: RANSAC with prior ===")
    ransac = ransac_with_prior(src_coarse, tgt_coarse, coarse_voxel)
    T_ransac = ransac.transformation @ T_init
    print(f" RANSAC fitness={ransac.fitness:.3f} rmse={ransac.inlier_rmse:.4f}")

    print("=== Global registration: FGR with prior ===")
    fgr = fgr_with_prior(src_coarse, tgt_coarse, coarse_voxel)
    T_fgr = fgr.transformation @ T_init

    # Pick better init by a quick ICP check
    fit_r, rmse_r = quick_icp_score(src, tgt, T_ransac, voxels[0])
    fit_f, rmse_f = quick_icp_score(src, tgt, T_fgr,   voxels[0])
    if fit_r > fit_f:
        T0, init_name, init_fit, init_rmse = T_ransac, "RANSAC", fit_r, rmse_r
    else:
        T0, init_name, init_fit, init_rmse = T_fgr, "FGR", fit_f, rmse_f
    print(f"Chosen init: {init_name}  fitness={init_fit:.3f} rmse={init_rmse:.4f}")

    # Local refinement: robust point-to-plane ICP
    print("=== Multi-scale robust point-to-plane ICP ===")
    T_refined, icp_hist = multiscale_icp(src, tgt, T0, voxels=voxels)

    # Save outputs
    out_prefix = Path(args.out_prefix)
    T_path = out_prefix.with_suffix("").as_posix() + "_T_refined.npy"
    np.save(T_path, T_refined)

    src_aligned = o3d.geometry.PointCloud(src)
    src_aligned.transform(T_refined.copy())
    aligned_path = out_prefix.with_suffix("").as_posix() + "_src_in_tgt.ply"
    o3d.io.write_point_cloud(aligned_path, src_aligned)

    # Pretty print
    def fmt(T):
        return "\n".join(["[ " + "  ".join(f"{v:+.6f}" for v in row) + " ]" for row in T])
    print("\nRefined transform (src -> tgt):")
    print(fmt(T_refined))

    # JSON report
    report = {
        "source_points": len(src.points),
        "target_points": len(tgt.points),
        "source_extent_m": ext_src.tolist(),
        "target_extent_m": ext_tgt.tolist(),
        "scene_diag_m": float(scene_diag),
        "voxel_pyramid_m": voxels,
        "scaled_source_mm_to_m": bool(scaled_src),
        "scaled_target_mm_to_m": bool(scaled_tgt),
        "init_choice": init_name,
        "init_quick_icp_fitness": float(init_fit),
        "init_quick_icp_rmse_m": float(init_rmse),
        "icp_history": icp_hist,
        "T_refined_4x4": T_refined.tolist(),
        "output_transform_npy": T_path,
        "output_aligned_ply": aligned_path
    }
    json_path = out_prefix.with_suffix("").as_posix() + "_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved:\n - {T_path}\n - {aligned_path}\n - {json_path}")

if __name__ == "__main__":
    np.set_printoptions(suppress=False, linewidth=120, precision=6)
    main()
