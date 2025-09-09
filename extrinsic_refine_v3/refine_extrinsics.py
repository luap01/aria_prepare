
#!/usr/bin/env python3
import os, json, argparse, yaml, numpy as np
import open3d as o3d

from io_utils import read_point_cloud_any, load_clouds_from_dir
from pair_loader import load_paired_dirs
from masking import build_static_cloud
from icp_utils import multiscale_icp, chamfer_distance, transform_pcd, plane_snap_rotation
from roi_utils import apply_roi
from report_utils import save_transform_json, save_metrics_json, save_metrics_csv

def parse_args():
    p = argparse.ArgumentParser(description="Rigid extrinsic refinement Orbbec → Aria (flexible inputs)")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return p.parse_args()

def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def as_np4x4(x):
    arr = np.array(x, dtype=np.float64)
    if arr.shape != (4,4):
        raise ValueError("Expected 4x4 matrix, got {}".format(arr.shape))
    return arr

def scale_and_transform(pcd, scale=1.0, T=None):
    q = o3d.geometry.PointCloud(pcd)
    if scale is not None and scale != 1.0:
        q.scale(scale, center=(0,0,0))
    if T is not None:
        q.transform(T)
    return q

def make_static(pcd_list, fused_path, v_static, min_ratio, final_downsample=None, roi_cfg=None, scale=1.0, preT=None):
    """
    Accept either a list of frames or a single fused path. Returns a static point cloud.
    Applies optional ROI, scale, and pre-transform (for Orbbec) BEFORE aggregation.
    """
    if pcd_list and len(pcd_list) > 0:
        # Apply ROI, scale, preT per frame
        prepped = []
        for p in pcd_list:
            q = scale_and_transform(p, scale=scale, T=preT)
            if roi_cfg:
                q = apply_roi(q, roi_cfg)
            prepped.append(q)
        if len(prepped) == 1:
            return prepped[0]
        return build_static_cloud(prepped, voxel=v_static, min_frames_ratio=min_ratio)
    if fused_path:
        p = read_point_cloud_any(fused_path)
        p = scale_and_transform(p, scale=scale, T=preT)
        if roi_cfg:
            p = apply_roi(p, roi_cfg)
        if final_downsample and final_downsample > 0:
            p = p.voxel_down_sample(final_downsample)
        return p
    raise ValueError("Neither a list of frames nor a fused_path was provided.")

def main():
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = cfg.get("output_dir", "refine_output")
    os.makedirs(out_dir, exist_ok=True)

    # Load inputs (each side can be directory or fused file)
    O_list = []
    A_list = []
    if cfg.get("orbbec_dir"):
        O_list, _, = load_clouds_from_dir(cfg["orbbec_dir"], cfg.get("orbbec_glob","*.ply"), cfg.get("limit"))
    if cfg.get("aria_dir"):
        A_list, _, = load_clouds_from_dir(cfg["aria_dir"], cfg.get("aria_glob","*.ply"), cfg.get("limit"))

    orbbec_fused = cfg.get("orbbec_fused_path")
    aria_fused   = cfg.get("aria_fused_path")

    if not (O_list or orbbec_fused):
        raise ValueError("Provide either 'orbbec_dir' or 'orbbec_fused_path'.")
    if not (A_list or aria_fused):
        raise ValueError("Provide either 'aria_dir' or 'aria_fused_path'.")

    # Per-side ROI (fall back to shared 'roi' if specific one not given)
    roi_O = cfg.get("roi_orbbec", cfg.get("roi", None))
    roi_A = cfg.get("roi_aria",   cfg.get("roi", None))

    # Unit/frame fixes for Orbbec
    orbbec_scale = float(cfg.get("orbbec_scale", 1.0))
    preT = cfg.get("orbbec_pre_transform", None)
    preT = np.array(preT, dtype=np.float64) if preT is not None else None

    # Static aggregation params
    v_static = float(cfg.get("static_voxel", 0.01))
    min_ratio = float(cfg.get("static_min_frames_ratio", 0.6))
    final_voxel = float(cfg.get("final_voxel", 0.005))

    # Build static clouds
    O_static = make_static(O_list, orbbec_fused, v_static, min_ratio, final_voxel, roi_cfg=roi_O, scale=orbbec_scale, preT=preT)
    A_static = make_static(A_list, aria_fused,   v_static, min_ratio, final_voxel, roi_cfg=roi_A, scale=1.0, preT=None)

    # Save statics for inspection
    o3d.io.write_point_cloud(os.path.join(out_dir, "O_static.ply"), O_static)
    o3d.io.write_point_cloud(os.path.join(out_dir, "A_static.ply"), A_static)

    # Initial T and optional plane snap
    T_init = as_np4x4(cfg["T_init"])
    if cfg.get("plane_snap", False):
        try:
            T_init = plane_snap_rotation(T_init, O_static, A_static, plane_dist_thresh=cfg.get("plane_snap_dist", 0.01))
        except Exception as e:
            print("Plane snap failed:", e)

    # ICP refinement
    voxel_schedule = cfg.get("voxel_schedule", [0.03, 0.015, 0.007])
    corr_schedule = cfg.get("corr_dist_schedule", [0.06, 0.03, 0.015])
    max_iters = cfg.get("max_iters", [120, 80, 60])
    method = cfg.get("icp_method", "point_to_plane")

    T_refined, icp_final = multiscale_icp(O_static, A_static, T_init, voxel_schedule, corr_schedule, max_iters, method=method)

    # Metrics + save
    O_aligned = transform_pcd(O_static, T_refined)
    chamfer = chamfer_distance(O_aligned, A_static, truncate=0.05)
    metrics = {
        "icp_inlier_rmse": float(icp_final.inlier_rmse),
        "icp_fitness_inlier_ratio": float(icp_final.fitness),
        "static_chamfer_mean_m": float(chamfer),
        "num_points_orbbec_static": int(len(O_static.points)),
        "num_points_aria_static": int(len(A_static.points)),
        "voxel_schedule_m": voxel_schedule,
        "corr_dist_schedule_m": corr_schedule,
        "icp_method": method
    }

    save_transform_json(os.path.join(out_dir, "T_A_from_O.json"), T_refined,
                        meta={"convention": "T maps Orbbec world into Aria world (meters)",
                              "orbbec_scale_applied": orbbec_scale,
                              "orbbec_pre_transform_applied": preT.tolist() if preT is not None else None})
    save_metrics_json(os.path.join(out_dir, "metrics.json"), metrics)
    save_metrics_csv(os.path.join(out_dir, "metrics.csv"), metrics)

    try:
        both = O_aligned + A_static
        o3d.io.write_point_cloud(os.path.join(out_dir, "aligned_overlay.ply"), both)
    except Exception as e:
        print("Visualization save failed:", e)

    print("=== Refinement complete ===")
    print("Final inlier RMSE:", metrics["icp_inlier_rmSE"] if 'icp_inlier_rmSE' in metrics else metrics["icp_inlier_rmse"])
    print("Static Chamfer (m):", metrics["static_chamfer_mean_m"])
    print("Saved to:", out_dir)

if __name__ == "__main__":
    main()
