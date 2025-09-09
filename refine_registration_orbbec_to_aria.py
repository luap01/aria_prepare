# pip install open3d==0.18.0 (or latest)
import numpy as np
import open3d as o3d
from pathlib import Path

# ---------- Config ----------
ORBBEC_PCD = "pointcloud_000060.ply"   # Orbbec room cloud (Orbbec-world coords)
ARIA_PCD   = "noise_cloud.ply"     # Aria room cloud (Aria-world coords)
T_INIT_NPY = "T_init.npy"       # 4x4 SE(3): Orbbec-world -> Aria-world
# Voxel pyramid (coarse -> fine):
# VOXELS = [0.10 / 1000, 0.05 / 1000, 0.02 / 1000]     # meters
VOXELS = [0.10, 0.05, 0.02]     # meters
NORMAL_RADIUS_MULT = 3.0 # 3.0
RANSAC_VOXEL = 0.05            # meters (feature radius / distance thresholds scale with this)
MAX_RANSAC_ITERS = 100000
MAX_RANSAC_VALID = 1000

# If your scene is very small/large, scale the values above accordingly.

# ---------- Helpers ----------
def load_pcd(path):
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        raise ValueError(f"Empty point cloud: {path}")
    return pcd

def down_normal(pcd, voxel):
    p = pcd.voxel_down_sample(voxel)
    p.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS_MULT*voxel, max_nn=60
        )
    )
    p.normalize_normals()
    return p

def compute_fpfh(pcd, voxel):
    # FPFH radius ~ 2-5x voxel; search_nn can stay default
    radius = 2.5 * voxel
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100)
    )

def ransac_with_prior(pcd_o_Tinit, pcd_a, voxel):
    """Global registration using feature matching + RANSAC.
       We pass the Orbbec cloud already pre-transformed by T_init to tighten thresholds.
    """
    f_o = compute_fpfh(pcd_o_Tinit, voxel)
    f_a = compute_fpfh(pcd_a, voxel)

    distance_thresh = 1.5 * voxel  # tight thanks to prior
    checker_edge   = o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.95)  # ~no scale change
    checker_dist   = o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_thresh)
    checker_normal = o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(np.deg2rad(30))

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd_o_Tinit, pcd_a, f_o, f_a,
        mutual_filter=True,
        max_correspondence_distance=distance_thresh,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),  # no scaling
        ransac_n=4,
        checkers=[checker_edge, checker_normal, checker_dist],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            MAX_RANSAC_ITERS, MAX_RANSAC_VALID
        )
    )
    return result

def fast_global_registration(pcd_o_Tinit, pcd_a, voxel):
    f_o = compute_fpfh(pcd_o_Tinit, voxel)
    f_a = compute_fpfh(pcd_a, voxel)
    option = o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=1.0 * voxel
    )
    return o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        pcd_o_Tinit, pcd_a, f_o, f_a, option
    )

def multiscale_icp(pcd_o, pcd_a, T_start, voxels):
    T = T_start.copy()
    huber = o3d.pipelines.registration.HuberLoss(k=1.0)
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(kernel=huber)
    for v in voxels:
        p_o = down_normal(pcd_o, v)
        p_a = down_normal(pcd_a, v)
        # Reject distant/oblique matches by lowering correspondence distance
        max_corr = 2.5 * v
        reg = o3d.pipelines.registration.registration_icp(
            p_o, p_a, max_corr, T,
            estimation_method=estimation,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=80, relative_fitness=1e-6, relative_rmse=1e-6
            )
        )
        T = reg.transformation
        print(f"[ICP @ {v*100:.0f}mm] fitness={reg.fitness:.3f}, rmse={reg.inlier_rmse:.4f}")
    return T

# ---------- Main ----------
if __name__ == "__main__":
    p_o = load_pcd(ORBBEC_PCD)
    p_a = load_pcd(ARIA_PCD)
    # T_init = np.load(T_INIT_NPY)  # Orbbec->Aria (4x4)
    T_init = np.array([
        [-0.46918556,  0.8819397,   0.0452469,   0.069228  ],
        [-0.0348995,  -0.06971398,  0.99695635,  0.52026004],
        [ 0.88240975,  0.46617845,  0.06348804,  0.30815   ],
        [ 0.,          0.,          0.,          1.       ]
    ], dtype=np.float32)
    T_init = np.linalg.inv(T_init)

    # Preprocess once for the global stage
    p_o_coarse = down_normal(p_o, RANSAC_VOXEL)
    p_a_coarse = down_normal(p_a, RANSAC_VOXEL)

    # Bring Orbbec cloud near Aria with the prior before global matching
    p_o_Tinit = p_o_coarse.transform(T_init.copy())

    print("=== Global registration (RANSAC) with prior ===")
    ransac = ransac_with_prior(p_o_Tinit, p_a_coarse, RANSAC_VOXEL)
    print(f"RANSAC fitness={ransac.fitness:.3f}, rmse={ransac.inlier_rmse:.4f}")

    # Compose back to get Orbbec->Aria (RANSAC)
    # ransac.transformation aligns (p_o_Tinit) -> p_a, so relative to original Orbbec:
    T_global = ransac.transformation @ T_init

    # (Optional) If RANSAC is weak (low fitness), try FGR and pick the better one
    print("=== Fast Global Registration (FGR) with prior ===")
    fgr = fast_global_registration(p_o_Tinit, p_a_coarse, RANSAC_VOXEL)
    T_fgr = fgr.transformation @ T_init

    # Pick the better global init by higher fitness after a quick ICP check
    def quick_icp_score(T):
        reg = o3d.pipelines.registration.registration_icp(
            down_normal(p_o, VOXELS[0]),
            down_normal(p_a, VOXELS[0]),
            max_correspondence_distance=2.5*VOXELS[0],
            init=T,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=15)
        )
        return reg.fitness, reg.inlier_rmse

    fit_g, rmse_g = quick_icp_score(T_global)
    fit_f, rmse_f = quick_icp_score(T_fgr)
    T0 = T_global if (fit_g > fit_f) else T_fgr
    print(f"Chosen init -> fitness={max(fit_g,fit_f):.3f}, rmse={rmse_g if fit_g>fit_f else rmse_f:.4f}")

    print("=== Multi-scale point-to-plane ICP (robust) ===")
    T_refined = multiscale_icp(p_o, p_a, T0, VOXELS)

    # Save results
    np.save("T_refined.npy", T_refined)
    p_o_aligned = p_o.transform(T_refined.copy())
    o3d.io.write_point_cloud("orbbec_in_aria_refined.ply", p_o_aligned)
    print("Saved T_refined.npy and orbbec_in_aria_refined.ply")

    # Visual check (press 'j'/'k' to toggle normals if needed)
    p_o_aligned.paint_uniform_color([0.1, 0.7, 0.1])
    p_a.paint_uniform_color([0.6, 0.6, 0.6])
    o3d.visualization.draw_geometries([p_a, p_o_aligned])
