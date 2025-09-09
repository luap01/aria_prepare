# refine_icp_local.py
import numpy as np, open3d as o3d
from pathlib import Path
import copy

def down_normal(pcd, voxel, nn=80, rad_mult=3.0):
    q = pcd.voxel_down_sample(voxel)
    q.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=rad_mult*voxel, max_nn=nn))
    q.normalize_normals()
    return q

def crop_to_overlap(src, tgt, T_init, radius=0.04, expand=0.15):
    # Transform src by prior and crop both to AABB of tgt expanded by 'expand' (fractional growth)
    src_T = copy.deepcopy(src)
    src_T.transform(T_init.copy())

    aabb = tgt.get_axis_aligned_bounding_box()
    aabb = aabb.scale(1.0 + expand, aabb.get_center())  # e.g., expand=0.30 → grow box by 30%

    # If you intended absolute padding instead, replace the two lines above with:
    # minb, maxb = aabb.get_min_bound(), aabb.get_max_bound()
    # pad = np.array([expand, expand, expand], dtype=float)
    # aabb = o3d.geometry.AxisAlignedBoundingBox(minb - pad, maxb + pad)

    return src_T.crop(aabb), tgt.crop(aabb)

def refine_icp_local(src_path, tgt_path, T_init_path, out_prefix="refined", finest_voxel=0.01):
    src = o3d.io.read_point_cloud(src_path)
    tgt = o3d.io.read_point_cloud(tgt_path)
    # T_init = np.load(T_init_path)
    T_init = np.array([
        [-0.46918556,  0.8819397,   0.0452469,   0.069228  ],
        [-0.0348995,  -0.06971398,  0.99695635,  0.52026004],
        [ 0.88240975,  0.46617845,  0.06348804,  0.30815   ],
        [ 0.,          0.,          0.,          1.       ]
    ], dtype=np.float32)

    src_c, tgt_c = crop_to_overlap(src, tgt, T_init, expand=0.30)
    voxels = [finest_voxel*5, finest_voxel*2.5, finest_voxel]

    T = T_init.copy()
    huber = o3d.pipelines.registration.HuberLoss(k=1.0)
    est = o3d.pipelines.registration.TransformationEstimationPointToPlane(kernel=huber)

    for v in voxels:
        s = down_normal(src_c, v)
        t = down_normal(tgt_c, v)
        reg = o3d.pipelines.registration.registration_icp(
            s, t, max_correspondence_distance=2.5*v, init=T,
            estimation_method=est,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=80, relative_fitness=1e-6, relative_rmse=1e-6
            )
        )
        T = reg.transformation
        print(f"[ICP {v*100:.0f}mm] fit={reg.fitness:.3f} rmse={reg.inlier_rmse:.4f}")

    np.save(f"{out_prefix}_T.npy", T)
    src_aligned = copy.deepcopy(src)
    src_aligned.transform(T)
    o3d.io.write_point_cloud(f"{out_prefix}_src_in_tgt.ply", src_aligned)
    print("Saved:", f"{out_prefix}_T.npy", f"{out_prefix}_src_in_tgt.ply")

if __name__ == "__main__":
    # Example:
    refine_icp_local("noise_cloud.ply", "pointcloud_000060.ply", "T_init.npy",
                     out_prefix="prior_local", finest_voxel=0.01)
