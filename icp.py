import numpy as np
from scipy.optimize import least_squares
import cv2

def pixels_to_rays(uv, K):
    """uv: (N,2) in rectified image; K: 3x3. Returns unit-bearing (N,3)."""
    fx, fy = K[0,0], K[1,1]; cx, cy = K[0,2], K[1,2]
    x = (uv[:,0] - cx) / fx
    y = (uv[:,1] - cy) / fy
    d = np.stack([x, y, np.ones_like(x)], axis=1)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    return d.astype(np.float64)

def pack(rvec, tvec):
    return np.concatenate([rvec.ravel(), tvec.ravel()])

def unpack(x):
    rvec = x[:3].reshape(3,1)
    tvec = x[3:].reshape(3,1)
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec

def point_to_ray_residuals(x, X_o, rays_A):
    """x= [rvec(3), tvec(3)] parameterizes T_AO. X_o: (N,3) Orbbec-world points.
       rays_A: (N,3) unit rays in Aria cam (rectified)."""
    R, t = unpack(x)
    pA = (R @ X_o.T + t).T  # (N,3)
    # residual: shortest distance from pA to ray through origin along rays_A
    cross = np.cross(pA, rays_A)
    return np.linalg.norm(cross, axis=1)

def refine_T_AO_point_to_ray(X_o, uv_A, K_A, R_init, t_init, weights=None):
    rays = pixels_to_rays(uv_A, K_A)
    rvec_init, _ = cv2.Rodrigues(R_init.astype(np.float64))
    x0 = pack(rvec_init, t_init.astype(np.float64))
    if weights is None:
        weights = np.ones(len(uv_A), dtype=np.float64)

    def fun(x):
        res = point_to_ray_residuals(x, X_o, rays)
        return np.sqrt(weights) * res

    # Huber is good here; f_scale ~ a few mm if your units are meters.
    sol = least_squares(fun, x0, method="trf", loss="huber", f_scale=0.01, max_nfev=200)
    R_opt, t_opt = unpack(sol.x)
    return R_opt, t_opt, sol

# After solving, you can still compute standard reprojection with cv2.projectPoints:
def project_points_cv2(X_o, R, t, K):
    p_h = np.array([p_orbbec_world_xyz[0], p_orbbec_world_xyz[1], p_orbbec_world_xyz[2], 1.0], dtype=np.float32)
    p_cam = (M @ p_h)[:3]
    return uv.reshape(-1,2)


# Build uv_A from your GT (rectified & rotated image)
# Example: keep first 21 points, and weights from the visibility flags
def parse_gt(flat):
    pts, w = [], []
    for i in range(0, 21*3, 3):
        pts.append([flat[i], flat[i+1]])
        # flags: 1 or 2; give 2x weight to '2'
        w.append(1.0 if flat[i+2]==1 else 2.0)
    return np.array(pts, dtype=np.float64), np.array(w, dtype=np.float64)

uv_left, w_left = parse_gt(left_gt_flat)    # your GT list
uv_right, w_right = parse_gt(right_gt_flat)

# Stack both hands
X_o = np.vstack([X_left_o, X_right_o])      # your Orbbec 3D (meters)
uv_A = np.vstack([uv_left, uv_right])
weights = np.concatenate([w_left, w_right])

R_opt, t_opt, sol = refine_T_AO_point_to_ray(X_o, uv_A, K_A, R_init, t_init, weights)

# Check improvement
uv_pred = project_points_cv2(X_o, R_opt, t_opt, K_A)
err = np.linalg.norm(uv_pred - uv_A, axis=1)
print("reproj mean/median px:", err.mean(), np.median(err))
