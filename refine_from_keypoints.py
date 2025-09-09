import numpy as np
from scipy.optimize import least_squares

# ---------- SE(3) utilities ----------
def hat(w):
    wx, wy, wz = w
    return np.array([[0,-wz,wy],[wz,0,-wx],[-wy,wx,0]], float)

def se3_exp(xi):
    """xi = [wx, wy, wz, tx, ty, tz] -> 4x4"""
    w = xi[:3]; t = xi[3:]
    th = np.linalg.norm(w)
    if th < 1e-12:
        R = np.eye(3)
        V = np.eye(3)
    else:
        K = hat(w/th)
        R = np.eye(3) + np.sin(th)*K + (1-np.cos(th))*(K@K)
        V = (np.eye(3) + (1-np.cos(th))*K + (th-np.sin(th))*(K@K)) / th
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=V@t
    return T

# ---------- Projection models ----------
def project_pinhole(K, Xc):
    """Xc: Nx3 in camera frame (no distortion)"""
    x = Xc[:,0]/Xc[:,2]; y = Xc[:,1]/Xc[:,2]
    u = K[0,0]*x + K[0,2]; v = K[1,1]*y + K[1,2]
    return np.stack([u,v], axis=1)

def project_fisheye_equidistant(K, D, Xc):
    """
    OpenCV fisheye model (equidistant): r = f * theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
    K: 3x3, D: (k1,k2,k3,k4)
    """
    x = Xc[:,0] / Xc[:,2]
    y = Xc[:,1] / Xc[:,2]
    r = np.sqrt(x*x + y*y) + 1e-12
    theta = np.arctan(r)
    k1,k2,k3,k4 = D
    theta_d = theta*(1 + k1*theta**2 + k2*theta**4 + k3*theta**6 + k4*theta**8)
    scale = theta_d / r
    xd = x*scale; yd = y*scale
    u = K[0,0]*xd + K[0,2]; v = K[1,1]*yd + K[1,2]
    return np.stack([u,v], axis=1)

# ---------- Solver ----------
def refine_T_from_keypoints(
    T_prior,              # 4x4 Orbbec->Aria (good initial guess)
    K,                    # 3x3 intrinsics of Aria camera
    cam_T_world,          # [T] list/array of 4x4 camera-to-world in Aria frame
    P3D_orbbec,           # [T,Kj,3] hand joints in Orbbec world
    kp2d,                 # [T,Kj,2] measured pixels in Aria image
    valid=None,           # [T,Kj] bool mask (True = use)
    model="fisheye",      # "fisheye" or "pinhole"
    D=(0,0,0,0),          # fisheye coeffs if model="fisheye"
    dof_mask=(1,1,1,1,1,1),  # lock some DOFs, e.g. (0,0,0,1,1,1) for translation-only
    huber_px=3.0,         # robust cutoff in pixels
    prior_sigma=(np.deg2rad(1), np.deg2rad(1), np.deg2rad(1), 0.01, 0.01, 0.01)  # small prior
):
    T_prior = T_prior.copy()
    cam_T_world = np.asarray(cam_T_world)
    P3D_orbbec = np.asarray(P3D_orbbec)
    kp2d = np.asarray(kp2d)
    T_frames, Kj, _ = P3D_orbbec.shape
    if valid is None: valid = np.ones((T_frames,Kj), bool)

    dof_mask = np.array(dof_mask, float)
    free_idx = np.where(dof_mask>0.5)[0]

    def pack(x_free):
        xi = np.zeros(6); xi[free_idx] = x_free
        return xi

    if model == "fisheye":
        proj = lambda K, Xc: project_fisheye_equidistant(K, D, Xc)
    else:
        proj = project_pinhole

    # Prebuild homogeneous points per frame
    Pw_o = np.concatenate([P3D_orbbec, np.ones((T_frames,Kj,1))], axis=2)  # [T,K,4]

    # Residuals
    def residuals(x_free):
        xi = pack(x_free)
        T = se3_exp(xi) @ T_prior
        res = []
        for t in range(T_frames):
            # Orbbec world -> Aria world
            Pw_a = (Pw_o[t] @ T.T)[:, :3]  # [K,3]
            # Aria world -> camera
            Twc = cam_T_world[t]                    # camera-to-world
            Tcw = np.linalg.inv(Twc)                # world-to-camera
            Xc = (np.c_[Pw_a, np.ones((Kj,1))] @ Tcw.T)[:, :3]
            uv = proj(K, Xc)
            m = valid[t]
            r = (uv[m] - kp2d[t,m]).reshape(-1)
            res.append(r)

        # Small quadratic prior on xi (keeps update tiny)
        w = 1.0/np.array(prior_sigma, float)
        r_prior = (w*xi).astype(float)
        return np.concatenate(res + [r_prior])

    # Huber on residuals (manually; LM does Gauss-Newton)
    def residuals_huber(x_free):
        r = residuals(x_free)
        s = np.abs(r)
        w = np.ones_like(s)
        mask = s > huber_px
        w[mask] = huber_px / s[mask]
        return r * w

    x0 = np.zeros(len(free_idx))
    sol = least_squares(residuals_huber, x0, method="lm", max_nfev=50)
    T_ref = se3_exp(pack(sol.x)) @ T_prior
    return T_ref, sol

# ---------------- Example usage ----------------
T_prior = np.load("T_init.npy")
K = np.array([
    [   0.,  -300.,   511.5],
    [ 300.,     0.,   511.5],
    [   0.,     0.,     1. ]], dtype=np.float32)
# K = ...  # 3x3
# D = (k1,k2,k3,k4)  # if fisheye
cam_T_world = [...]  # list/array of 4x4 for each frame (Aria cam-to-world)
P3D_orbbec = ...     # [T,Kj,3]
kp2d = ...           # [T,Kj,2]
# valid = ...          # [T,Kj] booleans (drop occluded joints)
T_ref, sol = refine_T_from_keypoints(T_prior, K, cam_T_world, P3D_orbbec, kp2d, valid,
                                     model="fisheye", D=None,
                                     dof_mask=(0,0,0,1,1,1))  # start with translation-only
np.save("T_refined_from_keypoints.npy", T_ref)
