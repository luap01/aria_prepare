import json
from pathlib import Path

import cv2
import numpy as np

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from HaMuCo.utils.camera import load_cam_infos, project_3d_to_2d, project_to_2d
from HaMuCo.utils.image import undistort_image

# ------------------ Helpers ------------------

def se3_inv(T):
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def R_to_quat(R):
    w = np.sqrt(max(0.0, 1.0 + np.trace(R))) / 2.0
    x = (R[2, 1] - R[1, 2]) / (4 * w + 1e-12)
    y = (R[0, 2] - R[2, 0]) / (4 * w + 1e-12)
    z = (R[1, 0] - R[0, 1]) / (4 * w + 1e-12)
    return np.array([w, x, y, z])

def quat_to_R(q):
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w),   1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)]
    ])

def robust_avg_SE3(Ts, iters=6):
    """Huber-ish averaging over SE(3)."""
    qs = np.stack([R_to_quat(T[:3, :3]) for T in Ts])
    ts = np.stack([T[:3, 3] for T in Ts])
    q = qs.mean(0); q /= np.linalg.norm(q)
    t = ts.mean(0)
    for _ in range(iters):
        R = quat_to_R(q)
        ang_errs = []
        for Ti in Ts:
            dR = R.T @ Ti[:3, :3]
            ang = np.arccos(np.clip((np.trace(dR) - 1) * 0.5, -1.0, 1.0))
            ang_errs.append(ang)
        ang = np.array(ang_errs)
        s = np.median(ang) + 1e-6
        w = 1.0 / np.maximum(1.0, ang / s)
        q = (qs * w[:, None]).sum(0); q /= np.linalg.norm(q)
        t = (ts * w[:, None]).sum(0) / (w.sum() + 1e-12)
    T = np.eye(4)
    T[:3, :3] = quat_to_R(q)
    T[:3, 3] = t
    return T

def pnp_square(obj_pts, img_pts, K, dist=None):
    """SolvePnP for a planar square; uses IPPE_SQUARE then falls back."""
    if dist is None:
        dist = np.zeros(5)
    obj = obj_pts.astype(np.float32)
    img = img_pts.astype(np.float32)
    for flag in (cv2.SOLVEPNP_IPPE_SQUARE, cv2.SOLVEPNP_ITERATIVE):
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=flag)
        if ok:
            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec.reshape(3)
            return T
    raise RuntimeError("solvePnP failed")

def click_four(win_name, img, existing=None):
    """Click 4 points clockwise: TL, TR, BR, BL. Press 'u' to undo, 'Enter' to accept."""
    disp = img.copy()
    pts = [] if existing is None else [tuple(p) for p in existing]
    for p in pts:
        cv2.circle(disp, p, 5, (0, 255, 0), -1)
    def cb(event, x, y, flags, param):
        nonlocal disp, pts
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
            cv2.circle(disp, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(win_name, disp)
    cv2.imshow(win_name, disp); cv2.setMouseCallback(win_name, cb)
    print(f"[{win_name}] Click TL, TR, BR, BL (clockwise). 'u' undo, Enter accept.")
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == ord('u') and pts:
            pts.pop()
            disp = img.copy()
            for p in pts:
                cv2.circle(disp, p, 5, (0, 255, 0), -1)
            cv2.imshow(win_name, disp)
        if (k == 13 or k == 10) and len(pts) == 4:
            break
    cv2.destroyWindow(win_name)
    return np.array(pts, dtype=np.float64)

# ------------------ Config ------------------
# Feel free to add more Orbbec cams here (camera02, camera07, ...).
# Put actual image paths. Extrinsics: set as world->camera (invert your cam2world if needed).

dataset = "20250519_Testing"
frame_num = 2544
frame = f"{frame_num:06d}"
calib_dir = f"../HaMuCo/data/OR/calib/{dataset.split('_')[0]}"
cam_params_orbbec_1 = load_cam_infos(f"{calib_dir}/camera01.json")
cam_params_orbbec_2 = load_cam_infos(f"{calib_dir}/camera02.json")
cam_params_orbbec_3 = load_cam_infos(f"{calib_dir}/camera03.json")
cam_params_orbbec_4 = load_cam_infos(f"{calib_dir}/camera04.json")
cam_params_marshall_1 = load_cam_infos(f"{calib_dir}/camera05.json")
cam_params_marshall_2 = load_cam_infos(f"{calib_dir}/camera06.json")

all_cam_params = {
    "camera01": cam_params_orbbec_1,
    "camera02": cam_params_orbbec_2,
    "camera03": cam_params_orbbec_3,
    "camera04": cam_params_orbbec_4,
    "camera05": cam_params_marshall_1,
    "camera06": cam_params_marshall_2,
}
with open(f"../data/{dataset}/Aria/export/calib/calib_{frame}.json", "r") as f:
    aria_calib = json.load(f)

DATA = {
    # ---- ORBBEC CAMERAS ----
    "camera01": {
        "world": "WO",
        "img": f"../data/{dataset}/Orbbec/export/color/color_{frame}_camera01.jpg",
        "K": cam_params_orbbec_1["intrinsics"],
        "T_world_cam": cam_params_orbbec_1["extrinsics"],
    },
    "camera02": {
        "world": "WO",
        "img": f"../data/{dataset}/Orbbec/export/color/color_{frame}_camera02.jpg",
        "K": cam_params_orbbec_2["intrinsics"],
        "T_world_cam": cam_params_orbbec_2["extrinsics"],
    },
    "camera03": {
        "world": "WO",
        "img": f"../data/{dataset}/Orbbec/export/color/color_{frame}_camera03.jpg",
        "K": cam_params_orbbec_3["intrinsics"],
        "T_world_cam": cam_params_orbbec_3["extrinsics"],
    },
    "camera04": {
        "world": "WO",
        "img": f"../data/{dataset}/Orbbec/export/color/color_{frame}_camera04.jpg",
        "K": cam_params_orbbec_4["intrinsics"],
        "T_world_cam": cam_params_orbbec_4["extrinsics"],
    },
    "camera05": {
        "world": "WO",
        "img": f"../data/{dataset}/Marshall/export/color/color_{frame}_camera01.jpg",
        "K": cam_params_marshall_1["intrinsics"],
        "T_world_cam": cam_params_marshall_1["extrinsics"],
    },
    "camera06": {
        "world": "WO",
        "img": f"../data/{dataset}/Marshall/export/color/color_{frame}_camera02.jpg",
        "K": cam_params_marshall_2["intrinsics"],
        "T_world_cam": cam_params_marshall_2["extrinsics"],
    },
    # ---- ARIA ----
    "aria": {
        "world": "WA",
        "img": f"../data/{dataset}/Aria/export/color/color_{frame}_camera07.jpg",
        # Use a simple pinhole K; we rotate pixels to handle the 90°-skew used in your renderer
        "K": np.array([[300.0, 0, 511.5],
                       [0, 300.0, 511.5],
                       [0, 0, 1.0]]),
        "T_world_cam": aria_calib['extrinsics']["T_world_cam"],
        "rotate_pixels_minus_90deg": True,  # rotate around principal point for fisheye render
    },
}

# Box size (meters) — use real size if you have it; otherwise any consistent size works.
W, H = 0.20, 0.20
BOX_3D = np.array([[0,0,0], [W,0,0], [W,H,0], [0,H,0]], dtype=float)

# Where to cache/load clicks
CLICK_FILE = Path("box_clicks.json")

# ------------------ Main ------------------

def rotate_pixels_minus90(pix, cx=511.5, cy=511.5):
    uv = pix - np.array([cx, cy])
    R = np.array([[0, 1], [-1, 0]], float)  # -90°
    return (uv @ R.T) + np.array([cx, cy])

def main():
    # Load or collect clicks
    clicks = {}
    if CLICK_FILE.exists():
        try:
            clicks = json.loads(CLICK_FILE.read_text())
        except Exception:
            clicks = {}

    # Click on each image
    for name, cfg in DATA.items():
        if name == "camera03":
            continue
        img_path = Path(cfg["img"])
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.imread(str(img_path))
        img = undistort_image(img, all_cam_params[name], modality="color", orbbec=False if name in ["camera05", "camera06"] else True) if name != "aria" else img
        prev = clicks.get(name, None)
        prev_np = np.array(prev, dtype=float) if prev is not None else None
        pts = click_four(name, img, prev_np)
        clicks[name] = pts.tolist()

    # Save clicks
    CLICK_FILE.write_text(json.dumps(clicks, indent=2))
    print(f"Saved clicks to {CLICK_FILE.resolve()}")

    # Solve PnP for each camera
    T_world_box_by_space = {}  # {"WO":[T,...], "WA":[T,...]}
    for name, cfg in DATA.items():
        if name == "camera03":
            continue
        K = cfg["K"]
        T_wc = cfg["T_world_cam"]
        pts2d = np.array(clicks[name], dtype=float)

        # Aria pixel rotation trick (convert fisheye-rendered pixels to standard pinhole orientation)
        if cfg.get("rotate_pixels_minus_90deg", False):
            pts2d = rotate_pixels_minus90(pts2d, cx=K[0,2], cy=K[1,2])

        # PnP (object->camera)
        T_cam_box = pnp_square(BOX_3D, pts2d, K)

        # Lift to world (world->box) via world->camera
        T_world_box = T_wc @ T_cam_box

        T_world_box_by_space.setdefault(cfg["world"], []).append(T_world_box)

    # We expect exactly one WA (Aria world) set and >=1 WO (Orbbec world) sets
    if "WA" not in T_world_box_by_space or len(T_world_box_by_space["WA"]) != 1:
        raise RuntimeError("Need exactly one Aria view (WA).")
    if "WO" not in T_world_box_by_space or len(T_world_box_by_space["WO"]) < 1:
        raise RuntimeError("Need at least one Orbbec view (WO).")

    T_WA_box = T_world_box_by_space["WA"][0]
    candidates = [T_WO_box @ se3_inv(T_WA_box) for T_WO_box in T_world_box_by_space["WO"]]
    T_orbbec_aria = robust_avg_SE3(candidates)

    np.set_printoptions(precision=8, suppress=True)
    print(
        f"""
        \nT_orbbec_aria (Aria world → Orbbec world):\n 
        T_orbbec_aria = np.arrray([
        [{T_orbbec_aria[0, 0]:.12f}, {T_orbbec_aria[0, 1]:.12f}, {T_orbbec_aria[0, 2]:.12f}, {T_orbbec_aria[0, 3]:.12f}],
        [{T_orbbec_aria[1, 0]:.12f}, {T_orbbec_aria[1, 1]:.12f}, {T_orbbec_aria[1, 2]:.12f}, {T_orbbec_aria[1, 3]:.12f}],
        [{T_orbbec_aria[2, 0]:.12f}, {T_orbbec_aria[2, 1]:.12f}, {T_orbbec_aria[2, 2]:.12f}, {T_orbbec_aria[2, 3]:.12f}],
        [{T_orbbec_aria[3, 0]:.12f}, {T_orbbec_aria[3, 1]:.12f}, {T_orbbec_aria[3, 2]:.12f}, {T_orbbec_aria[3, 3]:.12f}]
        ], dtype=np.float32)
        """
    )

    # Save results
    np.save("T_orbbec_aria.npy", T_orbbec_aria)
    print("Saved to T_orbbec_aria.npy")

if __name__ == "__main__":
    main()
