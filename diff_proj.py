#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
import numpy as np
import cv2

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)


HAND_EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
              (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
              (0,17),(17,18),(18,19),(19,20)]

def load_json(path):
    s = Path(path).read_text()
    s = s.replace("NaN","null").replace("Infinity","null").replace("-Infinity","null")
    return json.loads(s)

def project_pinhole(K, Xc):
    x = Xc[:,0]/Xc[:,2]; y = Xc[:,1]/Xc[:,2]
    u = K[0,0]*x + K[0,2]; v = K[1,1]*y + K[1,2]
    return np.stack([u,v], 1)

def project_fisheye_equidistant(K, D, Xc):
    k1,k2,k3,k4 = D
    x = Xc[:,0]/Xc[:,2]; y = Xc[:,1]/Xc[:,2]
    r = np.sqrt(x*x + y*y) + 1e-12
    theta = np.arctan(r)
    theta_d = theta*(1 + k1*theta**2 + k2*theta**4 + k3*theta**6 + k4*theta**8)
    s = theta_d / r
    xd, yd = x*s, y*s
    u = K[0,0]*xd + K[0,2]; v = K[1,1]*yd + K[1,2]
    return np.stack([u,v], 1)

def load_side(d, key):
    if key not in d:  # return empty 21x3 + False mask if missing
        return np.empty((0,3), float), np.zeros((0,), bool)
    X = np.asarray(d[key]["X3d"], float)
    m = np.ones(len(X), bool)
    if "mask" in d[key]:
        m = np.asarray(d[key]["mask"], int) > 0
        if len(m) != len(X):  # pad/trim if needed
            n = min(len(m), len(X)); m = m[:n]; X = X[:n]
    return X, m

def load_points_3d(path):
    d = load_json(path)
    XL, ML = load_side(d, "left_hand")
    XR, MR = load_side(d, "right_hand")
    return (XL, ML), (XR, MR)

def draw_skeleton(img, uv, color_pts, color_lines):
    vis = np.isfinite(uv).all(1)
    for i in range(len(uv)):
        if vis[i]:
            cv2.circle(img, (int(uv[i,0]), int(uv[i,1])), 3, color_pts, -1)
    for a,b in HAND_EDGES:
        if a < len(uv) and b < len(uv) and vis[a] and vis[b]:
            cv2.line(img, (int(uv[a,0]), int(uv[a,1])),
                          (int(uv[b,0]), int(uv[b,1])), color_lines, 2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--p3d", required=True)
    ap.add_argument("--T_ref", default="T_refined_from_keypoints.npy")
    ap.add_argument("--model", choices=["pinhole","fisheye"], default="fisheye")
    ap.add_argument("--use_rotated_dest_K", action="store_true")
    ap.add_argument("--D_json", help="JSON with D=[k1,k2,k3,k4] for fisheye")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None: raise RuntimeError(f"Cannot read {args.image}")

    calib = load_json(args.calib)
    Twc = np.asarray(calib["extrinsics"]["T_world_cam"], float)
    Tcw = np.linalg.inv(Twc)

    if args.model == "fisheye":
        K = np.asarray(calib["intrinsics"]["K"], float)
        D = (0,0,0,0)
        if args.D_json:
            Dj = load_json(args.D_json)
            D = tuple(Dj["D"] if isinstance(Dj, dict) else Dj)
        proj_fn = lambda Xc: project_fisheye_equidistant(K, D, Xc)
    else:
        key = "destination_intrinsics_rotated" if args.use_rotated_dest_K else "destination_intrinsics"
        K = np.asarray(calib[key]["K"], float)
        proj_fn = lambda Xc: project_pinhole(K, Xc)

    T_ref = np.asarray(np.load(args.T_ref), float)

    (XL, ML), (XR, MR) = load_points_3d(args.p3d)
    sets = [("L", XL, ML, RED, GREEN), ("R", XR, MR, BLUE, YELLOW)]

    M = Tcw @ T_ref  # Orbbec world -> Aria camera
    for _, Xw, mask, c_pt, c_ln in sets:
        if len(Xw) == 0: continue
        Xw_h = np.c_[Xw, np.ones((len(Xw),1))]
        Xc = (Xw_h @ M.T)[:, :3]
        uv = np.full((len(Xw),2), np.nan, float)
        z_ok = Xc[:,2] > 1e-6
        keep = z_ok & mask
        if np.any(keep):
            uv[keep] = proj_fn(Xc[keep])
        draw_skeleton(img, uv, c_pt, c_ln)

    out = args.out or str(Path(args.image).with_suffix("").as_posix() + "_overlay.jpg")
    os.makedirs(Path(out).parent, exist_ok=True)
    cv2.imwrite(out, img)
    print("Saved:", out)

if __name__ == "__main__":
    main()

# python3 diff_proj.py --image ../data/20250519_Testing/Aria/export/raw/raw_002465_camera07.jpg --calib ../data/20250519_Testing/Aria/export/calib/calib_002465.json --p3d ./orbbec_joints3d/frame_002465_joints3d.json --T_ref T_refined_from_keypoints.npy --model pinhole   