import argparse
import os
from pathlib import Path
import re
from typing import Dict, Tuple, Any, List

import cv2
import numpy as np
import sys

from camera import project_to_2d, load_cam_infos, rotation_to_homogenous  # expected to exist

# ------------------------
#  TRANSFORMS YOU PROVIDE
# ------------------------
# Replace this with your latest/approved Aria→Orbbec-world transform.
# This matrix maps Aria-frame points into the Orbbec world frame:
T_orbbec_world_aria = np.array([
    [-0.3977413223288592,  0.9174908459438001, -0.00351967632319674,  0.005649327336742365],
    [ 0.02202937620729426, 0.013384874720551236, 0.9996677206515334,  0.2644311962895414],
    [ 0.9172330931099286,  0.39753162482756593, -0.025535468782602863, 0.11725662452235236],
    [ 0, 0, 0, 1]
], dtype=np.float32)

# T_orbbec_world_aria = np.array([[-0.2454143464565277, 0.9693975448608398, -0.006340580526739359, 0.04169116169214249],
#  [0.007558995392173529, 0.008453970775008202, 0.9999356865882874, 0.2706661522388458],
#  [0.9693887829780579, 0.24535062909126282, -0.009402397088706493, 0.13133344054222107],
#  [0, 0, 0, 1]], dtype=np.float32)

# T_orbbec_world_aria = np.array([[-0.08733554929494858, 0.9961573481559753, -0.006563223898410797, 0.0624677836894989],
#  [-0.007182055152952671, 0.00595858646556735, 0.9999564290046692, 0.26872870326042175],
#  [0.9961530566215515, 0.08737887442111969, 0.006634060759097338, 0.1270017772912979],
#  [0, 0, 0, 1]], dtype=np.float32)

# T_orbbec_world_aria = np.array([[0.048111457377672195, 0.9988300800323486, -0.004882381297647953, 0.07925590127706528],
#  [-0.017865408211946487, 0.005747776944190264, 0.9998238682746887, 0.26740455627441406],
#  [0.9986822009086609, -0.0480157695710659, 0.018121039494872093, 0.1202501654624939],
#  [0, 0, 0, 1]], dtype=np.float32)

# T_orbbec_world_aria = np.array([[0.00132631731685251, 0.9999822974205017, -0.005807516630738974, 0.07351984083652496],
#  [-0.01037210039794445, 0.005820965860038996, 0.9999292492866516, 0.26833105087280273],
#  [0.999945342540741, -0.001265998580493033, 0.010379637591540813, 0.12181983143091202],
#  [0, 0, 0, 1]], dtype=np.float32)


# T_orbbec_world_aria = np.array([[0.00132631731685251, 0.9999822974205017, -0.005807516630738974, 0.07351984083652496],
#  [0.0030293958261609077, 0.005803477019071579, 0.9999785423278809, 0.26993951201438904],
#  [0.9999945163726807, -0.0013438933528959751, -0.0030216434970498085, 0.11821290850639343],
#  [0, 0, 0, 1]], dtype=np.float32)


# T_orbbec_world_aria = np.array([[0.00132631731685251, 0.9999822974205017, -0.005807516630738974, 0.04664194956421852],
#  [-0.01037210039794445, 0.005820965860038996, 0.9999292492866516, 0.26833105087280273],
#  [0.999945342540741, -0.001265998580493033, 0.010379637591540813, 0.11982982605695724],
#  [0, 0, 0, 1]], dtype=np.float32)

# T_orbbec_world_aria = np.array([[0.0013263284752473637, -0.010372101138519338, 0.9999453148635374, 0.2684467892955096],
#  [0.9999822158840965, 0.005820965675874906, -0.001265987284716896, 0.11930942485303062],
#  [-0.005807516413130564, 0.9999292815931526, 0.010379636833692647, 0.07701788144795531],
#  [0, 0, 0, 1]], dtype=np.float32)

# T_orbbec_world_aria = np.array([[ 0.00132632,  0.9999823,  -0.00580752,  0.04693232],
#  [-0.0103721,   0.00582097,  0.99992925,  0.21833459],
#  [ 0.99994534, -0.001266,    0.01037964,  0.01931085],
#  [ 0.,          0.,          0.,          1.        ]], dtype=np.float32)


T_orbbec_world_aria = np.array([[-0.3977413223288592, 0.9174908459438001, -0.00351967632319674, 0.020414896278126787],
 [0.02202937620729426, 0.013384874720551236, 0.9996677206515334, 0.2726872920512715],
 [0.9172330931099286, 0.39753162482756593, -0.025535468782602863, 0.1321804072024521],
 [0, 0, 0, 1]], dtype=np.float32)


X1 = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
    ], dtype=np.float32)


X2 = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
    ], dtype=np.float32)


# Your existing rotations
YZ_FLIP = rotation_to_homogenous(np.pi * np.array([1, 0, 0]))      # 180° about X (its own inverse)
YZ_SWAP = rotation_to_homogenous(np.pi/2 * np.array([1, 0, 0]))    # +90° about X

# T_orbbec_world_aria = T_orbbec_world_aria @ X1 @ X2

# T_orbbec_world_aria = YZ_SWAP_INV @ T_orbbec_world_aria @ YZ_FLIP_INV

# ------------------------
#     SMALL UTILITIES
# ------------------------

def deduce_cam_name(cam: int) -> str:
    """1..4 -> Orbbec, 5..6 -> Marshall."""
    if cam < 5:
        return "Orbbec"
    return "Marshall"

def camera_offset(cam: int) -> int:
    """
    Marshall image files often use _camera01/_camera02.
    For cams 5..6 map (5->1, 6->2) by applying offset -4.
    """
    return -4 if cam in (5, 6) else 0

def frame_index_from_name(filename: str) -> int:
    """Extract numeric frame index from e.g., 'raw_000123_camera06.jpg'."""
    m = re.search(r"_(\d+)_camera0\d+\.jpg$", filename)
    if m:
        return int(m.group(1))
    m2 = re.search(r"(\d+)", filename)
    return int(m2.group(1)) if m2 else 0

def list_images(input_dir: Path, img_type: str, cam: int) -> List[Path]:
    """
    Collect and sort images matching: <img_type>_******_camera0X.jpg for this cam.
    Uses offset so Marshall cams (5,6) search for IDs 1,2 respectively.
    """
    off = camera_offset(cam)
    cam_id = cam + off
    pattern = f"{img_type}_*_camera0{cam_id}.jpg"
    return sorted(input_dir.glob(pattern), key=lambda p: frame_index_from_name(p.name))

def input_dir_for_cam(root: Path, img_type: str, cam: int) -> Path:
    """
    Resolve the folder that holds images for a given cam.
    Default convention: <root>/<Orbbec|Marshall>/export/<img_type>/
    """
    cam_name = deduce_cam_name(cam)
    path = root / cam_name / "export" / img_type
    return path

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def write_video_from_frames(frames_dir: Path, video_path: Path, fps: int):
    frames = sorted(frames_dir.glob("*.jpg"), key=lambda p: frame_index_from_name(p.name))
    if not frames:
        print(f"[WARN] No frames found in {frames_dir}, skipping video assembly for {video_path.name}.")
        return

    first = cv2.imread(str(frames[0]))
    if first is None:
        print(f"[ERROR] Couldn't read first frame: {frames[0]}")
        return
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

    for f in frames:
        img = cv2.imread(str(f))
        if img is None:
            print(f"[WARN] Skipping unreadable frame: {f}")
            continue
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        vw.write(img)

    vw.release()
    print(f"[OK] Wrote video: {video_path}")

# ------------------------
#   CALIB HELPERS
# ------------------------

def get_cam_entry(cam_infos: Dict[str, Any], cam: int) -> Dict[str, Any]:
    """
    Robustly fetch the calibration entry for cam index 1..6.
    Accepts several common key styles.
    """
    candidates = [
        f"camera0{cam}", f"camera{cam}", str(cam), f"{cam:02d}",
        f"cam{cam}",  # generic
    ]
    # Also try labeled styles
    if cam <= 4:
        candidates += [f"Orbbec{cam}"]
    else:
        candidates += [f"Marshall{cam-4}"]

    for k in candidates:
        if k in cam_infos:
            return cam_infos[k]

    # As a last resort, accept exact int keys if present
    if cam in cam_infos:
        return cam_infos[cam]

    raise KeyError(
        f"Could not find entry for camera {cam}. "
        f"Available keys: {list(cam_infos.keys())}"
    )

def get_intrinsics_extrinsics(entry: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (K, T_cam_world) as float32.
    Accepts:
      - entry['K'] (or entry['intrinsics']['K'])
      - entry['T_cam_world'] OR entry['T_world_cam'] (we invert the latter)
    """
    # Intrinsics
    if "K" in entry:
        K = np.array(entry["K"], dtype=np.float32)
    elif "intrinsics" in entry and "K" in entry["intrinsics"]:
        K = np.array(entry["intrinsics"]["K"], dtype=np.float32)
    else:
        raise KeyError("Intrinsics K not found in cam entry. Expected 'K' or 'intrinsics.K'.")

    # Extrinsics
    if "T_cam_world" in entry:
        T_cam_world = np.array(entry["T_cam_world"], dtype=np.float32)
    elif "T_world_cam" in entry:
        T_cam_world = np.linalg.inv(np.array(entry["T_world_cam"], dtype=np.float32))
    elif "extrinsics" in entry and "T_world_cam" in entry["extrinsics"]:
        T_cam_world = np.linalg.inv(np.array(entry["extrinsics"]["T_world_cam"], dtype=np.float32))
    else:
        raise KeyError("Extrinsics not found. Expected 'T_cam_world' or 'T_world_cam' (or in 'extrinsics').")

    return K, T_cam_world

# ------------------------
#    DRAWING (ARIA→CAM)
# ------------------------

def draw_aria_axes_on_image(
    img: np.ndarray,
    K: np.ndarray,
    T_cam_world: np.ndarray,
    T_orbbec_world_aria: np.ndarray,
    axis_length: float = 0.2,
) -> np.ndarray:
    """
    Projects Aria-frame origin and axes into the given camera image.

    Steps:
      Aria point --(T_orbbec_world_aria)--> Orbbec world point --(T_cam_world)--> camera coords --(K)--> pixels
    """
    # Define Aria-frame points
    aria_pts = [
        np.array([0.0, 0.0, 0.0], dtype=np.float32),                 # origin
        np.array([axis_length, 0.0, 0.0], dtype=np.float32),         # x
        np.array([0.0, axis_length, 0.0], dtype=np.float32),         # y
        np.array([0.0, 0.0, axis_length], dtype=np.float32),         # z
    ]

    def aria_to_orbbec_world(p_aria: np.ndarray) -> np.ndarray:
        p_h = np.concatenate([p_aria, np.array([1.0], dtype=np.float32)])
        p_w = T_orbbec_world_aria @ p_h
        return p_w[:3]

    # Transform to Orbbec world
    world_pts = [(p) for p in aria_pts]

    # Project into pixels using the camera's extrinsics
    origin_px = tuple(map(int, project_to_2d(world_pts[0], K, T_cam_world)))
    x_px     = tuple(map(int, project_to_2d(world_pts[1], K, T_cam_world)))
    y_px     = tuple(map(int, project_to_2d(world_pts[2], K, T_cam_world)))
    z_px     = tuple(map(int, project_to_2d(world_pts[3], K, T_cam_world)))

    # Debug (optional)
    # print(f"px: origin={origin_px}, x={x_px}, y={y_px}, z={z_px}")

    # x=red, y=blue, z=green (kept consistent with your earlier convention)
    colors = {
        "red":   (255, 0, 0),
        "blue":  (0, 0, 255),
        "green": (0, 255, 0),
    }
    thickness = 10
    cv2.arrowedLine(img, origin_px, x_px, colors["red"],   thickness, tipLength=0.1)
    cv2.arrowedLine(img, origin_px, y_px, colors["blue"],  thickness, tipLength=0.1)
    cv2.arrowedLine(img, origin_px, z_px, colors["green"], thickness, tipLength=0.1)

    return img

# ------------------------
#           CLI
# ------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Project Aria axes (origin in Aria frame) into each of the six camera views and write per-camera videos."
    )
    p.add_argument("--root", required=True, type=str,
                   help="Root folder containing Orbbec/ and Marshall/ subfolders (each with export/<img_type>/ images).")
    p.add_argument("--img_type", default="color", type=str,
                   help="Image type subfolder name under each camera (e.g., 'raw' or 'color').")
    p.add_argument("--out_dir", default=None, type=str,
                   help="Output directory for overlays & videos. Default: <root>/overlays_aria_origin/")
    p.add_argument("--fps", default=30, type=int, help="FPS for output videos.")
    p.add_argument("--axis_length", default=0.2, type=float, help="Axis length in meters (Aria frame).")
    p.add_argument("--max_frames", default=0, type=int,
                   help="Optional limit on number of frames per camera (0 = no limit).")
    return p.parse_args()

# ------------------------
#           MAIN
# ------------------------

def main():
    args = parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise NotADirectoryError(f"Root not found: {root}")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (root / "overlays_aria_origin")
    ensure_dir(out_dir)

    print(f"Root:   {root}")
    print(f"Out:    {out_dir}")
    print(f"img_type='{args.img_type}', fps={args.fps}, axis_length={args.axis_length}")
    print("-" * 60)

    # Load all camera intrinsics/extrinsics once
    dataset = str(root).split("/")[-1].replace("_Testing", "")    
    for cam in range(1, 7):
        cam_infos = load_cam_infos(f"calib/{dataset}/camera0{cam}.json")  # expected to return dict for cameras 1..6
        cam_name = deduce_cam_name(cam)
        in_dir = input_dir_for_cam(root, args.img_type, cam)

        if not in_dir.exists():
            print(f"[WARN] Missing input dir for cam{cam} ({cam_name}): {in_dir} — skipping.")
            continue

        print(f"[CAM {cam}] {cam_name}  |  images: {in_dir}")

        try:
            # entry = get_cam_entry(cam_infos, cam)
            K, T_cam_world = get_intrinsics_extrinsics(cam_infos)
        except Exception as e:
            print(f"[ERROR] Calibration for cam{cam} not found/invalid: {e}")
            continue

        images = list_images(in_dir, args.img_type, cam)
        if not images:
            print(f"[WARN] No images found for cam{cam} in {in_dir}. Pattern: {args.img_type}_*_camera0*.jpg")
            continue

        # Per-camera output
        saved_dir = out_dir / f"camera0{cam}"
        ensure_dir(saved_dir)

        limit = args.max_frames if args.max_frames and args.max_frames > 0 else len(images)
        for i, img_path in enumerate(images[:limit], 1):
            if i % 100 == 1 or i == limit:
                print(f"  [{i}/{limit}] {img_path.name}")

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] Failed to read image, skipping: {img_path}")
                continue

            # Draw Aria axes projected into this camera view
            img = draw_aria_axes_on_image(
                img, K, T_cam_world, T_orbbec_world_aria, axis_length=args.axis_length
            )

            # Write overlay frame (keep name for sorting)
            out_path = saved_dir / img_path.name
            cv2.imwrite(str(out_path), img)

        # Assemble per-camera video
        video_path = out_dir / f"overlay_camera0{cam}.mp4"
        # write_video_from_frames(saved_dir, video_path, args.fps)

    print("[DONE] All requested camera videos assembled.")

if __name__ == "__main__":
    main()
