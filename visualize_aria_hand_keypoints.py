#!/usr/bin/env python3
"""
Project 3D hand keypoints (in WORLD coordinates) onto rotated Aria images.

Expected inputs (for a given frame token, e.g. 000123 or -000005):
- Image:  ../data/20250519_Testing/Aria/export/color/color_<token>_camera07.jpg
- Hands:  ../data/20250519_Testing/Aria/export/hand/hand_<token>.npy
- Calib:  ../data/20250519_Testing/Aria/export/calib/calib_<token>.json

The calib JSON is assumed to contain:
- destination_intrinsics_rotated: { K: 3x3 }
- extrinsics: { R: 3x3, t: 3x1 }  # where R, t represent T_cam_world

Output:
- Annotated images saved in --out_dir (default: ./output)
"""

from pathlib import Path
import argparse
import json
import re
import sys
from typing import Dict, Tuple, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# -------------- Projection utilities --------------

def load_calib(calib_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load rotated intrinsics K and extrinsics (R, t) for T_cam_world."""
    with open(calib_path, "r") as f:
        data = json.load(f)

    # K for the already-rotated (90° CW) image coordinates
    K = np.array(data["destination_intrinsics_rotated"]["K"], dtype=float)

    # R, t are for T_cam_world (as saved in your script)
    R = np.array(data["extrinsics"]["R"], dtype=float).reshape(3, 3)
    t = np.array(data["extrinsics"]["t"], dtype=float).reshape(3)

    return K, R, t


def project_world_to_image(Xw: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray):
    """
    Project Nx3 world points into image using:
        Xc = R @ Xw + t
        u ~ K @ Xc
    Returns:
        uv (Nx2), valid mask (z>0), z (Nx,)
    """
    if Xw.ndim == 1:
        Xw = Xw[None, :]

    Xc = (R @ Xw.T).T + t  # (N,3)
    z = Xc[:, 2]
    valid = z > 1e-6

    uvw = (K @ Xc.T).T  # (N,3)
    uv = uvw[:, :2] / z[:, None]
    return uv, valid, z


# -------------- Drawing utilities --------------

COLORS = {
    "left_wrist": (255, 0, 0),       # red
    "left_palm": (255, 140, 0),      # dark orange
    "right_wrist": (0, 255, 255),    # cyan
    "right_palm": (30, 144, 255),    # dodger blue
    # normals (lines) will use slightly dimmer variants
    "left_wrist_normal": (200, 0, 0),
    "left_palm_normal": (200, 120, 0),
    "right_wrist_normal": (0, 200, 200),
    "right_palm_normal": (20, 120, 220),
}

LABELS_ORDER = [
    "left_wrist", "left_palm",
    "right_wrist", "right_palm",
]

NORMALS_PAIRS = [
    ("left_wrist", "left_wrist_normal"),
    ("left_palm", "left_palm_normal"),
    ("right_wrist", "right_wrist_normal"),
    ("right_palm", "right_palm_normal"),
]


def draw_point(draw: ImageDraw.ImageDraw, xy: Tuple[float, float], color, radius: int):
    x, y = xy
    r = radius
    draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=None)


def draw_label(draw: ImageDraw.ImageDraw, xy: Tuple[float, float], text: str, color, font):
    x, y = xy
    draw.text((x + 6, y - 10), text, fill=color, font=font)


def draw_normal_arrow(
    draw: ImageDraw.ImageDraw,
    start_xy: Tuple[float, float],
    end_xy: Tuple[float, float],
    color,
    width: int = 2,
    head_len: int = 8,
):
    # main line
    draw.line([start_xy, end_xy], fill=color, width=width)
    # simple arrow head
    sx, sy = start_xy
    ex, ey = end_xy
    vx, vy = ex - sx, ey - sy
    norm = (vx**2 + vy**2) ** 0.5
    if norm < 1e-3:
        return
    ux, uy = vx / norm, vy / norm
    # perpendicular vector
    px, py = -uy, ux
    # two head lines
    hx1 = ex - ux * head_len + px * (head_len * 0.6)
    hy1 = ey - uy * head_len + py * (head_len * 0.6)
    hx2 = ex - ux * head_len - px * (head_len * 0.6)
    hy2 = ey - uy * head_len - py * (head_len * 0.6)
    draw.line([ (hx1, hy1), (ex, ey) ], fill=color, width=width)
    draw.line([ (hx2, hy2), (ex, ey) ], fill=color, width=width)


# -------------- Main processing --------------

def process_frame(
    img_path: Path,
    hand_path: Path,
    calib_path: Path,
    out_path: Path,
    point_radius: int = 6,
    draw_normals: bool = True,
    normal_length_m: float = 0.07,  # scale for visual arrow length in meters
    line_width: int = 2,
):
    # Load inputs
    if not hand_path.exists():
        print(f"[skip] missing hand npy: {hand_path}")
        return
    if not calib_path.exists():
        print(f"[skip] missing calib json: {calib_path}")
        return

    hand_dict = np.load(hand_path, allow_pickle=True).item()
    K, R, t = load_calib(calib_path)

    # Gather points in a fixed order for consistent coloring/labeling
    points_world: Dict[str, np.ndarray] = {}
    for k in LABELS_ORDER:
        val = hand_dict.get(k, None)
        if val is not None:
            points_world[k] = np.asarray(val, dtype=float)

    # Project joints
    labels = list(points_world.keys())
    if len(labels) == 0:
        print(f"[skip] no valid joints in {hand_path.name}")
        return

    Xw = np.stack([points_world[k] for k in labels], axis=0)  # (N,3)
    uv, valid, z = project_world_to_image(Xw, K, R, t)

    # Prepare image canvas
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    font = ImageFont.load_default()

    # Draw joints
    for i, k in enumerate(labels):
        if not valid[i]:
            continue
        x, y = uv[i]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        if x < 0 or y < 0 or x >= W or y >= H:
            continue
        draw_point(draw, (x, y), COLORS.get(k, (255, 255, 255)), point_radius)
        draw_label(draw, (x, y), k, COLORS.get(k, (255, 255, 255)), font)

    # Optionally draw normals as short arrows
    if draw_normals:
        for joint_key, normal_key in NORMALS_PAIRS:
            pw = hand_dict.get(joint_key, None)
            nw = hand_dict.get(normal_key, None)
            if pw is None or nw is None:
                continue
            pw = np.asarray(pw, dtype=float)
            nw = np.asarray(nw, dtype=float)

            # Direction from joint toward the saved "normal" endpoint
            d = nw - pw
            n = np.linalg.norm(d)
            if n < 1e-9:
                continue
            direction = d / n
            head_world = pw + direction * normal_length_m

            # Project both endpoints
            uv_seg, valid_seg, _ = project_world_to_image(
                np.stack([pw, head_world], axis=0), K, R, t
            )
            if not (valid_seg[0] and valid_seg[1]):
                continue

            (x0, y0), (x1, y1) = uv_seg[0], uv_seg[1]
            if (0 <= x0 < W and 0 <= y0 < H and
                0 <= x1 < W and 0 <= y1 < H and
                np.isfinite([x0, y0, x1, y1]).all()):
                draw_normal_arrow(
                    draw, (x0, y0), (x1, y1),
                    COLORS.get(normal_key, (200, 200, 200)),
                    width=line_width
                )

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"[ok] {img_path.name} -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize 3D hand keypoints on Aria images.")
    parser.add_argument("--images_dir", type=Path,
                        default=Path("../data/20250519_Testing/Aria/export/color"),
                        help="Directory with color_<frame>_camera07.jpg")
    parser.add_argument("--hands_dir", type=Path,
                        default=Path("../data/20250519_Testing/Aria/export/hand"),
                        help="Directory with hand_<frame>.npy")
    parser.add_argument("--calib_dir", type=Path,
                        default=Path("../data/20250519_Testing/Aria/export/calib"),
                        help="Directory with calib_<frame>.json")
    parser.add_argument("--out_dir", type=Path, default=Path("./output"),
                        help="Where to save visualizations")
    parser.add_argument("--draw_normals", action="store_true", default=True,
                        help="Draw short arrows indicating normal directions")
    parser.add_argument("--no-draw_normals", dest="draw_normals", action="store_false",
                        help="Disable normal arrows")
    parser.add_argument("--point_radius", type=int, default=6)
    parser.add_argument("--normal_length_m", type=float, default=0.07,
                        help="Arrow length in meters (approx screen length varies with depth)")
    parser.add_argument("--line_width", type=int, default=2)
    parser.add_argument("--camera_tag", type=str, default="camera07",
                        help="Camera tag segment in image filename")
    args = parser.parse_args()

    # Find images: color_<token>_<camera_tag>.jpg
    pattern = re.compile(rf"^color_([-\d]+)_{re.escape(args.camera_tag)}\.jpg$")
    image_files = sorted(p for p in args.images_dir.glob(f"color_*_{args.camera_tag}.jpg") if pattern.match(p.name))

    if not image_files:
        print(f"No images found in: {args.images_dir} for tag {args.camera_tag}")
        sys.exit(1)

    for img_path in image_files:
        m = pattern.match(img_path.name)
        token = m.group(1)  # e.g., "000123" or "-000005"

        hand_path = args.hands_dir / f"hand_{token}.npy"
        calib_path = args.calib_dir / f"calib_{token}.json"
        out_path = args.out_dir / f"overlay_{token}_{args.camera_tag}.jpg"

        process_frame(
            img_path=img_path,
            hand_path=hand_path,
            calib_path=calib_path,
            out_path=out_path,
            point_radius=args.point_radius,
            draw_normals=args.draw_normals,
            normal_length_m=args.normal_length_m,
            line_width=args.line_width,
        )


if __name__ == "__main__":
    main()
