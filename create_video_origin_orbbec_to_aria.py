import argparse
import json
import os
from pathlib import Path
import re
from typing import List, Tuple

import cv2
import numpy as np
import sys

# Ensure relative imports work like your original
from camera import project_to_2d, rotation_to_homogenous  # using your existing util

# ---------- Constants (kept from your script) ----------

T_orbbec_aria_feb = np.array([
    [-8.74827802e-01, -4.84434068e-01,  1.08598066e-07, -2.67419100e-01],
    [ 7.64798784e-08,  8.60619664e-08,  1.00000000e+00,  4.54925746e-01],
    [-4.84434068e-01,  8.74827802e-01, -3.82399392e-08, -1.44934177e-01],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]],
    dtype=np.float32,
)

T_orbbec_aria_feb_rot = np.array([
    [-4.84434068e-01,  8.74827802e-01, -3.82399392e-08, -1.44934192e-01],
    [ 7.64798784e-08,  8.60619664e-08,  1.00000000e+00,  4.54925746e-01],
    [ 8.74827802e-01,  4.84434068e-01, -1.08598066e-07,  2.67419130e-01],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
    ], dtype=np.float32,
)

T_orbbec_aria_may_blender = np.array([
    [-0.499695413510, 0.866106831943, 0.012784732288, 0.029228000000],
    [0.865497844508, 0.498640616331, 0.047655187522, 0.288147000000],
    [0.034899496703, 0.034878236872, -0.998782025130, -0.430261000000],
    [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]
    ], dtype=np.float32)
    

T_icp = np.array([
    [-0.49969545006752014, 0.8654978275299072, 0.03489949554204941, -0.21871385504823773],
    [0.8661068677902222, 0.49864062666893005, 0.034878239035606384, -0.15232917858016465],
    [0.012784730643033981, 0.04765518754720688, -0.9987820386886597, -0.43349152738645297],
    [0, 0, 0, 1]
    ], dtype=np.float32
)

T_icp = np.array([
    [-0.49969545006752014, 0.8654978275299072, 0.03489949554204941, -0.21091119176894652],
    [0.8661068677902222, 0.49864062666893005, 0.034878239035606384, -0.1528711937178671],
    [0.012784730643033981, 0.04765518754720688, -0.9987820386886597, -0.4221053760582762],
    [0, 0, 0, 1]], dtype=np.float32
)

T_pnp = np.array([
    [0.7332905801432457, -0.027173162453435817, 0.6793721692235128, 0.7181260845778901],
    [0.38560292076419667, -0.806347114667832, -0.44845815653746596, -0.27632275199514694],
    [0.5599958147801667, 0.5908180345158235, -0.5808086927031619, -0.09947017799619422],
    [0, 0, 0, 1]
 ], dtype=np.float32
)

T_icp = np.array([
    [
      -0.46918556,
      0.8819397,
      0.0452469,
      0.069228
    ],
    [
      -0.0348995,
      -0.06971398,
      0.99695635,
      0.5202601
    ],
    [
      0.88240975,
      0.46617845,
      0.06348804,
      0.30815
    ],
    [
      0.0,
      0.0,
      0.0,
      1.0
    ]
], dtype=np.float32)

T_pnp_orbbec_aria = np.array([
    [-0.455273032932499, 0.8857139689466593, 0.0907591907037516, 0.06902000411429118],
    [-0.373933868890656, -0.2827219437537165, 0.8833129480634743, 0.4106072142562965],
    [0.808022231860553, 0.36821062957616624, 0.45991412794804853, 0.411978253187228],
    [0, 0, 0, 1]
], dtype=np.float32)


T_pnp_aria_orbbec = np.array([
    [-0.45527303402857233, -0.37393386711232257, 0.8080222320659516, -0.14792469679095754],
    [0.8857139679600721, -0.2827219476841727, 0.36821062893144885, -0.09673908402470302],
    [0.09075919483360093, 0.8833129475582777, 0.45991412810335, -0.5584334876848073],
    [0, 0, 0, 1]
], dtype=np.float32)

T_pnp_aria_orbbec = np.array([[-0.455273032932499, -0.373933868890656, 0.808022231860553, -0.1479246967909576],
 [0.8857139689466593, -0.2827219437537165, 0.36821062957616624, -0.0967390840247032],
 [0.0907591907037516, 0.8833129480634743, 0.45991412794804853, -0.5584334876848069],
 [0, 0, 0, 1]], dtype=np.float32)

T_pnp_aria_orbbec = np.array(
    [[-0.6143831098301207, 0.365893801261343, 0.699038711770668, -0.174815667561526],
 [0.5088572616640257, -0.49334412635461944, 0.7054614519895377, 0.0002060307885308199],
 [0.602990614858357, 0.7891345254073747, 0.11691458079648664, -0.6271173674131791],
 [0, 0, 0, 1]], dtype=np.float32
)

T_pnp_aria_orbbec = np.array([[-0.49410263592253445, -0.08350628128815846, 0.8653838952510189, -0.1979546224161305],
 [0.8666503039615192, -0.1264622259971846, 0.4826225813606772, -0.13594245846240965],
 [0.06913635670043644, 0.9884503054687472, 0.1348560632709187, -0.5135281725678464],
 [0, 0, 0, 1]], dtype=np.float32)

T_pnp_aria_orbbec = np.array([[-0.47495032973538764, -0.047250022473182496, 0.8787432046169863, -0.19507716808305736],
 [0.8794410282102352, -0.061468745598899144, 0.4720223206745867, -0.16177255351592706],
 [0.03171217723164482, 0.9969899842479771, 0.07074820933739001, -0.5167651948063208],
 [0, 0, 0, 1]], dtype=np.float32)

T_pnp_aria_orbbec = np.array([[-0.47517845996967645, -0.06030715879474213, 0.8778202992520465, -0.21475135948771817],
 [0.8797554779452527, -0.04997184919331632, 0.4727928862763489, -0.14990210745356436],
 [0.015353507943362654, 0.9969282125039427, 0.07680109965050552, -0.5435707565497186],
 [0, 0, 0, 1]], dtype=np.float32)

T_pnp_aria_orbbec_motion = np.array(
    [[0.9890129210119528, 0.09170987881359502, -0.11594283159989044, 0.10910595346725643],
    [-0.07471258017002719, 0.9868573632578109, 0.14328494320122517, -0.049248612937645375],
    [0.12755968185810945, -0.1330482721114287, 0.9828665651310056, -0.32347887398561526],
    [0, 0, 0, 1]], dtype=np.float32
)

T_pnp_aria_orbbec = np.array(
    [[-0.4579225341945394, -0.02925879756206719, 0.8885104813349547, -0.22067738616677993],
 [0.8875503181493414, -0.071945798557574, 0.45505849604519744, -0.16732908859358395],
 [0.050610131691732264, 0.9969793001036357, 0.05891425748501408, -0.5393107150732688],
 [0, 0, 0, 1]]
)

T_aria_world_orbbec = np.array([[-0.45168289232295594, 0.0008420960419838819, 0.8921781524208192, -0.13293935255942463],
 [0.8921325937876252, 0.0105760394197034, 0.4516498450038007, -0.15670980033821225],
 [-0.009055378762560488, 0.9999437175483673, -0.005528277400566351, -0.3833568448904635],
 [0, 0, 0, 1]], dtype=np.float32)

T_aria_world_orbbec = np.array([[-0.41444880977524595, -0.010311649677426628, 0.910014205359901, -0.16022347556076968],
 [0.9096387374216253, 0.02617982395486803, 0.4145744615864363, -0.1528526485014864],
 [-0.028098958305838355, 0.999604064967036, -0.0014703208849121529, -0.39903872892991593],
 [0, 0, 0, 1]], dtype=np.float32)

T_aria_world_orbbec = np.array([[-0.4128559860482618, -0.02132361835377084, 0.9105466699101263, -0.12238773489020839],
 [0.9090479093948297, 0.0522681028100119, 0.41340046426380744, -0.08080384492828407],
 [-0.056407740683410076, 0.9984054029946676, -0.0021950084392710527, -0.2698165414718096],
 [0, 0, 0, 1]])


T_orbbec_world_aria = np.array(
    [[-0.4681186142558247, 0.8835560378220539, -0.013917291956442377, 0.06874350185616492],
     [-0.01220315975679898, 0.009284221034147984,  0.9998824361552404, 0.38641361546505054],
     [ 0.8835813747919075, 0.46823341536885565,   0.0064360587986766835, 0.19066101889192952],
     [ 0., 0., 0., 1. ]], dtype=np.float32
)

T_orbbec_world_aria = np.array([[-0.3977413223288592, 0.9174908459438001, -0.00351967632319674, 0.020414896278126787],
 [0.02202937620729426, 0.013384874720551236, 0.9996677206515334, 0.2726872920512715],
 [0.9172330931099286, 0.39753162482756593, -0.025535468782602863, 0.1321804072024521],
 [0, 0, 0, 1]], dtype=np.float32)

# T_orbbec_world_aria = np.array([[-0.3977413223288592, 0.9174908459438001, -0.00351967632319674, 0.005639011140329665],
#  [0.02202937620729426, 0.013384874720551236, 0.9996677206515334, 0.26443239724066686],
#  [0.9172330931099286, 0.39753162482756593, -0.025535468782602863, 0.11725443044765561],
#  [0, 0, 0, 1]], dtype=np.float32)

T_orbbec_world_aria = np.array([[-0.39774131774902344, 0.9174908399581909, -0.0035196763928979635, 0.020414896309375763],
 [0.022029375657439232, 0.013384874910116196, 0.9996677041053772, 0.27268728613853455],
 [0.9172331094741821, 0.3975316286087036, -0.025535468012094498, 0.12468025833368301],
 [0, 0, 0, 1]], dtype=np.float32)


T_orbbec_world_aria = np.array([[-0.39774131774902344, 0.9174908399581909, -0.0035196763928979635, -0.003527136752381921],
 [0.022029375657439232, 0.013384874910116196, 0.9996677041053772, 0.27268728613853455],
 [0.9172331094741821, 0.3975316286087036, -0.025535468012094498, 0.12015814334154129],
 [0, 0, 0, 1]], dtype=np.float32)

T_orbbec_world_aria = np.array(
 [[-0.39774132,  0.91749084, -0.00351968, 0.025335115],
 [ 0.02202938,  0.01338487,  0.9996677,   0.3527039 ],
 [ 0.9172331,   0.39753163, -0.02553547,  0.20143492],
 [ 0.,          0.,          0.,          1.        ]], dtype=np.float32)


T_orbbec_world_aria = np.array(
 [[-0.39774132,  0.91749084, -0.00351968, 0.07087158],
 [ 0.02202938,  0.01338487,  0.9996677,   0.39764744 ],
 [ 0.9172331,   0.39753163, -0.02553547,  0.20956215],
 [ 0.,          0.,          0.,          1.        ]], dtype=np.float32)


T_orbbec_world_aria = np.array(
 [[-0.39774132,  0.91749084, -0.00351968, -0.08232071],
 [ 0.02202938,  0.01338487,  0.9996677,   0.4521537  ],
 [ 0.9172331,   0.39753163, -0.02553547,  0.25785166 ],
 [ 0.,          0.,          0.,          1.        ]], dtype=np.float32)


# T_orbbec_world_aria = np.array(
# [[-0.4362234380106891, 0.8986148873006364, -0.04690838359823188, -0.05391588843357089],
#  [0.02701271884376204, 0.06518370740127519, 0.9975075926077421, 0.3300016405988716],
#  [0.8994328352638692, 0.43386906851156076, -0.05270869224433826, 0.15616514532791967],
#  [0, 0, 0, 1]], dtype=np.float32)


T_orbbec_world_aria = np.array(
 [[-0.39774132,  0.91749084, -0.00351968, -0.05391588843357089],
 [ 0.02202938,  0.01338487,  0.9996677,   0.3300016405988716  ],
 [ 0.9172331,   0.39753163, -0.02553547,  0.15616514532791967 ],
 [ 0.,          0.,          0.,          1.        ]], dtype=np.float32)

T_orbbec_world_aria = np.array(
 [[-0.39774132,  0.91749084, -0.00351968, -0.08265147 ],
 [ 0.02202938,  0.01338487,  0.9996677,   0.45190677  ],
 [ 0.9172331,   0.39753163, -0.02553547,  0.2571405   ],
 [ 0.,          0.,          0.,          1.        ]], dtype=np.float32)

    
# T_orbbec_world_aria = np.array(
#  [[-0.39774132,  0.91749084, -0.00351968, -0.00222763],
#  [ 0.02202938,  0.01338487,  0.9996677,   0.3800042 ],
#  [ 0.9172331,   0.39753163, -0.02553547,  0.16168538],
#  [ 0.,          0.,          0.,          1.        ]], dtype=np.float32)


# T_orbbec_world_aria = np.array([[ 0.00132632,  0.9999823,  -0.00580752,  0.04693232],
#  [-0.0103721,   0.00582097,  0.99992925,  0.21833459],
#  [ 0.99994534, -0.001266,    0.01037964,  0.01931085],
#  [ 0.,          0.,          0.,          1.        ]], dtype=np.float32)


YZ_FLIP = rotation_to_homogenous(np.pi * np.array([1, 0, 0]))      # 180° about X (its own inverse)
YZ_SWAP = rotation_to_homogenous(np.pi/2 * np.array([1, 0, 0]))    # +90° about X

# ---------- Helpers ----------

def draw_axes(img, intrinsics, extrinsics, axis_length=0.2):
    """
    Draws the 3D coordinate axes (x=red, y=green, z=blue) projected into the image.
    """
    origin = project_to_2d(np.zeros(3), intrinsics, extrinsics)
    x_axis = project_to_2d(np.array([axis_length, 0, 0]), intrinsics, extrinsics)
    y_axis = project_to_2d(np.array([0, axis_length, 0]), intrinsics, extrinsics)
    z_axis = project_to_2d(np.array([0, 0, axis_length]), intrinsics, extrinsics)

    origin = tuple(map(int, origin))
    x_axis = tuple(map(int, x_axis))
    y_axis = tuple(map(int, y_axis))
    z_axis = tuple(map(int, z_axis))

    # Debug print of projected axes
    print(f"Axes (px): origin={origin}, x={x_axis}, y={y_axis}, z={z_axis}")

    colors = {
        "red": (255, 0, 0),    # x+
        "green": (0, 255, 0),  # z+
        "blue": (0, 0, 255),   # y+
    }
    cv2.arrowedLine(img, origin, x_axis, colors["red"], 10, tipLength=0.1)
    cv2.arrowedLine(img, origin, y_axis, colors["blue"], 10, tipLength=0.1)
    cv2.arrowedLine(img, origin, z_axis, colors["green"], 10, tipLength=0.1)

    return img


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterate images, overlay 3D axes using per-frame calibration, and assemble a video."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        type=str,
        help="Directory containing input images, e.g. ../data/<dataset>_Testing/<CamName>/export/raw/"
    )
    parser.add_argument(
        "--cam",
        default=1,
        type=int,
        help="Which camera index to project (to match the file suffix _camera0X)."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for overlays. Default: <input_dir>/../overlays/"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS for the output video."
    )
    parser.add_argument(
        "--img_type",
        type=str,
        default="raw",
        help="Image type subfolder name (usually 'raw')."
    )
    parser.add_argument(
        "--video_name",
        type=str,
        default="overlay.mp4",
        help="Filename for the output video (MP4)."
    )
    return parser.parse_args()


def deduce_cam_name(cam: int) -> str:
    """
    Mirrors your original mapping: 1..4 -> Orbbec, 5..6 -> Marshall, else Aria.
    """
    if cam < 5:
        return "Orbbec"
    elif cam < 7:
        return "Marshall"
    return "Aria"


def camera_offset(cam: int) -> int:
    """
    Same logic as your original: offset -4 for cams 5 or 6; else 0.
    """
    return -4 if (cam > 4 and cam < 7) else 0


def list_images(input_dir: Path, img_type: str, cam: int) -> List[Path]:
    """
    Collect and sort images matching the pattern: <img_type>_######_camera0X.jpg
    Only include files for the selected camera (with offset applied).
    """
    off = camera_offset(cam)
    cam_id = cam + off
    pattern = f"{img_type}_*_camera0{cam_id}.jpg"
    files = sorted(input_dir.glob(pattern), key=lambda p: frame_index_from_name(p.name))
    return files


def frame_index_from_name(filename: str) -> int:
    """
    Extracts the numeric frame index from names like 'raw_000123_camera06.jpg'.
    Falls back to lexicographic if not found.
    """
    m = re.search(r"_(\d+)_camera0\d+\.jpg$", filename)
    if m:
        return int(m.group(1))
    # fallback: try any number sequence
    m2 = re.search(r"(\d+)", filename)
    return int(m2.group(1)) if m2 else 0


def calib_path_for_image(img_path: Path, img_type: str, cam: int) -> Path:
    """
    Mirrors your original replacement chain:
      - replace '/<img_type>/' with '/calib/'
      - drop the '_camera0X' suffix
      - change .jpg -> .json
    """
    off = camera_offset(cam)
    cam_id = cam + off

    # switch directory from .../<img_type>/ to .../calib/
    calib_dir = Path(str(img_path.parent).replace(f"/{img_type}", "/calib"))

    # drop '_camera0X' and change extension to .json
    stem = img_path.stem.replace(img_type, "calib")  # e.g. 'raw_000123_camera06'
    stem_wo_camera = re.sub(rf"_camera0{cam_id}$", "", stem)
    calib_file = calib_dir / f"{stem_wo_camera}.json"
    return calib_file


def build_M_and_intrinsics(calib_json: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build M = T_cam_world_aria @ T_aria_world_orbbec and return (M, K_aria).
    Uses destination_intrinsics_rotated['K'] like your current code.
    """
    K_aria = np.array(calib_json["destination_intrinsics_rotated"]["K"], dtype=np.float32)
    T_world_cam_aria = np.array(calib_json["extrinsics"]["T_world_cam"], dtype=np.float32)  # 4x4
    # YZ_FLIP = rotation_to_homogenous(np.pi * np.array([1, 0, 0]))
    # YZ_SWAP = rotation_to_homogenous(np.pi/2 * np.array([1, 0, 0]))

    Fh = np.eye(4, dtype=np.float32)
    Fh[1,1] = -1.0; Fh[2,2] = -1.0     # 180° about +X (YZ_FLIP)
    # T_world_cam_aria = YZ_SWAP @ T_world_cam_aria @ YZ_FLIP
    T_cam_world_aria = np.linalg.inv(T_world_cam_aria)
    # T_cam_world_aria = np.linalg.inv(YZ_SWAP) @ T_cam_world_aria @ np.linalg.inv(YZ_FLIP)

    # Start from your ICP and apply your two X matrices (axis reorders + flip Y)
    T_orbbec_aria = T_orbbec_world_aria.copy()
    T_aria_world_orbbec = np.linalg.inv(T_orbbec_aria)
    # T_aria_world_orbbec = T_aria_world_orbbec.copy()

    # X = np.array([
    #     [1, 0, 0, 0],
    #     [0, 0, 1, 0],
    #     [0, 1, 0, 0],
    #     [0, 0, 0, 1]
    #     ], dtype=np.float32)

    # T_aria_world_orbbec = T_aria_world_orbbec @ X  # Apply the rotation to the transformation matrix

    # X = np.array([
    #     [1, 0, 0, 0],
    #     [0, -1, 0, 0],
    #     [0, 0, 1, 0],
    #     [0, 0, 0, 1]
    #     ], dtype=np.float32)
    
    # T_aria_world_orbbec = T_aria_world_orbbec @ X  # Flip Y axis

    # For logging, you had:
    # print("Transformation matrix from Orbbec-world to Aria-world (inverse shown):")
    # print(np.linalg.inv(T_aria_world_orbbec))

    # Final M for projection: world(Orbbec) → camera(Aria)
    M = T_cam_world_aria @ T_aria_world_orbbec
    # print("Transformation matrix from Orbbec-world to Aria-camera (M):")
    # print(M)

    return M, K_aria


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_video_from_frames(frames_dir: Path, video_path: Path, fps: int):
    """
    Create an MP4 video from all jpgs in frames_dir, sorted by numeric index in filename.
    """
    frames = sorted(frames_dir.glob("*.jpg"), key=lambda p: frame_index_from_name(p.name))
    if not frames:
        print(f"[WARN] No frames found in {frames_dir}, skipping video assembly.")
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


def main():
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    cam = int(args.cam)
    cam_name = deduce_cam_name(cam)
    off = camera_offset(cam)

    # Default output directory next to input_dir
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (input_dir.parent / "overlays")
    ensure_dir(out_dir)

    print(f"Input:  {input_dir}")
    print(f"Output: {out_dir}")
    print(f"Camera: {cam} ({cam_name}), offset applied: {off}")
    print(f"Image type folder: '{args.img_type}'")
    print("-" * 60)

    images = list_images(input_dir, args.img_type, cam)
    if not images:
        print(f"[WARN] No images found with pattern '{args.img_type}_*_camera0{cam+off}.jpg' in {input_dir}")
        return

    saved_dir = out_dir / f"camera0{cam}"
    ensure_dir(saved_dir)

    # Process each image
    for i, img_path in enumerate(images, 1):
        if i > 1800:
            break
        print(f"[{i}/{len(images)}] {img_path.name}")

        # Derive calibration path
        calib_path = calib_path_for_image(img_path, args.img_type, cam)
        if not calib_path.exists():
            print(f"[WARN] Missing calibration JSON for frame: {calib_path}")
            continue

        # Load calibration
        with open(calib_path, "r") as f:
            calib = json.load(f)

        # Build projection matrix and intrinsics
        M, K_aria = build_M_and_intrinsics(calib)
        # print(f"Aria intrinsics K:\n{K_aria}")

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[WARN] Failed to read image, skipping: {img_path}")
            continue

        # Draw axes
        image = draw_axes(image, K_aria, M)

        # Save overlay (keep input filename for consistent sorting)
        out_path = saved_dir / img_path.name.replace(f"/{args.img_type}/", "/")  # safe no-op
        ensure_dir(out_path.parent)
        cv2.imwrite(str(out_path), image)
        # print(f"Saved overlay: {out_path}")

    # Assemble video
    video_path = out_dir / args.video_name
    write_video_from_frames(saved_dir, video_path, args.fps)


if __name__ == "__main__":
    main()


# python3 create_video_origin.py --input ../data/20250519_Testing/Aria/export/color --img_type color --cam 7 --out_dir investigate_registration