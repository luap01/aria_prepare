import json
import numpy as np


import argparse
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))


from HaMuCo.utils.camera import load_cam_infos, project_3d_to_2d, project_to_2d
from HaMuCo.utils.image import undistort_image
# from utils.viz_from_author import draw_2d_skeleton


### 20250206
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
    [-4.69185573e-01,  8.81939689e-01,  4.52469007e-02,  6.92280000e-02],
    [ 8.82409725e-01,  4.66178448e-01,  6.34880450e-02,  3.08150000e-01],
    [ 3.48994967e-02,  6.97139799e-02, -9.96956361e-01, -5.20260000e-01],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
], dtype=np.float32)

T_orbbec_aria_may_blender = np.array([
    [-0.406674695115, 0.913236685669, 0.024788067713, 0.039228000000],
    [0.913406320245, 0.405932447652, 0.030128758370, 0.218147000000],
    [0.017452406437, 0.034894181340, -0.999238614955, -0.390261000000],
    [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]
    ], dtype=np.float32)

T_orbbec_aria_may_blender = np.array([
    [-0.469185573395, 0.881939689375, 0.045246900704, 0.039228000000],
    [0.882409725042, 0.466178447914, 0.063488044954, 0.218147000000],
    [0.034899496703, 0.069713979985, -0.996956361194, -0.390261000000],
    [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]
    ], dtype=np.float32)

# T_orbbec_aria_may_blender = np.array([
#     [-0.406736643076, 0.911320106822, 0.063725709730, 0.039228000000],
#     [0.913545457643, 0.405745853056, 0.028372513963, 0.168147000000],
#     [-0.000000000000, 0.069756473744, -0.997564050260, -0.320261000000],
#     [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]
#     ], dtype=np.float32)

# T_orbbec_aria_may_blender = np.array([
#     [-0.406488870102, 0.910329920365, 0.077886035791, 0.039228000000],
#     [0.912988950433, 0.407969848252, -0.003432099099, 0.208150000000],
#     [-0.034899496703, 0.069713979985, -0.996956361194, -0.320261000000],
#     [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]
#     ], dtype=np.float32)

# T_orbbec_aria_may_blender = np.array([
#     [-0.406736643076, 0.912988950433, 0.031882276687, 0.039228000000],
#     [0.913545457643, 0.406488870102, 0.014194904134, 0.208150000000],
#     [-0.000000000000, 0.034899496703, -0.999390827019, -0.430261000000],
#     [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]
#     ], dtype=np.float32)  
# 
T_orbbec_aria_may_blender = np.array([
    [-0.469185573395, 0.882981529607, 0.014440086234, 0.039228000000],
    [0.882409725042, 0.468110165415, 0.047179976547, 0.208150000000],
    [0.034899496703, 0.034878236872, -0.998782025130, -0.430261000000],
    [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]
    ], dtype=np.float32)  

T_orbbec_aria_may_blender = np.array([
    [-0.438104102931, 0.897712600210, -0.046657072551, 0.049228000000],
    [0.898246525251, 0.439198811493, 0.016049419254, 0.238147000000],
    [0.034899496703, -0.034878236872, -0.998782025130, -0.430261000000],
    [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]
    ], dtype=np.float32)

T_orbbec_aria_may_blender = np.array([
    [-0.406488870102, 0.913545457643, -0.014194904134, 0.049228000000],
    [0.912988950433, 0.406736643076, 0.031882276687, 0.238147000000],
    [0.034899496703, 0.000000000000, -0.999390827019, -0.430261000000],
    [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]
    ], dtype=np.float32)
    
T_orbbec_aria_may_blender = np.array([
    [-0.374378393201, 0.927183854567, -0.013073581572, 0.049228000000],
    [0.926619039214, 0.374606593416, 0.032358249875, 0.238147000000],
    [0.034899496703, 0.000000000000, -0.999390827019, -0.430261000000],
    [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]
    ], dtype=np.float32)

T_orbbec_aria_may_blender = np.array([
    [-0.469185573395, 0.882947592859, -0.016384321257, 0.049228000000],
    [0.882409725042, 0.469471562786, 0.030814426605, 0.238147000000],
    [0.034899496703, 0.000000000000, -0.999390827019, -0.430261000000],
    [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]
    ], dtype=np.float32)

T_orbbec_aria_may_blender = np.array([
    [-0.499695413510, 0.866025403784, -0.017449748351, 0.049228000000],
    [0.865497844508, 0.500000000000, 0.030223850724, 0.238147000000],
    [0.034899496703, 0.000000000000, -0.999390827019, -0.430261000000],
    [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]
    ], dtype=np.float32)

T_orbbec_aria_may_blender = np.array([
    [-0.469185573395, 0.882947592859, -0.016384321257, 0.049228000000],
    [0.882409725042, 0.469471562786, 0.030814426605, 0.338147000000],
    [0.034899496703, 0.000000000000, -0.999390827019, -0.430261000000],
    [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]
    ], dtype=np.float32)

T_orbbec_aria_may_blender = np.array([
    [-0.499695413510, 0.866025403784, -0.017449748351, 0.029228000000],
    [0.865497844508, 0.500000000000, 0.030223850724, 0.288147000000],
    [0.034899496703, 0.000000000000, -0.999390827019, -0.430261000000],
    [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]
    ], dtype=np.float32)

T_orbbec_aria_may_blender = np.array([
    [-0.499695413510, 0.866106831943, 0.012784732288, 0.029228000000],
    [0.865497844508, 0.498640616331, 0.047655187522, 0.288147000000],
    [0.034899496703, 0.034878236872, -0.998782025130, -0.430261000000],
    [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]
    ], dtype=np.float32)

T_orbbec_aria_may_rot = np.array([
    [-0.46918556,  0.8819397,   0.0452469,   0.069228  ],
    [-0.0348995,  -0.06971398,  0.99695635,  0.52026004],
    [ 0.88240975,  0.46617845,  0.06348804,  0.30815   ],
    [ 0.,          0.,          0.,          1.       ]
], dtype=np.float32)

T_test = np.array([[-0.6181538090350368, -0.1063716553803718, 0.778826607833798, -0.9931267024496195],
 [0.7785545315293727, 0.05371412809009257, 0.6252740380316221, -0.5667186101408318],
 [-0.10834543334856978, 0.9928745321139659, 0.049612547553643194, -1.0957737132220555],
 [0, 0, 0, 1]], dtype=np.float32)

T_orbbec_aria_pc = np.array([
    [0.9982601515265793, -0.042440806013835736, -0.04093223496252641, -0.08015957460701544],
    [0.041178362215549866, 0.9986643192323527, -0.03120769099800983, -0.010123113225094673],
    [0.04220204212329638, 0.029467871946889077, 0.9986744375238337, 0.0681677059867391],
    [0, 0, 0, 1]
    ], dtype=np.float32
)

T_orbbec_aria_manual = np.array([
    [-0.5792939300796813, 0.3518765587971109, -0.7352560301976045, 0.11931433162446434],
    [0.08708143653881648, 0.9235770560852863, 0.37339288274305943, 1.2860351818909719],
    [0.8104538024978187, 0.15227707917457373, -0.5656644987754555, -0.435745322435934],
    [0, 0, 0, 1]], dtype=np.float32)


T_chat = np.array([
    [0.8524148672800617, 0.4737642410868342, -0.22121559146527925, 0.019009140811304462],
    [-0.4253451622813542, 0.8743672003009622, 0.23358829585770804, -0.027341947685667188],
    [0.30408943908621067, -0.10502115456075868, 0.946836929006765, 0.4545024617122436],
    [0, 0, 0, 1]], dtype=np.float32)

T_registration = np.array([[0.21878664082595478, 0.973338419508004, 0.06888226013829635, 0.21404077149941395],
 [-0.9146769224519737, 0.17998955421259927, 0.3618976573359846, -0.32993483694885295],
 [0.3398508031002617, -0.14218338714767118, 0.9296694537050082, 0.646172914611638],
 [0, 0, 0, 1]], dtype=np.float32)

T_icp = np.array([[-0.6455235504324923, 0.7619369741406152, -0.052453925404275056, -0.01686039594837324],
 [-0.20213610341626118, -0.1042135259357412, 0.9737969588785886, 0.5982856677311795],
 [0.7365055169995216, 0.639211719656528, 0.22128723842033082, 0.5585193423744806],
 [0, 0, 0, 1]], dtype=np.float32)

T_icp = np.array([
    [
      -0.5007856980125139,
      -0.21828719872806032,
      -0.8375943984216234,
      0.13615728989624445
    ],
    [
      -0.3008247739813681,
      -0.8634640892244222,
      0.4048879467278259,
      0.13062789423752466
    ],
    [
      -0.8116145462776966,
      0.4547312241058213,
      0.36674425873232325,
      0.1466796235374037
    ],
    [
      0.0,
      0.0,
      0.0,
      1.0
    ]
], dtype=np.float32)



T_icp = np.array([
    [-0.49969545006752014, 0.8654978275299072, 0.03489949554204941, -0.21871385504823773],
    [0.8661068677902222, 0.49864062666893005, 0.034878239035606384, -0.15232917858016465],
    [0.012784730643033981, 0.04765518754720688, -0.9987820386886597, -0.43349152738645297],
    [0, 0, 0, 1]
    ], dtype=np.float32
)

T_aria_orbbec = np.array([
    [[-0.45168289232295594, 0.0008420960419838819, 0.8921781524208192, -0.13293935255942463],
 [0.8921325937876252, 0.0105760394197034, 0.4516498450038007, -0.15670980033821225],
 [-0.009055378762560488, 0.9999437175483673, -0.005528277400566351, -0.3833568448904635],
 [0, 0, 0, 1]]
])

T_orbbec_world_aria = np.array(
    [[-0.4681186142558247, 0.8835560378220539, -0.013917291956442377, 0.06874350185616492],
     [-0.01220315975679898, 0.009284221034147984,  0.9998824361552404, 0.38641361546505054],
     [ 0.8835813747919075, 0.46823341536885565,   0.0064360587986766835, 0.19066101889192952],
     [ 0., 0., 0., 1. ]], dtype=np.float32
)

def draw_axes(img, intrinsics, extrinsics, axis_length=0.2):
    """
    Draws the 3D coordinate axes (x=red, y=green, z=blue).
    """
    origin = project_to_2d(np.zeros(3), intrinsics, extrinsics)
    x_axis = project_to_2d(np.array([axis_length, 0, 0]), intrinsics, extrinsics)
    y_axis = project_to_2d(np.array([0, axis_length, 0]), intrinsics, extrinsics)
    z_axis = project_to_2d(np.array([0, 0, axis_length]), intrinsics, extrinsics)

    origin = tuple(map(int, origin))
    x_axis = tuple(map(int, x_axis))
    y_axis = tuple(map(int, y_axis))
    z_axis = tuple(map(int, z_axis))

    print(origin, x_axis, y_axis, z_axis)
    colors = {
        "red": (255, 0, 0),  # Red
        "green": (0, 255, 0),  # Green
        "blue": (0, 0, 255),  # Blue
    }
    cv2.arrowedLine(img, origin, x_axis, colors["red"], 10, tipLength=0.1)   # x+ in red
    cv2.arrowedLine(img, origin, y_axis, colors["blue"], 10, tipLength=0.1)   # y+ in blue
    cv2.arrowedLine(img, origin, z_axis, colors["green"], 10, tipLength=0.1)   # z+ in green

    return img


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project 3D keypoints from .npz onto an image")
    parser.add_argument("--npz", required=True, type=str, help="Path to 3D keypoints .npz in 3d_data/")
    parser.add_argument("--cam", default=1, type=int, help="Which camera to project from the .npz")
    return parser.parse_args()


def load_keypoints(npz_file: Path, hand: str) -> np.ndarray:
    data = np.load(str(npz_file))
    if hand not in data:
        raise KeyError(f"Hand '{hand}' not found in {npz_file.name}. Available: {list(data.keys())}")
    xyz = data[hand]
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected shape (N,3), got {xyz.shape} in {npz_file}")
    return xyz.astype(np.float32)


def project_orbbec_world_to_aria_uv(p_orbbec_world_xyz, M, K_aria):
    p_h = np.array([p_orbbec_world_xyz[0], p_orbbec_world_xyz[1], p_orbbec_world_xyz[2], 1.0], dtype=np.float32)
    p_cam = (M @ p_h)[:3]
    uvw = K_aria @ p_cam
    return uvw[:2] / uvw[2]  # (u, v) in pixels

def project_points(X_world_orbbec, rvec, t, K_aria, dist=None):
    # X_world_orbbec: (N,3)
    # transform to Aria camera is already inside M via rvec,t when using projectPoints
    # so just pass the *orbbec-world* 3D points:
    img_pts, _ = cv2.projectPoints(
        X_world_orbbec.astype(np.float32).reshape(-1,1,3),
        rvec, t.astype(np.float32),
        K_aria, dist
    )
    return img_pts.reshape(-1,2)

def visualize_2d_points(points_2d, img, dot_colour, line_colour):
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),      # Index
        (0, 9), (9, 10), (10, 11), (11, 12), # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
    ]
    for i, (x, y) in enumerate(points_2d):
        cv2.circle(img, (int(x), int(y)), 4, dot_colour, -1)

    for idx1, idx2 in HAND_CONNECTIONS:
        x1, y1 = int(points_2d[idx1][0]), int(points_2d[idx1][1])
        x2, y2 = int(points_2d[idx2][0]), int(points_2d[idx2][1])
        cv2.line(img, (x1, y1), (x2, y2), line_colour, 2)

    return img

def main():
    args = parse_args()
    test_aria_kps = False

    npz_path = Path(args.npz).expanduser().resolve()
    img_idx = str(int(npz_path.stem.split("_")[0].split(".")[0])).zfill(6)
    if not npz_path.exists():
        raise FileNotFoundError(f".npz file not found: {npz_path}")
    
    dataset = str(npz_path).split("/")[-2]
    offset = -4 if int(args.cam) > 4 and int(args.cam) < 7 else 0
    cam_name = 'Orbbec' if int(args.cam) < 5 else 'Marshall' if int(args.cam) < 7 else 'Aria'
    img_type = "raw"
    img_path = f"../data/{dataset}_Testing/{cam_name}/export/{img_type}/{img_type}_{img_idx}_camera0{int(args.cam) + offset}.jpg"
    print(img_path)


    # T_aria_world_orbbec = np.linalg.inv(T_test)
    # T_aria_world_orbbec = np.linalg.inv(T_registration)
    # T_aria_world_orbbec = T_orbbec_aria_may_rot
    # T_aria_world_orbbec = np.linalg.inv(T_orbbec_aria_may_blender)  # Use the rotation matrix for the transformation
    T_aria_world_orbbec = T_icp
    X = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
        ], dtype=np.float32)

    T_aria_world_orbbec = T_aria_world_orbbec @ X  # Apply the rotation to the transformation matrix

    X = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
        ], dtype=np.float32)
    
    T_aria_world_orbbec = T_aria_world_orbbec @ X  # Flip Y axis

    T_aria_world_orbbec = np.linalg.inv(T_orbbec_world_aria)
    # T_aria_world_orbbec = (T_icp)
    # T_aria_world_orbbec = np.linalg.inv(T_aria_world_orbbec)
    print(f"Transformation matrix from Orbbec-world to Aria-world: {np.linalg.inv(T_aria_world_orbbec)}")

    with open(img_path.replace("color", "calib").replace('raw', 'calib').replace(".jpg", ".json").replace(f"_camera0{int(args.cam) + offset}", ""), "r") as f:
        calib = json.load(f)

    K_aria = np.array(calib["destination_intrinsics_rotated"]['K'], dtype=np.float32)
    # K_aria = np.array(calib["destination_intrinsics"]['K'], dtype=np.float32)
    # K_aria = np.array(calib["intrinsics"]['K'], dtype=np.float32)
    print(f"Aria intrinsics: {K_aria}")
    
    T_world_cam_aria = np.array(calib["extrinsics"]["T_world_cam"], dtype=np.float32)  # 4x4
    T_cam_world_aria = np.linalg.inv(T_world_cam_aria)
    
    M = T_cam_world_aria @ T_aria_world_orbbec # world(Orbbec) → camera(Aria)

    
    # M = M @ X  # Apply the rotation to the transformation matrix

    # X = np.array([
    #     [1, 0, 0, 0],
    #     [0, -1, 0, 0],
    #     [0, 0, 1, 0],
    #     [0, 0, 0, 1]
    #     ], dtype=np.float32)
    
    # M = M @ X

    print(f"Transformation matrix from Orbbec-world to Aria-camera: {M}")
    
    # Load data
    # xyz_world_left = load_keypoints(npz_path, "left")
    # xyz_world_right = load_keypoints(npz_path, "right")

    # Project points
    # projected_uv_left = []
    # projected_uv_right = []
    # for xyz in xyz_world_left:
    #     _2d_point = project_orbbec_world_to_aria_uv(xyz, M, K_aria)
    #     projected_uv_left.append(_2d_point)
    # for xyz in xyz_world_right:
    #     _2d_point = project_orbbec_world_to_aria_uv(xyz, M, K_aria)
    #     projected_uv_right.append(_2d_point)

    # R = M[:3, :3]
    # t = M[:3, 3]
    # rvec, _ = cv2.Rodrigues(R.astype(np.float32))
    # dist = np.zeros((5,1), np.float32)  # since rectified

    # print(xyz_world_left)

    # projected_uv_left = project_points(xyz_world_left, rvec, t, K_aria, dist)
    # projected_uv_right = project_points(xyz_world_right, rvec, t, K_aria, dist)
   
    
    # projected_uv_left = np.array(projected_uv_left)
    # projected_uv_right = np.array(projected_uv_right)

    # Draw and save
    image = cv2.imread(str(img_path.replace("color", img_type)))
    if image is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    os.makedirs(f"project_results/{dataset}/camera0{int(args.cam)}", exist_ok=True)
    out_path = f"project_results/{dataset}/camera0{int(args.cam)}/{img_type}_{img_idx}_overlay.jpg"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    YELLOW = (0, 255, 255)

    if test_aria_kps:
        kps_pth = Path(img_path.replace("color", "hand").replace(".jpg", ".npy").replace(f"_camera0{int(args.cam) + offset}", ""))
        print(f"Loading keypoints from: {kps_pth}")
        kps_loaded = np.load(kps_pth, allow_pickle=True)

        # np.save of a dict yields a 0-d object array on load; extract the dict via .item()
        kps = kps_loaded.item() if isinstance(kps_loaded, np.ndarray) else kps_loaded
        left_kps = np.array([kps['left_wrist'], kps['left_palm']], dtype=np.float32)
        right_kps = np.array([kps['right_wrist'], kps['right_palm']], dtype=np.float32)

        left_kps = left_kps.reshape(2, 3)
        right_kps = right_kps.reshape(2, 3)

        projected_uv_left = []
        projected_uv_right = []
        for kp in left_kps:
            # Project a 3D Aria world point to the image using extrinsics and intrinsics
            kp_h = np.append(kp, 1.0)  # make homogeneous
            cam_xyz = T_cam_world_aria @ kp_h  # transform to camera coordinates
            uvw = K_aria @ cam_xyz[:3]  # project to image plane
            _2d_point = uvw[:2] / uvw[2]  # normalize to get (u, v) in pixels
            projected_uv_left.append(_2d_point)
        for kp in right_kps:
            kp_h = np.append(kp, 1.0)  # make homogeneous
            cam_xyz = T_cam_world_aria @ kp_h  # transform to camera coordinates
            uvw = K_aria @ cam_xyz[:3]  # project to image plane
            _2d_point = uvw[:2] / uvw[2]  # normalize to get (u, v) in pixels
            projected_uv_right.append(_2d_point)

        projected_uv_left = np.array(projected_uv_left).reshape(-1, 2)
        projected_uv_right = np.array(projected_uv_right).reshape(-1, 2)


        for x,y in projected_uv_left:
            cv2.circle(image, (int(x), int(y)), 4, RED, -1)
        for x,y in projected_uv_right:
            cv2.circle(image, (int(x), int(y)), 4, BLUE, -1)
    else:
        np.savez(out_path.replace(".jpg", ".npy"),
            left=projected_uv_left.astype(np.float32),
            right=projected_uv_right.astype(np.float32)
        )
        image = visualize_2d_points(projected_uv_left, image, RED, GREEN)
        image = visualize_2d_points(projected_uv_right, image, BLUE, YELLOW)
        print(K_aria)
        print(M)
        image = draw_axes(image, K_aria, M)
    cv2.imwrite(str(out_path), image)
    print(f"Saved overlay: {out_path}")


if __name__ == "__main__":
    main()