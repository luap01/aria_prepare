import numpy as np
import json
import cv2

from project_orbbec_to_aria import project_orbbec_world_to_aria_uv, load_keypoints, visualize_2d_points, draw_axes
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
# Given transformation values
# location = np.array([0.039228, 0.248147, -0.430261])
location = np.array([0.039228, 0.248147, -0.530261])
scale = np.array([1.0, 1.0, 1.0])

def make_transform_matrix(rot_deg, location, scale):
    # Convert degrees to radians (mod 360 to normalize angles)
    rotation_rad = np.deg2rad(rot_deg % 360)
    rx, ry, rz = rotation_rad

    # Rotation matrices for XYZ Euler
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])

    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])

    # Combined rotation for XYZ order: Rz @ Ry @ Rx
    R = Rz @ Ry @ Rx

    # Apply scale
    R_scaled = R * scale

    # Construct transformation matrix
    T = np.eye(4)
    T[:3, :3] = R_scaled
    T[:3, 3] = location

    return T

# Original with 293° Z rotation
T_original = make_transform_matrix(np.array([0, 180, 296]), location, scale)

T_orbbec_aria_may_blender = np.array([
    [-0.439208637,  0.897776552,  0.033061091,  0.040834397],
    [ 0.644760799,  0.340628146, -0.684292319, -0.136309333],
    [-0.625603136, -0.279230602, -0.728457951, -0.425655865],
    [ 0.000000000,  0.000000000,  0.000000000,  1.000000000]
], dtype=np.float32)

frame = 17900
T_aria_world_orbbec = np.linalg.inv(T_original)  # Use the rotation matrix for the transformation
img_pth = f"../data/20250519_Testing/Aria/export/color/color_{frame:06d}_camera07.jpg"
img = cv2.imread(img_pth)


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

T_original = T_aria_world_orbbec
print(

    f"""T_orbbec_aria_may_blender = np.array([
    [{T_original[0, 0]:.12f}, {T_original[0, 1]:.12f}, {T_original[0, 2]:.12f}, {T_original[0, 3]:.12f}],
    [{T_original[1, 0]:.12f}, {T_original[1, 1]:.12f}, {T_original[1, 2]:.12f}, {T_original[1, 3]:.12f}],
    [{T_original[2, 0]:.12f}, {T_original[2, 1]:.12f}, {T_original[2, 2]:.12f}, {T_original[2, 3]:.12f}],
    [{T_original[3, 0]:.12f}, {T_original[3, 1]:.12f}, {T_original[3, 2]:.12f}, {T_original[3, 3]:.12f}]
    ], dtype=np.float32)
    """
)

with open(img_pth.replace("color", "calib").replace(".jpg", ".json").replace(f"_camera07", ""), "r") as f:
        calib = json.load(f)

K_aria = np.array(calib["destination_intrinsics_rotated"]['K'], dtype=np.float32)

T_world_cam_aria = np.array(calib["extrinsics"]["T_world_cam"], dtype=np.float32)  # 4x4
T_cam_world_aria = np.linalg.inv(T_world_cam_aria)
    
M = T_cam_world_aria @ T_aria_world_orbbec # world(Orbbec) → camera(Aria)


npz_path = f"../HaMuCo/3d_data/20250519/{frame}.0_cam1.0_right_3d_keypoints.npz"
# npz_path = f"../dataset/hamuco_test/20250519/3D/{frame}.0_cam1.0_right_3d_keypoints.npz"
xyz_world_left = load_keypoints(npz_path, "left")
xyz_world_right = load_keypoints(npz_path, "right")

# Project points
projected_uv_left = []
projected_uv_right = []
for xyz in xyz_world_left:
    _2d_point = project_orbbec_world_to_aria_uv(xyz, M, K_aria)
    projected_uv_left.append(_2d_point)
for xyz in xyz_world_right:
    _2d_point = project_orbbec_world_to_aria_uv(xyz, M, K_aria)
    projected_uv_right.append(_2d_point)

image = visualize_2d_points(projected_uv_left, img, RED, GREEN)
image = visualize_2d_points(projected_uv_right, image, BLUE, YELLOW)
image = draw_axes(image, K_aria, M)

cv2.imshow("image", image)
cv2.waitKey(0)