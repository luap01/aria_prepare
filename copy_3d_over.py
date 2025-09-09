import os
import shutil

dataset = "20250519"
frames = os.listdir(f"aria_keypoints_2d/{dataset}")
frames = [int(f.split(".")[0]) for f in frames if f.endswith(".json")]

_3D_BASE_PATH = f"../HaMuCo/test_set_result/2025-08-20_22:37/3d_kps/{dataset}"
OUTPUT_PATH = f"orbbec_joints3d/{dataset}"
total_count = 0
for frame in frames:
    src = f"{_3D_BASE_PATH}/0138.0_cam1.0_left_3d_keypoints.npz".replace("0138", f"{frame:04d}")
    dst = f"{OUTPUT_PATH}/0138.0_cam1.0_left_3d_keypoints.npz".replace("0138", f"{frame:04d}")
    if os.path.exists(src):
        shutil.copy(src, dst)
    else:
        print(f"Left Hand of {frame} couldn't be copied...")
        total_count += 1

    src = f"{_3D_BASE_PATH}/0138.0_cam1.0_right_3d_keypoints.npz".replace("0138", f"{frame:04d}")
    dst = f"{OUTPUT_PATH}/0138.0_cam1.0_right_3d_keypoints.npz".replace("0138", f"{frame:04d}")
    if os.path.exists(src):
        shutil.copy(src, dst)
    else:
        print(f"Right Hand of {frame} couldn't be copied...")
        total_count += 1


print(total_count)