import os
import shutil
import json


files_path = "../hand_detection_preprocessing/output/20250519_Testing/mediapipe_0.10/camera07/json"
txt_file = "input.txt"
with open(txt_file, "r") as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split()
    if len(parts) != 5 or parts[1] != "camera" or parts[4] == "":
        print(f"Skipping malformed line: {line.strip()}")
        continue

    handside = parts[0]                     # "left" / "right" / "both"
    camera = f"camera0{parts[2]}"           # camera number
    frame_idx = f"{int(parts[4]):06d}"      # frame number (string, e.g., "1")

    src = f"{files_path}/{frame_idx}.json"
    dst = f"aria_keypoints_2d/{frame_idx}.json"

    with open(src, 'r') as f:
        data = json.load(f)

    if handside == "left":
        data['people'][0]['hand_right_conf'] = 0
    if handside == "right":
        data['people'][0]['hand_left_conf'] = 0

    with open(dst, 'w') as f:
        json.dump(data, f, indent=4)

