import os
import shutil


base_pth = "../data/20250206_Testing/Orbbec/camera04"
tar_pth = "../data/20250206_Testing/Orbbec/export/color/"
files = os.listdir(base_pth)

for file in files:
    new_name = "color_" + file.split(".")[0] + "_camera04" + ".jpg"
    shutil.copy(os.path.join(base_pth, file), os.path.join(tar_pth, new_name))