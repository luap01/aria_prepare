from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.mps.utils import get_nearest_pose

vrs_path = "/path/to/recording.vrs"
mps_traj_csv = "/path/to/mps/trajectory/closed_loop_trajectory.csv"
camera_label = "camera-slam-left"   # or "camera-slam-right" / "camera-rgb"

# 1) Open VRS + grab camera calibration and stream
provider = data_provider.create_vrs_data_provider(vrs_path)
device_calib = provider.get_device_calibration()
cam_calib   = device_calib.get_camera_calib(camera_label)   # intrinsics + T_Device_Camera
stream_id   = provider.get_stream_id_from_label(camera_label)

# 2) Load device trajectory (T_World_Device over time)
closed_loop_traj = mps.read_closed_loop_trajectory(mps_traj_csv)

# Example: for frame i
image_data, rec = provider.get_image_data_by_index(stream_id, i)  # rec has capture timestamp
ts_ns = rec.record.timestamp_ns

pose_info = get_nearest_pose(closed_loop_traj, ts_ns)
T_World_Device = pose_info.transform_world_device
T_Device_Cam   = cam_calib.get_transform_device_camera()

# 3) Compose to get camera pose and OpenCV-style extrinsics
#    T_A_B means “transform from B to A”. So:
#    T_World_Cam = T_World_Device * T_Device_Cam
T_World_Cam = T_World_Device @ T_Device_Cam

# OpenCV extrinsics [R|t] map World -> Camera, i.e. T_Cam_World = (T_World_Cam)^-1
T_Cam_World = T_World_Cam.inverse()
R = T_Cam_World.rotation().to_matrix()    # 3x3
t = T_Cam_World.translation()             # 3x1
