"""
Aria frame extraction and processing module.

This module provides functionality to extract and process frames from Aria device recordings,
including timestamp synchronization and image transformation operations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import sys

import open3d as o3d
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import rootutils
from PIL import Image
import json
from projectaria_tools.core import calibration, data_provider, mps
from projectaria_tools.core.image import InterpolationMethod
from projectaria_tools.core.mps.utils import (
    filter_points_from_confidence,
)
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import RecordableTypeId
from tqdm import tqdm
from point_cloud import denoise_point_cloud, voxelize_point_cloud

rootutils.setup_root(__file__, ".project-root", pythonpath=True)

def save_ply_xyz(path, pts):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
        for x,y,z in pts:
            f.write(f"{x} {y} {z}\n")


@dataclass
class ProcessingConfig:
    """Configuration for frame processing parameters."""

    aria_name: str
    recording_name: str
    base_dir: Path
    devignetting_mask_path: Path
    output_image_size: Tuple[int, int] = (1024, 1024)
    focal_length: float = 300.0
    camera_label: str = "camera-rgb"

    @property
    def mps_path(self) -> Path:
        """Get the MPS path for the recording."""
        return self.base_dir / f"mps_{self.aria_name}_vrs"

    @property
    def vrs_file(self) -> Path:
        """Get the VRS file path for the recording."""
        return self.base_dir / f"{self.aria_name}.vrs"


class AriaFrameProcessor:
    """Handles extraction and processing of frames from Aria device recordings."""

    def __init__(self, config: ProcessingConfig):
        """
        Initialize the frame processor.

        Args:
            config: Processing configuration parameters
        """
        self.config = config
        self.provider = self._initialize_provider()
        self.mps_data_provider = self._initialize_mps_provider()
        self.device_calib = self._initialize_calibration()
        self.point_cloud = self._get_point_cloud()
        self.hand_tracking_results = self._load_hand_tracking_results()

    def _initialize_provider(self) -> data_provider.VrsDataProvider:
        """Initialize the VRS data provider."""
        if not self.config.vrs_file.exists():
            raise FileNotFoundError(f"VRS file not found: {self.config.vrs_file}")
        return data_provider.create_vrs_data_provider(str(self.config.vrs_file))

    def _initialize_mps_provider(self) -> mps.MpsDataProvider:
        """Initialize the MPS data provider."""
        paths_provider = mps.MpsDataPathsProvider(str(self.config.mps_path))
        return mps.MpsDataProvider(paths_provider.get_data_paths())

    def _initialize_calibration(self) -> calibration.DeviceCalibration:
        """Initialize device calibration with devignetting masks."""
        device_calib = self.provider.get_device_calibration()
        device_calib.set_devignetting_mask_folder_path(str(self.config.devignetting_mask_path))
        return device_calib

    def _get_point_cloud(self):
        """Get point cloud data from the VRS file."""
        points = self.mps_data_provider.get_semidense_point_cloud()
        points = filter_points_from_confidence(points, threshold_invdep=0.01, threshold_dep=0.002)
        # Retrieve point position
        point_cloud = np.stack([it.position_world for it in points])
        return point_cloud
    
    def _load_hand_tracking_results(self):
        """
        Load 21-landmark hand-tracking results if present.
        Returns a list-like structure usable by mps.utils.get_nearest_hand_tracking_result.
        """
        # <mps_root>/hand_tracking/hand_tracking_results.csv
        csv_path = self.config.mps_path / "hand_tracking" / "hand_tracking_results.csv"
        if csv_path.exists():
            print(f"Hand tracking (21 landmarks): {csv_path}")
            return mps.hand_tracking.read_hand_tracking_results(str(csv_path))
        else:
            print("[INFO] 21-landmark hand_tracking_results.csv not found; "
                  "falling back to wrist/palm only (if available).")
            return None

    @staticmethod
    def _transform_points(T_world_device_mat: np.ndarray, pts_device: np.ndarray) -> np.ndarray:
        """Apply world<-device to Nx3 points."""
        pts_device = np.asarray(pts_device, dtype=np.float32).reshape(-1, 3)
        ones = np.ones((pts_device.shape[0], 1), dtype=np.float32)
        return (T_world_device_mat @ np.hstack([pts_device, ones]).T).T[:, :3]

    @staticmethod
    def _rotate_vectors(R_world_device: np.ndarray, vecs_device: np.ndarray) -> np.ndarray:
        """Rotate Nx3 vectors from device to world (no translation)."""
        vecs_device = np.asarray(vecs_device, dtype=np.float32).reshape(-1, 3)
        return (R_world_device @ vecs_device.T).T

    def _query_hand_21_world(self, device_timestamp_us: int, T_world_device_mat: np.ndarray):
        """
        Return a dict with 21-landmark arrays (in world) and confidences, if available for this timestamp.
        Keys: left_landmarks_world (21x3), right_landmarks_world (21x3),
              left_conf, right_conf,
              left_wrist_normal_world/right_wrist_normal_world (3,), optional
              left_palm_normal_world/right_palm_normal_world (3,), optional
        """
        out = {}
        if self.hand_tracking_results is None:
            return out

        # Locate the nearest hand-tracking row to this timestamp
        htr = mps.utils.get_nearest_hand_tracking_result(
            self.hand_tracking_results,
            int(device_timestamp_us * 1000)  # ns
        )
        if not htr:
            return out

        R_wd = T_world_device_mat[:3, :3]

        for tag, perhand in (("left", htr.left_hand), ("right", htr.right_hand)):
            if perhand and getattr(perhand, "confidence", 0.0) > 0.0:
                # 21x3 in device frame -> world
                lm_dev = np.asarray(perhand.landmark_positions_device, dtype=np.float32)
                lm_w   = self._transform_points(T_world_device_mat, lm_dev)
                out[f"{tag}_landmarks_world"] = lm_w
                out[f"{tag}_conf"]            = float(perhand.confidence)

                # normals: rotate only (no translation)
                wn = getattr(perhand, "wrist_and_palm_normal_device", None)
                if wn is not None:
                    out[f"{tag}_wrist_normal_world"] = self._rotate_vectors(
                        R_wd, np.asarray(wn.wrist_normal_device, dtype=np.float32)
                    ).reshape(3)
                    out[f"{tag}_palm_normal_world"] = self._rotate_vectors(
                        R_wd, np.asarray(wn.palm_normal_device, dtype=np.float32)
                    ).reshape(3)
        return out


    def process_frames(self, output_dirs: Dict[str, Path]) -> None:
        """
        Process and extract frames using synchronized timestamps.

        Args:
            output_dirs: Dictionary mapping camera labels to output directories
        """
        self._ensure_output_dirs(output_dirs)
        timestamps = self._get_synchronized_timestamps()
        self._extract_frames(timestamps, output_dirs)

    def _ensure_output_dirs(self, output_dirs: Dict[str, Path]) -> None:
        """Create output directories if they don't exist."""
        for output_dir in output_dirs.values():
            output_dir.mkdir(parents=True, exist_ok=True)

    def _get_synchronized_timestamps(self) -> pd.DataFrame:
        """Get synchronized timestamps between Aria and Orbecc devices."""
        orbecc_timestamps = self._load_orbecc_timestamps()
        aria_timestamps = self._load_aria_timestamps()
        return self._synchronize_timestamps(aria_timestamps, orbecc_timestamps)

    def _load_orbecc_timestamps(self) -> pd.DataFrame:
        """Load Orbecc timestamps from Arrow file."""
        timestamps_file = (
            # Path("../data") / "recordings" / self.config.recording_name / "Orbbec" / "tables_timestamps.arrow"
            Path(str(self.config.base_dir).replace("Aria", "Orbbec")) / "tables_timestamps.arrow"
        )
        if not timestamps_file.exists():
            raise FileNotFoundError(f"Orbecc timestamps file not found: {timestamps_file}")
        print(f"Orbbec timestamps: {timestamps_file}")
        return ds.dataset(timestamps_file, format="arrow").to_table().to_pandas()

    def _load_aria_timestamps(self) -> pd.DataFrame:
        """Load Aria timestamps from CSV file."""
        timestamps_file = self.config.mps_path / "slam" / "closed_loop_trajectory.csv"
        if not timestamps_file.exists():
            raise FileNotFoundError(f"Aria timestamps file not found: {timestamps_file}")
        print(f"Aria timestamps: {timestamps_file}")
        return pd.read_csv(timestamps_file, usecols=["tracking_timestamp_us", "utc_timestamp_ns"])

    def _synchronize_timestamps(self, aria_timestamps: pd.DataFrame, orbecc_timestamps: pd.DataFrame) -> pd.DataFrame:
        """
        Synchronize timestamps between Aria and Orbecc devices.

        Args:
            aria_timestamps: DataFrame containing Aria device timestamps
            orbecc_timestamps: DataFrame containing Orbecc device timestamps

        Returns:
            DataFrame with synchronized timestamps
        """
        # Calculate median timestamps for Orbecc frames
        timestamp_cols = [col for col in orbecc_timestamps.columns if col != "frame_number"]
        orbecc_median_timestamps = orbecc_timestamps[timestamp_cols].median(axis=1)

        # Initialize result with frame numbers
        result = pd.DataFrame()

        frame_nums = []
        tracking_timestamps = []
        for frame_num, orbecc_median_timestamp in zip(orbecc_timestamps["frame_number"], orbecc_median_timestamps):
            closest_idx = (aria_timestamps["utc_timestamp_ns"] - orbecc_median_timestamp).abs().idxmin()
            diff_ns = np.abs((aria_timestamps["utc_timestamp_ns"][closest_idx] - orbecc_median_timestamp))

            if diff_ns > 1e7:
                print(f"Skipping {frame_num}...Difference in nanoseconds: {diff_ns}")
                continue
            
            if self.config.recording_name == "20250206_Testing":
                offset = -7
            else:
                offset = 15
            frame_nums.append(frame_num - offset)
            # frame_nums.append(frame_num + 7)
            tracking_timestamps.append(aria_timestamps.loc[closest_idx, "tracking_timestamp_us"])

        result["frame_number"] = frame_nums
        result["tracking_timestamp_us"] = tracking_timestamps
        return result

    def _extract_frames(self, timestamps: pd.DataFrame, output_dirs: Dict[str, Path]) -> None:
        """
        Extract and process frames using the provided timestamps.

        Args:
            timestamps: DataFrame containing synchronized timestamps
            output_dirs: Dictionary mapping camera labels to output directories
        """
        options = self.provider.get_default_deliver_queued_options()
        rgb_stream_ids = options.get_stream_ids(RecordableTypeId.RGB_CAMERA_RECORDABLE_CLASS)
        options.set_subsample_rate(rgb_stream_ids[0], 1)

        # Initialize calibrations
        src_calib = self.device_calib.get_camera_calib(self.config.camera_label)
        dst_calib = calibration.get_linear_camera_calibration(
            self.config.output_image_size[0],
            self.config.output_image_size[1],
            self.config.focal_length,
            self.config.camera_label,
        )
        devignetting_mask = self.device_calib.load_devignetting_mask(self.config.camera_label)

        for _, row in tqdm(timestamps.iterrows(), total=len(timestamps)):
            self._process_single_frame(
                row["frame_number"],
                row["tracking_timestamp_us"],
                rgb_stream_ids[0],
                src_calib,
                dst_calib,
                devignetting_mask,
                output_dirs[self.config.camera_label],
            )

    def _process_single_frame(
        self,
        frame_number: int,
        device_timestamp: int,
        stream_id: int,
        src_calib: calibration.CameraCalibration,
        dst_calib: calibration.CameraCalibration,
        devignetting_mask: np.ndarray,
        output_dir: Path,
    ) -> None:
        """
        Process a single frame with the given parameters.

        Args:
            frame_number: Frame number for the output filename
            device_timestamp: Device timestamp in microseconds
            stream_id: RGB stream ID
            src_calib: Source camera calibration
            dst_calib: Destination camera calibration
            devignetting_mask: Devignetting mask for image correction
            output_dir: Output directory for the processed frame
        """
        image_data = self.provider.get_image_data_by_time_ns(
            stream_id,
            int(device_timestamp * 1000),
            TimeDomain.DEVICE_TIME,
            TimeQueryOptions.CLOSEST,
        )

        fx, fy = src_calib.get_focal_lengths()         # floats
        cx, cy = src_calib.get_principal_point()
        dfx, dfy = dst_calib.get_focal_lengths()
        dcx, dcy = dst_calib.get_principal_point()
        height, width = dst_calib.get_image_size()
        # Compute rotated destination intrinsics for 90° CW (np.rot90 k=3)
        dst_h, dst_w = height, width
        # print(f"Image {frame_number}: {dst_h}, {dst_w}")
        # After 90° CW rotation: fx' = fy, fy' = fx, cx' = H-1-cy, cy' = cx
        rdfx, rdfy = float(dfy), float(dfx)
        rdcx = float(dst_h - 1 - dcy)
        rdcy = float(dcx)
        
        A = np.array([[0, -1, height - 1],
                  [1,  0,           0],
                  [0,  0,           1]], dtype=np.float32)

        K_aria = np.array([[fx, 0,  cx],
                    [0,  fy, cy],
                    [0,  0,   1]], dtype=np.float32)
        
        K_destination = np.array([[dfx, 0, dcx],
                  [0,  dfy, dcy],
                  [0,  0, 1]], dtype=np.float32)
        
        rotated_intrinsics = np.array([[0, -dfy, -dcy + height - 1],
                  [dfx,  0, dcx],
                  [0,  0, 1]], dtype=np.float32)


        T_Device_Cam = src_calib.get_transform_device_camera()

        if not image_data:
            print(f"Warning: No image data found for frame {frame_number}")
            return

        # [INFO] Get wrist and palm pose
        pose_info = self.mps_data_provider.get_closed_loop_pose(
            int(device_timestamp * 1000), TimeQueryOptions.CLOSEST
        )
        T_World_Device = pose_info.transform_world_device
        T_world_device_matrix = T_World_Device.to_matrix()
        wrist_and_palm_pose = self.mps_data_provider.get_wrist_and_palm_pose(
            int(device_timestamp * 1000), TimeQueryOptions.CLOSEST
        )

        T_World_Cam = T_World_Device @ T_Device_Cam
        T_Cam_World = T_World_Cam.inverse()

        R = T_Cam_World.rotation().to_matrix()    # 3x3
        t = T_Cam_World.translation()             # 3x1

        # Save per-frame intrinsics/extrinsics as JSON
        calib_dir = Path(str(output_dir).replace("color", "calib"))
        calib_dir.mkdir(parents=True, exist_ok=True)
        calib_dict = {
            "frame_number": int(frame_number),
            "device_timestamp_us": int(device_timestamp),
            "camera_label": self.config.camera_label,
            "intrinsics": {
                "fx": float(fx),
                "fy": float(fy),
                "cx": float(cx),
                "cy": float(cy),
                "K": K_aria.tolist(),
            },
            "destination_intrinsics": {
                "fx": float(dfx),
                "fy": float(dfy),
                "cx": float(dcx),
                "cy": float(dcy),
                'K': K_destination.tolist(),
            },
            "destination_intrinsics_rotated": {
                "fx": float(rdfx),
                "fy": float(rdfy),
                "cx": float(rdcx),
                "cy": float(rdcy),
                "K": rotated_intrinsics.tolist(),
            },
            "extrinsics": {
                "R": np.asarray(R).tolist(),
                "t": np.asarray(t).tolist(),
                "T_world_cam": T_World_Cam.to_matrix().tolist(),
                "T_world_device": T_World_Device.to_matrix().tolist(),
                "T_device_cam": T_Device_Cam.to_matrix().tolist(),
            },
        }
        calib_path = calib_dir / f"calib_{int(frame_number):06d}.json"
        with open(calib_path, "w") as f:
            json.dump(calib_dict, f, indent=2)

        hand21_dict = self._query_hand_21_world(device_timestamp, T_world_device_matrix)
        # hand_information = None
        # if wrist_and_palm_pose is not None:
        #     hand_information = {
        #         "left_wrist": wrist_and_palm_pose.left_hand.wrist_position_device,
        #         "left_palm": wrist_and_palm_pose.left_hand.palm_position_device,
        #         "right_wrist": wrist_and_palm_pose.right_hand.wrist_position_device,
        #         "right_palm": wrist_and_palm_pose.right_hand.palm_position_device,
        #         "left_wrist_normal": wrist_and_palm_pose.left_hand.wrist_and_palm_normal_device.wrist_normal_device,
        #         "left_palm_normal": wrist_and_palm_pose.left_hand.wrist_and_palm_normal_device.palm_normal_device,
        #         "right_wrist_normal": wrist_and_palm_pose.right_hand.wrist_and_palm_normal_device.wrist_normal_device,
        #         "right_palm_normal": wrist_and_palm_pose.right_hand.wrist_and_palm_normal_device.palm_normal_device,
        #     }

        #     # Transform to world coordinate system
        #     for key in hand_information.keys():
        #         if hand_information[key] is not None:
        #             hand_information[key] = (T_world_device @ np.r_[hand_information[key], [1]])[:3]

        wpp = wrist_and_palm_pose
        left = getattr(wpp, "left_hand", None)
        right = getattr(wpp, "right_hand", None)

        hand_information = {
            "left_wrist": getattr(left, "wrist_position_device", None),
            "left_palm": getattr(left, "palm_position_device", None),
            "right_wrist": getattr(right, "wrist_position_device", None),
            "right_palm": getattr(right, "palm_position_device", None),
            "left_wrist_normal": getattr(getattr(left, "wrist_and_palm_normal_device", None), "wrist_normal_device", None) if left else None,
            "left_palm_normal": getattr(getattr(left, "wrist_and_palm_normal_device", None), "palm_normal_device", None) if left else None,
            "right_wrist_normal": getattr(getattr(right, "wrist_and_palm_normal_device", None), "wrist_normal_device", None) if right else None,
            "right_palm_normal": getattr(getattr(right, "wrist_and_palm_normal_device", None), "palm_normal_device", None) if right else None,
        }

        # Transform positions with full SE(3), normals with rotation only
        R_wd = T_world_device_matrix[:3, :3]
        for key, val in list(hand_information.items()):
            if val is None:
                continue
            v = np.asarray(val, dtype=np.float32).reshape(3)
            if key.endswith("_normal"):
                hand_information[key] = (R_wd @ v).astype(np.float32)
            else:
                hand_information[key] = (T_world_device_matrix @ np.r_[v, 1.0])[:3].astype(np.float32)

        # Merge 21-landmark result (already in world)
        hand_information.update(hand21_dict)

        raw_image = image_data[0].to_numpy_array()

        # cv2.circle(raw_image, pos.astype(np.int32), 10, (255, 0, 0), -1)

        processed_image = self._transform_image(
            raw_image,
            devignetting_mask,
            dst_calib,
            src_calib,
        )

        output_path = output_dir / f"color_{int(frame_number):06d}_camera07.jpg"
        Image.fromarray(processed_image).save(output_path)
        Image.fromarray(raw_image).save(str(output_dir).replace("color", "raw") + f"/raw_{int(frame_number):06d}_camera07.jpg")
        if hand_information is not None:
            np.save(f"../data/{self.config.recording_name}/Aria/export/hand/hand_{int(frame_number):06d}.npy", hand_information)

    def _transform_image(
        self,
        raw_image: np.ndarray,
        devignetting_mask: np.ndarray,
        dst_calib: calibration.CameraCalibration,
        src_calib: calibration.CameraCalibration,
    ) -> np.ndarray:
        """
        Apply transformation pipeline to the raw image.

        Args:
            raw_image: Raw input image
            devignetting_mask: Devignetting mask for image correction
            dst_calib: Destination camera calibration
            src_calib: Source camera calibration

        Returns:
            Transformed image array
        """
        # Apply devignetting correction
        corrected_image = calibration.devignetting(raw_image, devignetting_mask)

        # Apply distortion correction
        undistorted_image = calibration.distort_by_calibration(
            corrected_image, dst_calib, src_calib, InterpolationMethod.BILINEAR
        )

        # Rotate image
        return np.rot90(undistorted_image, k=3)


def export_scene_point_cloud(processor: "AriaFrameProcessor", out_path: Path,
                             voxel_size: float = 0.01,  # meters
                             nb_neighbors: int = 20, std_ratio: float = 2.0):
    """
    Export the MPS semi-dense world-space point cloud to a .ply file.
    Applies optional denoising and voxel downsampling using your utilities.
    """
    # 1) raw (N,3) points in world frame
    pts = processor.point_cloud  # already filtered by confidence in _get_point_cloud()

    # 2) (optional) denoise & voxelize with your helpers
    try:
        # these are from: from point_cloud import denoise_point_cloud, voxelize_point_cloud
        pts = denoise_point_cloud(pts, nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        pts = voxelize_point_cloud(pts, voxel_size=voxel_size)
    except Exception as _:
        # If your helpers expect Open3D geometries, skip gracefully and just save raw points
        pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_ply_xyz(out_path, pts)
    print(f"Saved point cloud: {out_path}  (#pts={len(pts)})")



def main():
    """Main entry point for the frame extraction process."""
    RECORDING = "20250206_Testing"
    config = ProcessingConfig(
        # aria_name="b0bb3408-2f28-46d0-bf48-ad2892708a1c",
        # aria_name="929c8945-8451-4134-9de7-e483f2aa8f54",
        aria_name="20250206_Testing",
        recording_name=RECORDING,
        base_dir=Path("../data") / RECORDING / "Aria",
        devignetting_mask_path=Path("../data/aria_devignetting_masks"),
    )

    output_dirs = {
        "camera-rgb": config.base_dir / "export" / "color",
        "raw": config.base_dir / "export" / "raw",
        "hand": config.base_dir / "export" / "hand",
        "calib": config.base_dir / "export" / "calib",
    }

    try:
        processor = AriaFrameProcessor(config)
        export_scene_point_cloud(
            processor,
            out_path=config.base_dir / "export" / "pointcloud" / f"{config.aria_name}_semidense_world.ply",
            voxel_size=0.01,
        )
        processor.process_frames(output_dirs)
    except Exception as e:
        print(f"Error processing frames: {str(e)}")
        raise


if __name__ == "__main__":
    main()
