import numpy as np
from pathlib import Path
import json
from scipy.spatial.transform import Rotation


def extract_intrinsics_matrix(intrinsics_json: dict) -> np.ndarray:
    """
    Convert a camera intrinsics dictionary to a 3x3 camera intrinsics matrix.

    Parameters
    ----------
    intrinsics_json : dict
        Dictionary containing camera intrinsics parameters with keys 'm00' through 'm22'
        representing elements of the 3x3 intrinsics matrix.

    Returns
    -------
    np.ndarray
        A 3x3 camera intrinsics matrix containing focal lengths and principal point offsets:
        [[fx  s  cx]
         [0   fy cy]
         [0   0   1]]
    """
    return np.asarray([[intrinsics_json['m00'], intrinsics_json['m10'], intrinsics_json['m20']],
                    [intrinsics_json['m01'], intrinsics_json['m11'], intrinsics_json['m21']],
                    [intrinsics_json['m02'], intrinsics_json['m12'], intrinsics_json['m22']]])


def load_rotation_matrix(rot: dict) -> np.ndarray:
    """
    Convert a quaternion rotation dictionary to a 3x3 rotation matrix.

    Parameters
    ----------
    rot : dict
        Dictionary containing quaternion rotation parameters with keys 'x', 'y', 'z', 'w'
        representing the quaternion components.

    Returns
    -------
    np.ndarray
        A 3x3 rotation matrix representing the same rotation as the input quaternion.

    Examples
    --------
    >>> rot = {'x': 0, 'y': 0, 'z': 0, 'w': 1}  # Identity quaternion
    >>> load_rotation_matrix(rot)
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    """
    return Rotation.from_quat([rot['x'], rot['y'], rot['z'], rot['w']]).as_matrix()


def load_transform_matrix(trans: dict, rot: dict) -> np.ndarray:
    """
    Create a 4x4 homogeneous transformation matrix from translation and rotation components.

    Parameters
    ----------
    trans : dict
        Dictionary containing translation parameters with keys 'm00', 'm10', 'm20'
        representing x, y, z translations respectively.
    rot : dict
        Dictionary containing quaternion rotation parameters with keys 'x', 'y', 'z', 'w'.

    Returns
    -------
    np.ndarray
        A 4x4 homogeneous transformation matrix combining the rotation and translation.
        The matrix has the form:
        [R R R Tx]
        [R R R Ty]
        [R R R Tz]
        [0 0 0  1]
        where R represents rotation components and T represents translation components.
    """
    transform = np.zeros((4, 4), dtype=np.float32)
    transform[:3, :3] = load_rotation_matrix(rot)
    transform[:, 3] = [trans['m00'], trans['m10'], trans['m20'], 1]
    return transform


def rotation_to_homogenous(vec):
    """
    Convert a rotation vector to a 4x4 homogeneous transformation matrix.

    Parameters
    ----------
    vec : np.ndarray
        A 3D rotation vector specifying rotation axis and magnitude.

    Returns
    -------
    np.ndarray
        A 4x4 homogeneous transformation matrix representing the rotation:
        [[R R R 0]
         [R R R 0]
         [R R R 0]
         [0 0 0 1]]
        where R represents the rotation components.
    """
    rot_mat = Rotation.from_rotvec(vec)
    swap = np.identity(4)
    swap = np.zeros((4, 4))
    swap[:3, :3] = rot_mat.as_matrix()
    swap[3, 3] = 1
    return swap


def load_cam_infos(camera_path: Path, orbbec: bool = True) -> dict:
    """
    Load and process camera calibration information from JSON files in a directory.

    Parameters
    ----------
    take_path : Path
        Path to the directory containing a 'calibration' subdirectory with camera
        calibration JSON files named 'camera*.json'.
    orbbec : bool
        If True, load first 4 cameras (Orbbec), if False load remaining cameras.

    Returns
    -------
    dict
        Dictionary containing camera parameters for each camera, with keys 'camera0X' where
        X is the camera number. Each camera's dictionary contains:
        - intrinsics: 3x3 camera intrinsics matrix
        - extrinsics: 4x4 camera extrinsics matrix
        - fov_x: horizontal field of view
        - fov_y: vertical field of view
        - c_x: principal point x-coordinate
        - c_y: principal point y-coordinate
        - width: image width in pixels
        - height: image height in pixels
        - radial_params: radial distortion parameters
        - tangential_params: tangential distortion parameters
        - depth_extrinsics: 4x4 depth camera extrinsics matrix

    Notes
    -----
    The function applies several coordinate transformations including YZ flip and swap
    to align with a specific coordinate system convention.
    """
    with open(camera_path, 'r') as f:
        cam_info = json.load(f)

    if 'camera07' in str(camera_path):
        t = [-0.08265147,  0.45190677,  0.2571405 ] if "20250206" in str(camera_path) else [0.07087158, 0.39764744, 0.20956215]
        t0 = t[0]
        t1 = t[1]
        t2 = t[2]
        T_orbbec_world_aria = np.array(
            [[-0.39774132, 0.91749084, -0.00351968,  t0 ],
            [ 0.02202938,  0.01338487,  0.9996677,   t1 ],
            [ 0.9172331,   0.39753163, -0.02553547,  t2 ],
            [ 0.,          0.,          0.,          1.        ]], dtype=np.float32)
        
        T_aria_orbbec = np.linalg.inv(T_orbbec_world_aria)
        T_world_cam_aria = np.array(cam_info["extrinsics"]["T_world_cam"], dtype=np.float32)  # 4x4
        T_cam_world_aria = np.linalg.inv(T_world_cam_aria)
        extrinsics = T_cam_world_aria @ T_aria_orbbec
        return {
            'intrinsics': np.array(cam_info['destination_intrinsics_rotated']['K'], dtype=np.float32),
            'extrinsics': np.linalg.inv(extrinsics),
            'fov_x': cam_info['destination_intrinsics_rotated']['fx'],
            'fov_y': cam_info['destination_intrinsics_rotated']['fy'],
            'c_x': cam_info['destination_intrinsics_rotated']['cx'],
            'c_y': cam_info['destination_intrinsics_rotated']['cy'],
            'width': 1024,
            'height': 1024,
            'radial_params': None,
            'tangential_params': None,
            'depth_extrinsics': None,
            'rotation': None
        }

    cam_info = cam_info['value0']
    params = cam_info['color_parameters']
    intrinsics = extract_intrinsics_matrix(params['intrinsics_matrix'])
    intrinsics[2, 1] = 0
    intrinsics[2, 2] = 1
    extrinsics = load_transform_matrix(cam_info['camera_pose']['translation'], cam_info['camera_pose']['rotation'])

    if 'color2depth_transform' in cam_info:
        color2depth_transform = load_transform_matrix(cam_info['color2depth_transform']['translation'],
                                                    cam_info['color2depth_transform']['rotation'])

        depth_extrinsics = extrinsics.copy()
        extrinsics = np.matmul(extrinsics, color2depth_transform)

        YZ_FLIP = rotation_to_homogenous(np.pi * np.array([1, 0, 0]))
        YZ_SWAP = rotation_to_homogenous(np.pi/2 * np.array([1, 0, 0]))

        extrinsics = YZ_SWAP @ extrinsics @ YZ_FLIP
    else:
        T_marshall_orbbec = np.eye(4)
        T_marshall_orbbec[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
        extrinsics = np.linalg.inv(extrinsics)
        extrinsics = T_marshall_orbbec @ extrinsics
        depth_extrinsics = extrinsics

    radial_params = tuple(params['radial_distortion'].values())
    tangential_params = tuple(params['tangential_distortion'].values())

    camera_parameters = {
        'intrinsics': intrinsics,
        'extrinsics': extrinsics,
        'fov_x': params['fov_x'],
        'fov_y': params['fov_y'],
        'c_x': params['c_x'],
        'c_y': params['c_y'],
        'width': params['width'],
        'height': params['height'],
        'radial_params': radial_params,
        'tangential_params': tangential_params,
        'depth_extrinsics': depth_extrinsics,
        'rotation': cam_info['camera_pose']['rotation']
    }

    return camera_parameters


# Additional helper function to project 3D points to 2D using camera parameters
def project_to_2d(point_3d, intrinsic_matrix, extrinsic_matrix):
    """
    Project a 3D point to 2D image coordinates using camera parameters.

    Parameters
    ----------
    point_3d : np.ndarray
        3D point coordinates in world space (x, y, z).
    intrinsic_matrix : np.ndarray
        3x3 camera intrinsics matrix.
    extrinsic_matrix : np.ndarray
        4x4 camera extrinsics matrix (world to camera transform).

    Returns
    -------
    np.ndarray
        2D integer pixel coordinates (u, v) of the projected point in the image plane.

    Notes
    -----
    The projection pipeline:
    1. Convert 3D point to homogeneous coordinates
    2. Transform to camera space using extrinsic matrix
    3. Project to image plane using intrinsic matrix
    4. Normalize by depth (z-coordinate)
    5. Convert to integer pixel coordinates
    """
    # Convert the point to homogeneous coordinates
    point_3d_hom = np.append(point_3d, 1)
    # Apply extrinsic matrix
    point_cam = np.dot(extrinsic_matrix, point_3d_hom)
    # Apply intrinsic matrix
    point_img_hom = np.dot(intrinsic_matrix, point_cam[:3])
    # Normalize by the third (z) coordinate to get image coordinates
    point_img = point_img_hom[:2] / point_img_hom[2]
    return point_img.astype(np.int32)


# Additional helper function to calculate projection matrix for each camera view
def load_projection_matrix(cam_params: dict[str, dict]):
    """
    Calculate the projection matrix for a camera view.

    Parameters
    ----------
    cam_params : dict
        Dictionary containing camera parameters for each camera, with keys 'camera0X' where
        X is the camera number. Each camera's dictionary contains:
        - intrinsics: 3x3 camera intrinsics matrix
        - extrinsics: 4x4 camera extrinsics matrix

    Returns
    -------
    dict
        Dictionary containing projection matrices for each camera, with keys 'camera0X' where
        X is the camera number.
    """
    projection_matrices = {}
    for cam_id in cam_params.keys():
        # projection_matrices[cam_id] = np.matmul(cam_info['intrinsics'], cam_info['extrinsics'][:3, :])
        K = cam_params[cam_id]['intrinsics']  # 3x3 intrinsic matrix
        extrinsics = cam_params[cam_id]['extrinsics']  # 4x4 matrix
        
        # Extract R (3x3) and T (3x1) from extrinsics
        R = extrinsics[:3, :3]  # Top-left 3x3
        T = extrinsics[:3, 3]   # First 3 elements of last column
        RT = np.hstack((R, T.reshape(3, 1)))  # 3x4 [R | T]
        
        # Compute projection matrix P = K * [R | T]
        P = K @ RT
        projection_matrices[cam_id] = P
    return projection_matrices
