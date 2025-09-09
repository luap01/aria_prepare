
from typing import Tuple
import open3d as o3d
import numpy as np

def denoise_point_cloud(
    point_cloud: o3d.geometry.PointCloud,
    nb_neighbors: int = 40,
    std_ratio: float = 1.0
) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """
    Remove noise from a point cloud using statistical outlier removal.
    
    This function identifies and removes outlier points that are further away from their 
    neighbors compared to the average points in the point cloud. It uses the statistical
    outlier removal method where points with average distances to their neighbors beyond 
    a threshold are considered outliers.
    
    Args:
        point_cloud: Input point cloud to be denoised
        nb_neighbors: Number of neighbors to analyze for each point (default: 40)
            Higher values consider more points in the statistical analysis
        std_ratio: Standard deviation ratio multiplier (default: 1.0)
            Points with average distances larger than (mean + std_ratio * std) of the 
            average distances to k-nearest neighbors are considered outliers
    
    Returns:
        Tuple containing:
            - Denoised point cloud (inlier points)
            - Noise point cloud (outlier points)
    
    Raises:
        ValueError: If point cloud is empty or if parameters are invalid
        
    Example:
        >>> cleaned_cloud, noise = denoise_point_cloud(raw_point_cloud, nb_neighbors=30)
    """
    if len(point_cloud.points) == 0:
        raise ValueError("Input point cloud is empty")
    if nb_neighbors <= 0:
        raise ValueError("Number of neighbors must be positive")
    if std_ratio <= 0:
        raise ValueError("Standard deviation ratio must be positive")
        
    # Perform statistical outlier removal
    cleaned_cloud, inlier_indices = point_cloud.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    
    # Extract noise points by inverting the selection
    noise_cloud = point_cloud.select_by_index(inlier_indices, invert=True)
    
    return cleaned_cloud, noise_cloud


def voxelize_point_cloud(
    point_cloud: o3d.geometry.PointCloud,
    voxel_size: float = 0.01
) -> o3d.geometry.PointCloud:
    """
    Downsample a point cloud using voxel grid filtering.
    
    This function reduces the density of the point cloud by creating a 3D voxel grid
    and replacing all points within each voxel with their centroid. This is useful
    for reducing computation time in subsequent processing steps while maintaining
    the overall geometry of the point cloud.
    
    Args:
        point_cloud: Input point cloud to be downsampled
        voxel_size: Size of each voxel (in same units as point cloud) (default: 0.01)
            Larger values result in more aggressive downsampling
    
    Returns:
        Downsampled point cloud
        
    Raises:
        ValueError: If point cloud is empty or if voxel size is invalid
        
    Note:
        - The voxel size should be chosen based on the scale of the point cloud
          and the desired level of detail
        - Smaller voxel sizes preserve more detail but result in less reduction
        - The function maintains color and normal information if present
        
    Example:
        >>> # Downsample with 1cm voxels
        >>> downsampled = voxelize_point_cloud(dense_cloud, voxel_size=0.01)
    """
    if len(point_cloud.points) == 0:
        raise ValueError("Input point cloud is empty")
    if voxel_size <= 0:
        raise ValueError("Voxel size must be positive")
        
    # Calculate appropriate voxel size if point cloud units are in millimeters
    if np.mean(np.asarray(point_cloud.points)) > 100:
        print("Warning: Large point values detected. Consider converting to meters if "
              "point cloud is in millimeters.")
    
    # Perform voxel downsampling
    downsampled_cloud = point_cloud.voxel_down_sample(voxel_size)
    
    # Verify downsampling result
    if len(downsampled_cloud.points) == 0:
        raise ValueError(
            f"Voxel size {voxel_size} is too large and resulted in empty point cloud"
        )
    
    return downsampled_cloud