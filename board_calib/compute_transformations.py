import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

def rpy_to_matrix(translation, rpy):
    """Create 4x4 homogeneous transform from translation and rpy."""
    rotation_matrix = R_scipy.from_euler('xyz', rpy).as_matrix()
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = translation
    return matrix

def print_transform(name, matrix):
    """Print translation and RPY from transformation matrix."""
    translation = matrix[:3, 3]
    rpy = R_scipy.from_matrix(matrix[:3, :3]).as_euler('xyz')
    print(f"{name}:\n  Translation: {translation}\n  RPY (rad): {rpy}\n")

# ------------------------
# Define known transforms
# ------------------------

# map -> robot_base
T_map_robot = rpy_to_matrix([0.10, 0.0, 0.0], [0.0, 0.0, 0.0])

# map -> charuco
T_map_charuco = rpy_to_matrix([0.12, 0.22, 0.05], [0.0, 0.0, 0.0])

# image -> charuco (from OpenCV)
T_image_charuco = rpy_to_matrix([-0.0647, 0.0898, 0.3203], [3.226, -0.021, 0.016])

# camera_base -> image (from Realsense ROS)
T_camera_image = rpy_to_matrix([0.0, 0.01476, 0.0], [-1.570, -0.001, -1.570])

# ---------------------------------------
# Compute camera_base in map coordinates
# ---------------------------------------

# Invert image -> charuco to get charuco -> image
T_charuco_image = np.linalg.inv(T_image_charuco)

# Invert camera_base -> image to get image -> camera_base
T_image_camera = np.linalg.inv(T_camera_image)

# Compose:
# T_map_camera = T_map_charuco * T_charuco_image * T_image_camera
T_map_camera = T_map_charuco @ T_charuco_image @ T_image_camera

# --------------------------------------------------
# Compute camera_base in robot_base coordinates
# --------------------------------------------------

# Invert map -> robot to get robot -> map
T_robot_map = np.linalg.inv(T_map_robot)

# T_robot_camera = T_robot_map * T_map_camera
T_robot_camera = T_robot_map @ T_map_camera

# ----------------------------
# Print the resulting transforms
# ----------------------------

print_transform("Map -> Camera Base", T_map_camera)
print_transform("Robot Base -> Camera Base", T_robot_camera)