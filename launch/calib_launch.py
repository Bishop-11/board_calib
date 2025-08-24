from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import yaml
import xacro

def generate_launch_description():
    
    print("="*20 + " Launch File Execution Started " + "="*20)

    package_dir = get_package_share_directory('board_calib')
    print("Package Directory = ", package_dir)

    # Load calibration values
    calib_file = os.path.join(package_dir, 'config', 'calibration_values.yaml')
    print("Calibration Values File = ", calib_file)
    with open(calib_file, 'r') as f:
        calib_data = yaml.safe_load(f)

    xyz = calib_data['camera_transform']['xyz']
    rpy = calib_data['camera_transform']['rpy']
    base_frame = calib_data['camera_transform']['base_frame']
    camera_frame = calib_data['camera_transform']['camera_frame']

    print("="*10 + " Imported Initial Values " + "="*10)
    print("XYZ = ", xyz)
    print("RPY = ", rpy)
    print("Base Frame = ", base_frame)
    print("Camera Frame = ", camera_frame)

    # Process xacro with values
    urdf_path = os.path.join(package_dir, 'urdf', 'camera_setup.urdf.xacro')
    print("Using URDF File = ", urdf_path)
    doc = xacro.process_file(urdf_path, mappings={
        'camera_x': str(xyz[0]),
        'camera_y': str(xyz[1]),
        'camera_z': str(xyz[2]),
        'camera_roll': str(rpy[0]),
        'camera_pitch': str(rpy[1]),
        'camera_yaw': str(rpy[2]),
        'base_frame': str(base_frame),
        'camera_frame': str(camera_frame),
    })
    urdf_xml = doc.toxml()

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': urdf_xml}]
        ),
        Node(
            package='board_calib',
            executable='board_calib_node',
            name='board_calib_node',
            output='screen',
            parameters=[
                {'camera_info_topic': '/camera/camera/color/camera_info'},
                {'image_topic': '/camera/camera/color/image_raw'},
                {'charuco_board_file': os.path.join(package_dir, 'config', 'charuco_board.yaml')}
            ]
        )
    ])
