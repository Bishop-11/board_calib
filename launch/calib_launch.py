from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import yaml
import xacro

def generate_launch_description():
    
    print("="*20 + " Launch File - Execution Started " + "="*20)

    package_dir = get_package_share_directory('board_calib')
    print("Package Directory = ", package_dir)

    # Process xacro with values
    urdf_path = os.path.join(package_dir, 'urdf', 'camera_setup.urdf.xacro')
    print("Using URDF File = ", urdf_path)
    urdf_doc = xacro.process_file(urdf_path)
    urdf_xml = urdf_doc.toxml()

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='calibration_state_publisher',
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
