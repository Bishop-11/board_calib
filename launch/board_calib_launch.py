from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    launch_description = LaunchDescription([
        Node(
            package='env_camera_calib',
            executable='calib_node',
            output='screen',
            parameters=[
                'config/charuco_board.yaml',
                {'camera_info_topic': '/env_camera/camera_info'},
                {'image_topic': '/env_camera/image_raw'}
            ]),
        ])
    
    return 