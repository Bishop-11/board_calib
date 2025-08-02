from setuptools import find_packages, setup

package_name = 'board_calib'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/charuco_board.yaml', 'config/calibration_values.yaml']),
        ('share/' + package_name + '/urdf', ['urdf/camera_setup.urdf.xacro', 'urdf/robot_description.urdf.xacro']),
        ('share/' + package_name + '/launch', ['launch/calib_launch.py']),
    ],
    install_requires=['setuptools', 'opencv-python', 'numpy', 'PyYAML'],
    zip_safe=True,
    maintainer='initial',
    maintainer_email='bishop_prakash@jp.honda',
    description='Simple camera calibration using charuco boards',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'board_calib_node = board_calib.board_calib_node:main',
        ],
    },
)
