import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from ament_index_python.packages import get_package_share_directory
import os

from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import TransformStamped

from cv_bridge import CvBridge
import cv2

import numpy as np
import yaml
import tf2_ros

#from tf_transformations import euler_from_quaternion, quaternion_from_euler
from transforms3d.euler import euler2quat, quat2euler
from scipy.spatial.transform import Rotation as R_scipy

class EnvCamCalibNode(Node):
    def __init__(self):
        super().__init__('board_calib')

        # Load camera intrinsic params
        self.camera_matrix = None
        self.dist_coeffs = None
        self.got_camera_info = False

        self.sub_info = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)

        # Load charuco params
        # Load charuco params via parameter (fallback to packaged default)
        default_charuco_path = os.path.join(
            get_package_share_directory('board_calib'),
            'config',
            'charuco_board.yaml'
        )
        charuco_yaml_path = self.declare_parameter(
            'charuco_board_file',
            default_charuco_path
        ).get_parameter_value().string_value

        with open(charuco_yaml_path, 'r') as f:
            self.charuco_cfg = yaml.safe_load(f)['charuco']


        self.dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, self.charuco_cfg['dictionary']))
        self.board = cv2.aruco.CharucoBoard_create(
            squaresX=self.charuco_cfg['squares_x'],
            squaresY=self.charuco_cfg['squares_y'],
            squareLength=self.charuco_cfg['square_length'],
            markerLength=self.charuco_cfg['marker_length'],
            dictionary=self.dictionary)

        self.bridge = CvBridge()
        self.sub_image = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)

        # Storage for poses
        self.robot_poses = []  # List of robot base -> board poses (known)
        self.cam_poses = []    # List of camera -> board poses (from detection)

        # tf listener to get robot pose at capture time
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Variables to hold calibration result
        self.latest_R = None
        self.latest_t = None

        self.calibrated_tf_pub = self.create_timer(0.1, self.publish_calibrated_tf_timer)

        self.calibrated = False

    def image_callback(self, msg):
        if self.calibrated:
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        corners, ids, _ = cv2.aruco.detectMarkers(cv_image, self.dictionary)
        if ids is not None and len(ids) > 0:
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=cv_image,
                board=self.board)

            if charuco_corners is not None and len(charuco_corners) > 3:
                # Get pose of board wrt camera
                
                if not self.got_camera_info:
                    self.get_logger().warn('Camera intrinsics not yet received')
                    return
                else:
                    self.get_logger().info(f'Camera matrix:\n{self.camera_matrix}')
                    self.get_logger().info(f'Distortion coefficients:\n{self.dist_coeffs}')
                    self.get_logger().info('---------------------------------------------------------------------------------------')

                ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, self.board,
                self.camera_matrix, self.dist_coeffs, None, None)

                if ret:
                    # Save camera->board pose
                    cam_R, _ = cv2.Rodrigues(rvec)
                    cam_t = tvec.reshape(3)

                    # Lookup robot base->board transform from tf or config
                    try:
                        stamp = msg.header.stamp
                        t = self.tf_buffer.lookup_transform(
                            'base_link',
                            'charuco_board_link',
                            stamp,
                            timeout=Duration(seconds=0.5)
                        )
                        
                        # Extract translation and rotation as numpy arrays
                        trans = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
                        rot = np.array([t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w])

                        # Convert rot quaternion to rotation matrix
                        # Store these poses for hand-eye calibration
                        self.robot_poses.append((rot, trans))
                        self.cam_poses.append((rvec, tvec))

                        if len(self.robot_poses) > 10:
                            self.perform_hand_eye_calib()
                            self.calibrated = True
                    except Exception as e:
                        self.get_logger().warn(f'TF lookup failed: {e}')

    def camera_info_callback(self, msg):
        if self.got_camera_info:
            return

        K = np.array(msg.k).reshape(3, 3)
        D = np.array(msg.d)

        self.camera_matrix = K
        self.dist_coeffs = D
        self.got_camera_info = True

        self.get_logger().info('Received camera intrinsics from /camera_info topic')

    def perform_hand_eye_calib(self):
        # Convert to rotation matrices and translations
        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []

        for (r_quat, t_vec), (rvec, tvec) in zip(self.robot_poses, self.cam_poses):
            # robot pose rotation matrix and translation
            Rr = quat_to_rot(r_quat)
            R_gripper2base.append(Rr)
            t_gripper2base.append(t_vec)

            # camera pose rotation matrix and translation
            R_cam, _ = cv2.Rodrigues(rvec)
            R_target2cam.append(R_cam)
            t_target2cam.append(tvec.reshape(3))

        # Use OpenCV calibrateHandEye
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base, R_target2cam, t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI)

        self.get_logger().info(f'Camera pose wrt robot base:\nRotation:\n{R_cam2gripper}\nTranslation:\n{t_cam2gripper}')

        save_calibration(R_cam2gripper, t_cam2gripper)
        self.get_logger().info('Calibration saved to calibration_values.yaml')

        self.latest_R = R_cam2gripper
        self.latest_t = t_cam2gripper

    def publish_calibrated_tf_timer(self):
        if not self.calibrated or self.latest_R is None or self.latest_t is None:
            return

        # Convert rotation matrix to quaternion
        rot = R_scipy.from_matrix(self.latest_R)
        q = rot.as_quat()  # [x, y, z, w]

        t_msg = TransformStamped()
        t_msg.header.stamp = self.get_clock().now().to_msg()
        t_msg.header.frame_id = 'base_link'
        t_msg.child_frame_id = 'camera_link_calibrated'

        t_msg.transform.translation.x = float(self.latest_t[0])
        t_msg.transform.translation.y = float(self.latest_t[1])
        t_msg.transform.translation.z = float(self.latest_t[2])
        t_msg.transform.rotation.x = float(q[0])
        t_msg.transform.rotation.y = float(q[1])
        t_msg.transform.rotation.z = float(q[2])
        t_msg.transform.rotation.w = float(q[3])

        self.tf_broadcaster.sendTransform(t_msg)

        self.get_logger().info("======= Calibrated Transform =======")
        self.get_logger().info(f"Translation (t): {self.latest_t.ravel()}")
        self.get_logger().info(f"Rotation matrix (R):\n{self.latest_R}")
        self.get_logger().info(f"Quaternion (x,y,z,w): {q}\n")



def quat_to_rot(q):
    """Convert [x, y, z, w] quaternion to 3x3 rotation matrix"""
    return R_scipy.from_quat(q).as_matrix()

def save_calibration(R, t, filename='calibration_values.yaml'):

    # Convert rotation matrix to roll-pitch-yaw (in radians)
    rpy = R_scipy.from_matrix(R).as_euler('xyz', degrees=False)

    # Flatten translation
    xyz = t.flatten()

    # YAML content
    data = {
        'camera_transform': {
            'xyz': [float(xyz[0]), float(xyz[1]), float(xyz[2])],
            'rpy': [float(rpy[0]), float(rpy[1]), float(rpy[2])],     
            'base_frame': "base_link",
            'camera_frame': "camera_link"
        },
    }

    with open(filename, 'w') as f:
        yaml.dump(data, f)

def main(args=None):
    rclpy.init(args=args)
    node = EnvCamCalibNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()