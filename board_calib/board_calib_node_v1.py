import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import TransformStamped

from ament_index_python.packages import get_package_share_directory

from cv_bridge import CvBridge
import cv2

import os
import numpy as np
import yaml
import tf2_ros

#from tf_transformations import euler_from_quaternion, quaternion_from_euler
from transforms3d.euler import euler2quat, quat2euler
from scipy.spatial.transform import Rotation as R_scipy

class EnvCamCalibNode(Node):
    def __init__(self):
        super().__init__('board_calib')

        print("="*10 + " Calibration Node Constructor Execution Started " + "="*10)

        # OpenCV bridge
        self.bridge = CvBridge()

        # Variables for camera intrinsic parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        self.got_camera_info = False

        # Subscribers
        self.sub_info = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)
        self.sub_image = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)

        # Publishers
        self.image_pub = self.create_publisher(Image, 'calibration/annotated_image', 10)

        # Load charuco params
        default_board_config_path = os.path.join(get_package_share_directory('board_calib'),'config','charuco_board.yaml')
        board_config_path = self.declare_parameter('charuco_board_file',default_board_config_path).get_parameter_value().string_value
        with open(board_config_path, 'r') as f:
            self.charuco_board_config = yaml.safe_load(f)['charuco']
            print("Loaded calibration board config from : ", board_config_path)
            print("Board Config : \n", self.charuco_board_config)

        self.dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.charuco_board_config['dictionary']))
        
        # This is for some other version of OpenCV
        # self.board = cv2.aruco.CharucoBoard_create(
        #     squaresX=self.charuco_board_config['squares_x'],
        #     squaresY=self.charuco_board_config['squares_y'],
        #     squareLength=self.charuco_board_config['square_length'],
        #     markerLength=self.charuco_board_config['marker_length'],
        #     dictionary=self.dictionary)
        
        self.board = cv2.aruco.CharucoBoard(
            (self.charuco_board_config['squares_x'], self.charuco_board_config['squares_y']),
            self.charuco_board_config['square_length'],
            self.charuco_board_config['marker_length'],
            self.dictionary)

        # Board origin
        # The origin of the board is always at the bottom-left corner
        # The x-axis increases to the right when looking at the board from front (columns)
        # The y-axis increases upwards (rows)
        # The z-axis comes out of the board plane (towards the camera facing it)
        # The front face of the board (i.e., the marker side you observe) faces in the +Z direction of the board frame
        # So, when a camera detects the board head-on, it's located somewhere on the +Z side of the board, looking toward the XY plane of the board.

        # Storage for poses
        self.robot_poses = []  # List of robot base -> board poses (known)
        self.cam_poses = []    # List of camera -> board poses (from detection)

        # tf listener to get robot pose at capture time
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # tf broadcaster for publishing the estimated transform
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Variables to hold calibration result
        self.latest_R = None
        self.latest_t = None

        # Publisher for computed transform
        self.calibrated_tf_pub = self.create_timer(0.1, self.publish_calibrated_tf_timer)

        # Calibration confirmation
        self.calibrated = False

    def image_callback(self, msg):
        # if self.calibrated:
        #     return

        print("="*15 + " Image Callback Execution Started " + "="*15)

        input_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray_image, self.dictionary)

        if marker_corners is not None and len(marker_corners) > 0:
            # Marker Corners Detected!
            cv2.aruco.drawDetectedMarkers(input_image, marker_corners, marker_ids)

        if marker_ids is not None and len(marker_ids) > 0:
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                markerCorners=marker_corners,
                markerIds=marker_ids,
                image=gray_image,
                board=self.board)

            if charuco_corners is not None and len(charuco_corners) > 3:                
                
                # Get pose of board wrt camera
                if not self.got_camera_info:
                    print("Camera intrinsics not yet received")
                    return

                ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    charuco_corners,
                    charuco_ids,
                    self.board,
                    self.camera_matrix,
                    self.dist_coeffs,
                    None,
                    None
                )

                if ret:

                    cv2.aruco.drawDetectedCornersCharuco(input_image, charuco_corners, charuco_ids)
                    for i in range(len(charuco_ids)):
                        corner = charuco_corners[i][0]
                        id_text = str(charuco_ids[i][0])
                        cv2.putText(input_image, id_text,
                                    (int(corner[0]), int(corner[1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    
                    # Publish annotated image
                    annotated_msg = self.bridge.cv2_to_imgmsg(input_image, encoding='bgr8')
                    annotated_msg.header = msg.header
                    self.image_pub.publish(annotated_msg)

                    # Taking multiple readings here
                    # Save camera->board pose
                    # cam_R, _ = cv2.Rodrigues(rvec)
                    # cam_t = tvec.reshape(3)

                    # Lookup for robot base->board transform from tf or config
                    # try:
                    #     stamp = msg.header.stamp
                    #     t = self.tf_buffer.lookup_transform(
                    #         'base_link',
                    #         'charuco_board_link',
                    #         stamp,
                    #         timeout=Duration(seconds=0.5)
                    #     )
                        
                    #     # Extract translation and rotation as numpy arrays
                    #     trans = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
                    #     rot = np.array([t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w])

                    #     # Store these poses for hand-eye calibration
                    #     self.robot_poses.append((rot, trans))
                    #     self.cam_poses.append((rvec, tvec))

                    #     if len(self.robot_poses) > 10:
                    #         self.perform_hand_eye_calib()
                    #         self.calibrated = True
                    # except Exception as e:
                    #     self.get_logger().warn(f'TF lookup failed: {e}')

                    if self.calibrated:
                        return

                    try:
                        stamp = msg.header.stamp
                        t = self.tf_buffer.lookup_transform(
                            'charuco_board_link',
                            'base_link',
                            stamp,
                            timeout=Duration(seconds=0.5)
                        )

                        # Transform from base_link → board by inverting (board → base_link)
                        trans_board2base = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
                        quat_board2base = np.array([t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w])

                        R_board2base = quat_to_rot(quat_board2base)
                        R_base2board = R_board2base.T
                        t_base2board = -R_board2base.T @ trans_board2base

                        # Save this robot pose (base_link → board)
                        self.robot_poses.append((R_base2board, t_base2board))

                        # Save camera pose (camera → board), will be inverted later
                        self.cam_poses.append((rvec, tvec))

                        if len(self.robot_poses) > 15:
                            self.perform_hand_eye_calib()
                            self.calibrated = True
                        
                    except Exception as e:
                        print(f'TF lookup failed: {e}')


    def camera_info_callback(self, msg):
        print("="*15 + " Camera Info Callback Execution Started " + "="*15)
        if self.got_camera_info:
            return

        K = np.array(msg.k).reshape(3, 3)
        D = np.array(msg.d)

        self.camera_matrix = K
        self.dist_coeffs = D
        self.got_camera_info = True

        print('Received camera intrinsics from /camera_info topic')
        print(f'Camera matrix:\n{self.camera_matrix}')
        print(f'Distortion coefficients:\n{self.dist_coeffs}')
        print("-"*30)

    
    def perform_hand_eye_calib_v1(self):

        print("="*10 + " Calibration Function Execution Started " + "="*10)
        
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

        print(f'')

        # Use OpenCV calibrateHandEye
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base, R_target2cam, t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI)

        print(f'Camera Base Link Pose wrt Robot Base:\nRotation:\n{R_cam2gripper}\nTranslation:\n{t_cam2gripper}')

        save_calibration(R_cam2gripper, t_cam2gripper)
        print('Calibration saved to calibration_values.yaml')

        self.latest_R = R_cam2gripper
        self.latest_t = t_cam2gripper
    
    def perform_hand_eye_calib_v2(self):
        print("="*10 + " Calibration Function Execution Started " + "="*10)

        R_gripper2base = []  # base_link → board
        t_gripper2base = []

        R_target2cam = []    # board → camera (after inversion)
        t_target2cam = []

        for (R_base2board, t_base2board), (rvec, tvec) in zip(self.robot_poses, self.cam_poses):
            # Append robot side (already in correct direction)
            R_gripper2base.append(R_base2board)
            t_gripper2base.append(t_base2board)

            # Invert camera → board to get board → camera
            R_cam2board, _ = cv2.Rodrigues(rvec)
            t_cam2board = tvec.reshape(3)

            R_board2cam = R_cam2board.T
            t_board2cam = -R_cam2board.T @ t_cam2board

            R_target2cam.append(R_board2cam)
            t_target2cam.append(t_board2cam)

        # Solve AX = XB
        R_cam2base, t_cam2base = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base,
            R_target2cam, t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI
        )

        print(f'Camera Pose wrt Base Link:\nR:\n{R_cam2base}\nt:\n{t_cam2base}')

        # Save + store
        save_calibration(R_cam2base, t_cam2base)
        print('Calibration saved to calibration_values.yaml')

        self.latest_R = R_cam2base
        self.latest_t = t_cam2base
    

    def perform_hand_eye_calib(self):
        print("="*10 + " Static Calibration Execution Started " + "="*10)

        # Use only the most recent single observation
        if not self.robot_poses or not self.cam_poses:
            self.get_logger().error("No pose data available for calibration.")
            return

        # Get the only (or latest) pose pair
        (R_base_to_board, t_base_to_board) = self.robot_poses[-1]
        (rvec, tvec) = self.cam_poses[-1]

        # From Charuco detection
        R_camera_to_board, _ = cv2.Rodrigues(rvec)
        t_camera_to_board = tvec.reshape(3)

        # Compute transform: base_link → camera_link
        R_base_to_camera, t_base_to_camera = compute_static_base_to_camera(
            (R_base_to_board, t_base_to_board),
            (R_camera_to_board, t_camera_to_board)
        )

        print(f"Static Camera Pose wrt Base Link:\nRotation:\n{R_base_to_camera}\nTranslation:\n{t_base_to_camera}")

        # Save to file
        save_calibration(R_base_to_camera, t_base_to_camera)
        print('Calibration saved to calibration_values.yaml')

        # Store result for TF broadcasting
        self.latest_R = R_base_to_camera
        self.latest_t = t_base_to_camera


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

        print("="*10 + " Calibrated Transform " + "="*10)
        print(f"Translation (t): {self.latest_t.ravel()}")
        print(f"Rotation matrix (R):\n{self.latest_R}")
        print(f"Quaternion (x,y,z,w): {q}\n")


def compute_static_base_to_camera(T_base_to_board, T_camera_to_board):
    """
    Given:
    - base_link → board
    - camera → board

    Computes:
    - base_link → camera_link

    Args:
    - T_base_to_board: (R, t)
    - T_camera_to_board: (R, t)

    Returns:
    - R, t for base_link → camera_link
    """
    R_bb, t_bb = T_base_to_board
    R_cb, t_cb = T_camera_to_board

    # Invert camera → board to get board → camera
    R_bc = R_cb.T
    t_bc = -R_cb.T @ t_cb

    # base → camera = base → board * board → camera
    R_bc_final = R_bb @ R_bc
    t_bc_final = R_bb @ t_bc + t_bb

    return R_bc_final, t_bc_final

def quat_to_rot(q):
    """Convert [x, y, z, w] quaternion to 3x3 rotation matrix"""
    return R_scipy.from_quat(q).as_matrix()

def save_calibration(R, t, filename='src/board_calib/config/calibration_values_estimated.yaml'):

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