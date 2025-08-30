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

        print("="*10 + " Calibration Node Constructor - Execution Started " + "="*10)

        # OpenCV bridge
        self.bridge = CvBridge()

        # TF Link Names
        self.robot_base_link = 'base_link'
        self.calibration_board_link = 'charuco_board_link'
        self.camera_base_link = 'camera_link'
        self.camera_image_link = 'camera_color_optical_frame'
        self.camera_holder_link = 'link5'

        # Camera Type (Environment/Robot)
        self.camera_type = 'Environment'
        #self.camera_type = 'Robot'

        # Store all estimated transforms for saving and publishing
        self.estimated_transforms = {}

        # Variables for camera intrinsic parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        self.got_camera_info = False

        # Subscribers
        self.sub_info = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)
        self.sub_image = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        # Realsense D435 - '/camera/camera/color/image_raw'
        # Realsense D405 - '/camera/camera/color/image_rect_raw'

        # Publishers
        self.image_pub = self.create_publisher(Image, 'calibration/annotated_image', 10)

        # Load Charuco Board parameters
        default_board_config_path = os.path.join(get_package_share_directory('board_calib'),'config','charuco_board.yaml')
        board_config_path = self.declare_parameter('charuco_board_file',default_board_config_path).get_parameter_value().string_value
        with open(board_config_path, 'r') as f:
            self.charuco_board_config = yaml.safe_load(f)['charuco']
            print("Loaded calibration board config from : ", board_config_path)
            print("Board Config : \n", self.charuco_board_config)

        # Get predefined marker shapes
        self.dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.charuco_board_config['dictionary']))

        # This is for OpenCV 4.5.4
        self.board = cv2.aruco.CharucoBoard_create(
            squaresX=self.charuco_board_config['squares_x'],
            squaresY=self.charuco_board_config['squares_y'],
            squareLength=self.charuco_board_config['square_length'],
            markerLength=self.charuco_board_config['marker_length'],
            dictionary=self.dictionary)
        
        # This is for OpenCV 4.7.0
        # self.board = cv2.aruco.CharucoBoard(
        #     (self.charuco_board_config['squares_x'], self.charuco_board_config['squares_y']),
        #     self.charuco_board_config['square_length'],
        #     self.charuco_board_config['marker_length'],
        #     self.dictionary)

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

        # TF Tools
        # tf listener to get robot pose at capture time
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # tf broadcaster for publishing the estimated transform
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Variables to hold calibration result
        self.latest_R_base2camera = None
        self.latest_t_base2camera = None
        self.latest_R_holder2camera = None
        self.latest_t_holder2camera = None

        # Publisher for computed transform
        self.calibrated_tf_pub = self.create_timer(0.1, self.publish_calibrated_tf_timer)
        self.other_tf_pub = self.create_timer(0.1, self.publish_tf_timer)

        # Calibration confirmation
        self.calibrated = False
    
        print("="*10 + " Calibration Node Constructor - Execution Finished " + "="*10)


    def image_callback(self, msg):

        # Callback function for Image subscriber

        # Stop in already calibrated
        #if self.calibrated:
        #    return

        print("="*15 + " Image Callback - Execution Started " + "="*15)

        input_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        # Checkered pattern detection
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray_image, self.dictionary)

        if marker_corners is not None and len(marker_corners) > 0:
            # Marker Corners Detected!
            cv2.aruco.drawDetectedMarkers(input_image, marker_corners, marker_ids)

        if marker_ids is not None and len(marker_ids) > 0:
            # Charuco marker detection
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                markerCorners=marker_corners,
                markerIds=marker_ids,
                image=gray_image,
                board=self.board)

            if charuco_corners is not None and len(charuco_corners) > 3:
                # Charuco Markers Detected!
                
                # Get the pose of calibration board wrt camera image frame
                if not self.got_camera_info:
                    print("Camera intrinsics not yet received!")
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
                    # Pose was successfully estimated

                    # Annotate detection on original input image
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

                    
                    # Save camera pose (camera → board), will be inverted later
                    self.cam_poses.append((rvec, tvec))
                    
                    R_camera_image_to_board, _ = cv2.Rodrigues(rvec)
                    t_camera_image_to_board = tvec.reshape(3)

                    self.estimated_transforms["camera_image_to_board"] = [
                        self.camera_image_link,
                        'charuco_board_link_estimated1',
                        R_camera_image_to_board, 
                        t_camera_image_to_board]

                    if self.calibrated:
                        print("Calibration Completed - Exiting Callback ...")
                        return
                    
                    # Get tranform from Robot base frame to charuco board frame from TF Tree
                    try:
                        stamp = msg.header.stamp
                        tf_base_to_board = self.tf_buffer.lookup_transform(
                            self.robot_base_link,
                            self.calibration_board_link,
                            stamp,
                            timeout=Duration(seconds=0.5)
                        )

                        # Transform from base_link → board by inverting (board → base_link)
                        t_base_to_board = np.array([
                            tf_base_to_board.transform.translation.x,
                            tf_base_to_board.transform.translation.y,
                            tf_base_to_board.transform.translation.z
                            ])
                        q_base_to_board = np.array([
                            tf_base_to_board.transform.rotation.x,
                            tf_base_to_board.transform.rotation.y,
                            tf_base_to_board.transform.rotation.z,
                            tf_base_to_board.transform.rotation.w
                            ])

                        R_base_to_board = quat_to_rot(q_base_to_board)

                        # Save this robot pose (base_link → board)
                        self.robot_poses.append((R_base_to_board, t_base_to_board))
                        
                        self.estimated_transforms["base_to_board"] = [
                            self.robot_base_link,
                            'charuco_board_link_estimated2',
                            R_base_to_board,
                            t_base_to_board]

                        if len(self.robot_poses) > 10:
                            self.perform_hand_eye_calib()
                            self.calibrated = True
                        
                    except Exception as e:
                        print(f'TF lookup failed: {e}')
        
        print("="*15 + " Image Callback - Execution Finished " + "="*15)


    def camera_info_callback(self, msg):

        # Callback function for Camera Info subscriber

        print("="*15 + " Camera Info Callback - Execution Started " + "="*15)
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
        print("="*15 + " Camera Info Callback - Execution Finished " + "="*15)
    

    def perform_hand_eye_calib(self):
        print("="*10 + " Static Calibration - Execution Started " + "="*10)

        # Confirm available pose values
        if not self.robot_poses or not self.cam_poses:
            self.get_logger().error("No pose data available for calibration.")
            return

        # Using only the most recent single observation
        
        # ROBOT BASE -> BOARD (From TF Tree)
        (R_base_to_board, t_base_to_board) = self.robot_poses[-1]
        save_transform(R_base_to_board, t_base_to_board, self.robot_base_link, self.calibration_board_link, "Transform : Robot Base to Charuco Board")

        # CAMERA IMAGE -> BOARD (From charuco detection)
        (rvec, tvec) = self.cam_poses[-1]
        R_camera_image_to_board, _ = cv2.Rodrigues(rvec)
        t_camera_image_to_board = tvec.reshape(3)
        save_transform(R_camera_image_to_board, t_camera_image_to_board, self.camera_image_link, self.calibration_board_link, "Transform : Camera Image Frame to Charuco Board")

        # BOARD -> CAMERA IMAGE (Invert camera image to board)
        R_board_to_camera_image = R_camera_image_to_board.T
        t_board_to_camera_image = -R_camera_image_to_board.T @ t_camera_image_to_board
        save_transform(R_board_to_camera_image, t_board_to_camera_image, self.calibration_board_link, self.camera_image_link, "Transform : Charuco Board to Camera Image Frame")

        # ROBOT BASE -> CAMERA IMAGE (Compute)
        R_base_to_camera_image = R_base_to_board @ R_board_to_camera_image
        t_base_to_camera_image = R_base_to_board @ t_board_to_camera_image + t_base_to_board
        save_transform(R_base_to_camera_image, t_base_to_camera_image, self.robot_base_link, self.camera_image_link, "Transform : Robot Base to Camera Image Frame")

        self.estimated_transforms["base_to_camera_image"] = [
                            self.robot_base_link,
                            'camera_color_optical_frame_estimated1',
                            R_base_to_camera_image, 
                            t_base_to_camera_image]

        # Try Robot Base to Camera Base
        try:
            
            # CAMERA BASE -> CAMERA IMAGE (From TF tree)
            tf_camera_base_to_image = self.tf_buffer.lookup_transform(
                self.camera_base_link,          # base link of camera
                self.camera_image_link,         # image (optical) frame of camera
                rclpy.time.Time(),              # latest available
                timeout=Duration(seconds=0.5)
            )

            # Extract transform
            t_camera_base_to_image = np.array([
                tf_camera_base_to_image.transform.translation.x,
                tf_camera_base_to_image.transform.translation.y,
                tf_camera_base_to_image.transform.translation.z
            ])
            q_camera_base_to_image = np.array([
                tf_camera_base_to_image.transform.rotation.x,
                tf_camera_base_to_image.transform.rotation.y,
                tf_camera_base_to_image.transform.rotation.z,
                tf_camera_base_to_image.transform.rotation.w
            ])
            R_camera_base_to_image = quat_to_rot(q_camera_base_to_image)
            save_transform(R_camera_base_to_image, t_camera_base_to_image, self.camera_base_link, self.camera_image_link, "Transform : Camera Base to Camera Image Frame")


            # CAMERA IMAGE -> CAMERA BASE (Invert camera base to camera image)
            R_image_to_camera_base = R_camera_base_to_image.T
            t_image_to_camera_base = -R_camera_base_to_image.T @ t_camera_base_to_image


            # ROBOT BASE -> CAMERA BASE (Final for event cameras)
            R_base_to_camera_base = R_base_to_camera_image @ R_image_to_camera_base
            t_base_to_camera_base = R_base_to_camera_image @ t_image_to_camera_base + t_base_to_camera_image
            save_transform(R_base_to_camera_base, t_base_to_camera_base, self.robot_base_link, self.camera_base_link, "Transform : Robot Base to Camera Base")

            print(f"\nFinal static camera_base pose wrt base_link:")
            print(f"Rotation matrix (R) :\n{R_base_to_camera_base}")
            print(f"Translation vector (t) : {t_base_to_camera_base}\n")

            rot_base_to_camera_base = R_scipy.from_matrix(R_base_to_camera_base)
            rpy_base_to_camera_base = rot_base_to_camera_base.as_euler('xyz')
            print(f"Rotation Angles (rpy) : {rpy_base_to_camera_base}\n")

            # Save the required calibration
            save_calibration(R_base_to_camera_base, t_base_to_camera_base, self.robot_base_link, self.camera_base_link)

            # Store result for TF publishing
            self.latest_R_base2camera = R_base_to_camera_base
            self.latest_t_base2camera = t_base_to_camera_base

            if self.camera_type == 'Robot':
                # Compute tranfrom from camera holder to camera base

                try:

                    # ROBOT BASE -> CAMERA HOLDER (From TF tree)
                    tf_base_to_camera_holder = self.tf_buffer.lookup_transform(
                        self.robot_base_link,          # base link of camera
                        self.camera_holder_link,        # holder link of camera
                        rclpy.time.Time(),              # latest available
                        timeout=Duration(seconds=0.5)
                    )

                    # Extract transform
                    t_base_to_camera_holder = np.array([
                        tf_base_to_camera_holder.transform.translation.x,
                        tf_base_to_camera_holder.transform.translation.y,
                        tf_base_to_camera_holder.transform.translation.z
                    ])
                    q_base_to_camera_holder = np.array([
                        tf_base_to_camera_holder.transform.rotation.x,
                        tf_base_to_camera_holder.transform.rotation.y,
                        tf_base_to_camera_holder.transform.rotation.z,
                        tf_base_to_camera_holder.transform.rotation.w
                    ])
                    R_base_to_camera_holder = quat_to_rot(q_base_to_camera_holder)
                    save_transform(R_base_to_camera_holder, t_base_to_camera_holder, self.robot_base_link, self.camera_holder_link, "Transform : Robot Base to Camera Holder Link")
                    
                    # CAMERA HOLDER -> ROBOT BASE (Invert robot base to camera holder)
                    R_camera_holder_to_base = R_base_to_camera_holder.T
                    t_camera_holder_to_base = -R_base_to_camera_holder.T @ t_base_to_camera_holder

                    # CAMERA HOLDER -> CAMERA BASE (Compute : Final for Robot Cameras)
                    R_camera_holder_to_camera_base = R_camera_holder_to_base @ R_base_to_camera_base
                    t_camera_holder_to_camera_base = R_camera_holder_to_base @ t_base_to_camera_base + t_camera_holder_to_base
                    save_transform(R_camera_holder_to_camera_base, t_camera_holder_to_camera_base, self.camera_holder_link, self.camera_base_link, "Transform : Camera Holder to Camera Base")

                    print(f"\nFinal static camera_base pose wrt camera_holder:")
                    print(f"Rotation matrix (R) :\n{R_camera_holder_to_camera_base}")
                    print(f"Translation vector (t) : {t_camera_holder_to_camera_base}\n")

                    rot_camera_holder_to_camera_base = R_scipy.from_matrix(R_camera_holder_to_camera_base)
                    rpy_camera_holder_to_camera_base = rot_camera_holder_to_camera_base.as_euler('xyz')
                    print(f"Rotation Angles (rpy) : {rpy_camera_holder_to_camera_base}\n")

                    # Save the required calibration
                    save_calibration(R_camera_holder_to_camera_base, t_camera_holder_to_camera_base, self.camera_holder_link, self.camera_base_link)

                    # Store result for TF publishing
                    self.latest_R_holder2camera = R_camera_holder_to_camera_base
                    self.latest_t_holder2camera = t_camera_holder_to_camera_base
                
                except Exception as e:
                    print(f'TF lookup failed: {e}')
                    self.get_logger().error(f"TF lookup for Robot Base -> Camera Holder failed: {e}")

            print("Calibration values saved!")

        except Exception as e:
            print(f'TF Lookup failed: {e}')
            self.get_logger().error(f"TF lookup for camera base -> camera image failed: {e}")
        
        print("="*10 + " Static Calibration - Execution Finished " + "="*10)


    def publish_calibrated_tf_timer(self):

        # Publish based on camera type
        latest_R = self.latest_R_base2camera
        latest_t = self.latest_t_base2camera
        parent_link = self.robot_base_link
        if self.camera_type == "Robot":
            latest_R = self.latest_R_holder2camera
            latest_t = self.latest_t_holder2camera
            parent_link = self.camera_holder_link
        
        child_link = self.camera_base_link
        #child_link = "camera_link_calibrated"

        # Confirm Calibration
        if not self.calibrated:
            print("TF Publisher (Calibration Result) - Calibration not finsihed yet")
            return
        
        if latest_R is None or latest_t is None:
            print("TF Publisher (Calibration Result) - Incomplete TF")
            return

        print("TF Publisher (Calibration Result) - Publishing TF!")
        # Convert rotation matrix to quaternion
        latest_rot = R_scipy.from_matrix(latest_R)
        quaternion_anlges = latest_rot.as_quat()  # [x, y, z, w]
        euler_angles = latest_rot.as_euler('xyz')

        t_msg = TransformStamped()
        t_msg.header.stamp = self.get_clock().now().to_msg()
        t_msg.header.frame_id = parent_link
        t_msg.child_frame_id = child_link

        t_msg.transform.translation.x = float(latest_t[0])
        t_msg.transform.translation.y = float(latest_t[1])
        t_msg.transform.translation.z = float(latest_t[2])
        t_msg.transform.rotation.x = float(quaternion_anlges[0])
        t_msg.transform.rotation.y = float(quaternion_anlges[1])
        t_msg.transform.rotation.z = float(quaternion_anlges[2])
        t_msg.transform.rotation.w = float(quaternion_anlges[3])

        self.tf_broadcaster.sendTransform(t_msg)

        print("="*10 + " Calibration Result Transform " + "="*10)
        print(f"Parent Link : {parent_link}\n")
        print(f"Child Link : {child_link}\n")
        print(f"Translation (t): {latest_t.ravel()}")
        print(f"Rotation matrix (R):\n{latest_R}")
        print(f"Quaternion (x,y,z,w): {quaternion_anlges}\n")
        print(f"RPY angles : [{euler_angles[0]:.2f}, {euler_angles[1]:.2f}, {euler_angles[2]:.2f}]")
    

    def publish_tf_timer(self):
        if not self.calibrated:
            print("TF Publisher (Other TFs) - Calibration not finsihed yet")
            return

        if not self.estimated_transforms:
            print("TF Publisher (Other TFs) - No TFs to publish!")
            return 

        print("TF Publsiher (Other TFs) - Publishing TFs")

        for tf_item in self.estimated_transforms.values():
            
            parent_link = tf_item[0]
            child_link = tf_item[1]
            Rot_matrix = tf_item[2]
            #quaternion_anlges = tf_item[2]
            trans_vector = tf_item[3]

            # Convert rotation matrix to quaternion
            rot = R_scipy.from_matrix(Rot_matrix)
            quaternion_anlges = rot.as_quat()  # [x, y, z, w]
            euler_angles = rot.as_euler('xyz')

            t_msg = TransformStamped()
            t_msg.header.stamp = self.get_clock().now().to_msg()
            t_msg.header.frame_id = parent_link
            t_msg.child_frame_id = child_link

            t_msg.transform.translation.x = float(trans_vector[0])
            t_msg.transform.translation.y = float(trans_vector[1])
            t_msg.transform.translation.z = float(trans_vector[2])
            t_msg.transform.rotation.x = float(quaternion_anlges[0])
            t_msg.transform.rotation.y = float(quaternion_anlges[1])
            t_msg.transform.rotation.z = float(quaternion_anlges[2])
            t_msg.transform.rotation.w = float(quaternion_anlges[3])

            self.tf_broadcaster.sendTransform(t_msg)
            # print("="*10 + " Estimated Transform " + "="*10)
            # print(f"Translation (t): {trans_vector.ravel()}")
            # print(f"Rotation matrix (R):\n{Rot_matrix}")
            # print(f"Quaternion (x,y,z,w): {quaternion_anlges}\n")
            # print(f"RPY angles : Roll={euler_angles[0]:.2f}, Pitch={euler_angles[1]:.2f}, Yaw={euler_angles[2]:.2f}")


def quat_to_rot(q):
    """Convert [x, y, z, w] quaternion to 3x3 rotation matrix"""
    return R_scipy.from_quat(q).as_matrix()

def save_transform(R: np.ndarray, t: np.ndarray, parent_link: str = "Unknown", child_link: str = "Unknown", label: str = "Transform"):
    """
    Print and save the translation vector, rotation matrix, and RPY angles.

    Args:
        R (np.ndarray): 3x3 rotation matrix
        t (np.ndarray): 3x1 or 1x3 translation vector
        parent_link (str): Parent Link of transform
        child_link (str): Child Link of transform
        label (str): Optional label for the transform
    """

    filename='src/board_calib/config/intermediate_values_estimated.yaml'

    # Ensure t is a flat array
    t_flat = t.flatten()

    print("\n" + "="*10 + f"=== {label} ===" + "="*10)
    print("Rotation Matrix (R):")
    print(R)

    # Convert rotation matrix to RPY (roll, pitch, yaw)
    rot = R_scipy.from_matrix(R)
    rpy = rot.as_euler('xyz')  # xyz order = roll, pitch, yaw

    print(f"Rotation Angles (rpy): [{rpy[0]:.4f}, {rpy[1]:.4f}, {rpy[2]:.4f}]")
    print(f"Translation (t): [{t_flat[0]:.4f}, {t_flat[1]:.4f}, {t_flat[2]:.4f}]")

    # Load existing YAML content if file exists
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            try:
                data = yaml.safe_load(f) or {}
            except yaml.YAMLError:
                data = {}
    else:
        data = {}

    # Add new transform
    data[label] = {
        'parent_frame': parent_link,
        'child_frame': child_link,
        'xyz': [float(t_flat[0]), float(t_flat[1]), float(t_flat[2])],
        'rpy': [float(rpy[0]), float(rpy[1]), float(rpy[2])]
    }
    
    with open(filename, 'w') as f:
        yaml.dump(data, f)

    print(f"Transform saved to {filename}")

def save_calibration(R, t, parent_link, child_link, filename='src/board_calib/config/calibration_values_estimated.yaml'):

    label = "Transform : " + parent_link + "_to_" + child_link

    # Convert rotation matrix to roll-pitch-yaw (in radians)
    rpy = R_scipy.from_matrix(R).as_euler('xyz', degrees=False)

    # Flatten translation
    xyz = t.flatten()

    # Load existing YAML content if file exists
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            try:
                data = yaml.safe_load(f) or {}
            except yaml.YAMLError:
                data = {}
    else:
        data = {}

    # YAML content
    data[label] = {
        'parent_frame': parent_link,
        'child_frame': child_link,
        'xyz': [float(xyz[0]), float(xyz[1]), float(xyz[2])],
        'rpy': [float(rpy[0]), float(rpy[1]), float(rpy[2])]
    }

    with open(filename, 'w') as f:
        yaml.dump(data, f)
    
    print(f"Calibration Results saved to : {filename}")

def main(args=None):
    rclpy.init(args=args)
    node = EnvCamCalibNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()