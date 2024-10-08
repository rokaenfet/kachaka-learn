import kachaka_api
import time
import asyncio
import cv2
import numpy as np
import keyboard
import mediapipe as mp
import aioconsole
import math
import glob
import os
import ultralytics.engine
import ultralytics.engine.results
import matplotlib.pyplot as plt
import math
import sys
import matplotlib.patches as patches
from kachaka_api.util.geometry import MapImage2DGeometry
from matplotlib.animation import FuncAnimation
from matplotlib.transforms import Affine2D
import io
from PIL import Image
from scipy import stats

from mebow_model import MEBOWFrame
from ultralytics import YOLO
import pyrealsense2 as rs
import ultralytics
from typing import Tuple, Optional, Dict
from functools import wraps

import logging

PI = math.pi

FONT = cv2.FONT_HERSHEY_PLAIN
WHITE = (255,255,255)
RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)
BLACK = (0,0,0)
LINEAR_SPEED = 0.3
AUTO_LINEAR_SPEED = 1
ANGULAR_SPEED = 1
AUTO_ANGULAR_SPEED = .8
EMERGENCY_STOP_DISTANCE = 0.2
THRE = 30
YOLO_CONF_THRE = 0.85
lazy_cv2_txt_params = (FONT, 3, GREEN, 3)
MIN_LINEAR_SPEED = 0.1
MIN_ANGULAR_SPEED = 0.1
MAX_LINEAR_SPEED = 1
MAX_ANGULAR_SPEED = 1
MOVE_ANGLE_THRE = 0.03
MOVE_EUCLIDEAN_DIST_THRE = 0.03
STRAIGHT_FOV_THRE = PI/16 # rads
MAX_D_STEP = 3 # meter
SCREEN_W, SCREEN_H = 1280, 720
CHARGE_THRE = (40, 80)

# log
logging.basicConfig(
    filename='stalk/app.log',          # The name of the log file
    level=logging.DEBUG,         # The minimum level of log messages to handle
    format='%(asctime)s - %(levelname)s - %(message)s',  # The format of log messages
    datefmt='%Y-%m-%d %H:%M:%S',  # The format for the date/time
    filemode="w"    
)

import asyncio
import logging
from functools import wraps
import numpy as np

def format_value(value):
    """Recursively formats values for logging, handling nested np.ndarray objects."""
    if isinstance(value, np.ndarray):
        return f'np.ndarray(shape={value.shape})'
    elif isinstance(value, (list, tuple)):
        return [format_value(item) for item in value]
    elif isinstance(value, dict):
        return {k: format_value(v) for k, v in value.items()}
    else:
        return value

def extract_kachaka_frame_id(args):
    """Extracts KachakaFrame.id from the class instance if present in the arguments."""
    if len(args) > 0 and hasattr(args[0], 'id'):
        return args[0].id
    return None

def log_function_data(func):
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        # Extract KachakaFrame.id if available
        kachaka_id = extract_kachaka_frame_id(args)
        kachaka_id_info = f' [KachakaFrame.id={kachaka_id}]' if kachaka_id is not None else ''
        
        # Format arguments and keyword arguments
        formatted_args = format_value(args)
        formatted_kwargs = format_value(kwargs)
        logging.info(f'Starting async function {func.__name__}{kachaka_id_info} with arguments {formatted_args} and keyword arguments {formatted_kwargs}')
        
        try:
            # Call the async function and await the result
            result = await func(*args, **kwargs)
            
            # Format return value
            formatted_result = format_value(result)
            logging.info(f'Async function {func.__name__}{kachaka_id_info} returned {formatted_result}')
            return result
        except Exception as e:
            # Log any exceptions that occur
            logging.error(f'Async function {func.__name__}{kachaka_id_info} raised an exception: {e}')
            raise

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        # Extract KachakaFrame.id if available
        kachaka_id = extract_kachaka_frame_id(args)
        kachaka_id_info = f' [KachakaFrame.id={kachaka_id}]' if kachaka_id is not None else ''
        
        # Format arguments and keyword arguments
        formatted_args = format_value(args)
        formatted_kwargs = format_value(kwargs)
        logging.info(f'Starting function {func.__name__}{kachaka_id_info} with arguments {formatted_args} and keyword arguments {formatted_kwargs}')
        
        try:
            # Call the function and get the result
            result = func(*args, **kwargs)
            
            # Format return value
            formatted_result = format_value(result)
            logging.info(f'Function {func.__name__}{kachaka_id_info} returned {formatted_result}')
            return result
        except Exception as e:
            # Log any exceptions that occur
            logging.error(f'Function {func.__name__}{kachaka_id_info} raised an exception: {e}')
            raise

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

class RealSenseCamera:
    def __init__(self, depth_width: int = 1280, depth_height: int = 720, 
                 rgb_width: int = 1920, rgb_height: int = 1080, depth_fps: int = 30, rgb_fps: int = 30) -> None:
        # Configuration parameters
        self.depth_width: int = depth_width
        self.depth_height: int = depth_height
        self.depth_fps: int = depth_fps
        self.rgb_width: int = rgb_width
        self.rgb_height: int = rgb_height
        self.rgb_fps: int = rgb_fps

        # Create a pipeline
        self.pipeline: rs.pipeline = rs.pipeline()

        # Check device and config
        pipeline = rs.pipeline()
        config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        print(f"{C.BLUE}Detected Camera{C.RESET}: {device_product_line}")
        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print(f"{C.RED}Failed{C.RESET} to find RGB camera in RealSense")
            sys.exit()
        
        # Create a config object and enable streams
        self.config: rs.config = rs.config()
        self.config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, depth_fps)
        self.config.enable_stream(rs.stream.color, rgb_width, rgb_height, rs.format.bgr8, rgb_fps)
        self.config.enable_stream(rs.stream.gyro)
        self.config.enable_stream(rs.stream.accel)
        
        # Start pipeline
        self.profile = self.pipeline.start(self.config)
        
        # Align depth to color
        self.align: rs.align = rs.align(rs.stream.color)
        
        # Set depth scaling (to convert depth from integer to meters)
        self.depth_scale: float = self.get_depth_scale()

        # Post-processing filters
        self.depth_to_disparity = rs.disparity_transform(True)
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()
        self.disparity_to_depth = rs.disparity_transform(False)

        # for orientation
        self.prev_angles = {"pitch":0, "roll":0, "yaw":0, "time":time.time()} # assumes it starts at orientation(0,0,0)

    def get_depth_scale(self) -> float:
        """
        Get the depth scale from the camera to convert depth values to meters.
        """
        profile: rs.pipeline_profile = self.pipeline.get_active_profile()
        depth_sensor: rs.sensor = profile.get_device().first_depth_sensor()
        return depth_sensor.get_depth_scale()
    
    @log_function_data
    def get_depth_at_pixel(self, x: int, y: int, depth_image: np.ndarray, color_image: np.ndarray) -> Optional[float]:
        """
        Get the depth value at the specific pixel (x, y) in the color image.
        Returns the depth in meters, or None if depth data is unavailable.
        """
        if depth_image.shape[:2] != color_image.shape[:2]:
            return None
        if depth_image is None or color_image is None:
            return None

        # Ensure the pixel coordinates are within the depth image bounds
        if 0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]:
            depth_value = depth_image[y, x]  # Depth is in millimeters
            if depth_value == 0:
                return None
            return depth_value * self.depth_scale  # Convert depth to meters
        else:
            return None

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Wait for the next set of frames and return aligned depth and color images.
        Returns a tuple of (depth_image, color_image), or (None, None) if frames are unavailable.
        """
        frames: rs.frameset = self.pipeline.wait_for_frames()
        aligned_frames: rs.frameset = self.align.process(frames)
        depth_frame: rs.frame = aligned_frames.get_depth_frame()
        color_frame: rs.frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None
        
        # Apply post-processing filters
        depth_frame = self.apply_post_processing(depth_frame)
        
        # Convert frames to numpy arrays
        depth_image: np.ndarray = np.asanyarray(depth_frame.get_data())
        color_image: np.ndarray = np.asanyarray(color_frame.get_data())
        return depth_image, color_image
    
    def get_orientation(self) -> tuple[float, float, float]:
        """Estimate pitch, roll, yaw from D435i camera IMU

        return:
            tuple:
                yaw[float]

                pitch[float]

                roll[float]
        """
        frames = self.pipeline.wait_for_frames()
        gyro_frame = frames.first_or_default(rs.stream.gyro)
        accel_frame = frames.first_or_default(rs.stream.accel)
        if gyro_frame and accel_frame:
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()
            accel_data = accel_frame.as_motion_frame().get_motion_data() 
            # print(f"Gyro: [{gyro_data.x}, {gyro_data.y}, {gyro_data.z}]")
            # print(f"Accel: [{accel_data.x}, {accel_data.y}, {accel_data.z}]")   
            alpha = 0.98  # Complementary filter coefficient (tweak this value)

            # Extract accelerometer and gyroscope data
            ax, ay, az = accel_data.x, accel_data.y, accel_data.z
            gx, gy, gz = gyro_data.x, gyro_data.y, gyro_data.z

            # Compute pitch and roll from accelerometer (in radians)
            pitch_accel = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
            roll_accel = np.arctan2(ay, az)
            
            # Integrate gyroscope data to get angular change
            ct = time.time()
            dt = ct - self.prev_angles["time"]
            pitch_gyro = self.prev_angles['pitch'] + gx * dt
            roll_gyro = self.prev_angles['roll'] + gy * dt
            yaw_gyro = self.prev_angles['yaw'] + gz * dt

            # Apply complementary filter to combine accelerometer and gyroscope
            pitch = alpha * pitch_gyro + (1 - alpha) * pitch_accel
            roll = alpha * roll_gyro + (1 - alpha) * roll_accel
            yaw = yaw_gyro  # Yaw cannot be corrected by accelerometer; rely on gyro

            self.prev_angles["time"] = ct
            self.prev_angles["pitch"] = pitch
            self.prev_angles["roll"] = roll
            self.prev_angles["yaw"] = yaw

            return yaw, pitch, roll

    def apply_post_processing(self, depth_frame: rs.frame) -> rs.frame:
        """
        Apply post-processing filters to improve depth data quality.
        """
        depth_frame = self.depth_to_disparity.process(depth_frame)
        depth_frame = self.spatial_filter.process(depth_frame)
        depth_frame = self.temporal_filter.process(depth_frame)
        depth_frame = self.disparity_to_depth.process(depth_frame)
        return depth_frame

    @log_function_data
    def get_intrinsics(self) -> Dict[str, rs.intrinsics]:
        """
        Get the intrinsic camera parameters for both depth and color streams.
        Returns a dictionary with 'depth_intrinsics' and 'color_intrinsics'.
        """
        depth_stream: rs.video_stream_profile = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_stream: rs.video_stream_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        
        depth_intrinsics: rs.intrinsics = depth_stream.get_intrinsics()
        color_intrinsics: rs.intrinsics = color_stream.get_intrinsics()
        return {'depth_intrinsics': depth_intrinsics, 'color_intrinsics': color_intrinsics}

    @log_function_data
    def get_extrinsics(self) -> rs.extrinsics:
        """
        Get the extrinsic parameters (transformation) between depth and color streams.
        Returns an rs.extrinsics object.
        """
        depth_stream: rs.stream_profile = self.profile.get_stream(rs.stream.depth)
        color_stream: rs.stream_profile = self.profile.get_stream(rs.stream.color)
        
        extrinsics: rs.extrinsics = depth_stream.get_extrinsics_to(color_stream)
        return extrinsics

    def get_colormap(self, depth_image: np.ndarray) -> np.ndarray:
        """
        Convert depth image to a color map for visualization.
        """
        depth_colormap: np.ndarray = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )
        return depth_colormap

    @log_function_data
    def stop(self) -> None:
        """
        Stop the pipeline and release any resources.
        """
        self.pipeline.stop()
        cv2.destroyAllWindows()

class KachakaFrame():
    human_detection_result: ultralytics.engine.results.Boxes

    def __init__(self, IP:str, id:int):
        logging.info(f"KachakaFrame.__init__() start. params:[{IP}, {id}]")
        self.id = id
        self.sync_client = kachaka_api.KachakaApiClient(IP)
        self.async_client = kachaka_api.aio.KachakaApiClient(IP)
        print(f"Kachaka ID:{self.id} has {C.GREEN}connected{C.RESET} with address: {IP}")
        self.stream_i = self.async_client.front_camera_ros_compressed_image.stream()
        self.stream_d = self.async_client.object_detection.stream()
        print(f"{C.GREEN}got{C.RESET} stream")
        _ = self.sync_client.get_front_camera_ros_compressed_image()
        self.undistort_map = get_camera_info(self.sync_client)
        print(f"{C.GREEN}got{C.RESET} camera info")
        self.error_code = self.sync_client.get_robot_error_code()
        print(f"{C.GREEN}got{C.RESET} error code")
        self.need_to_emergency_stop = False
        self.target_found = False
        print(f"{C.GREEN}set{C.RESET} manual control = True; auto homing = False")
        self.sync_client.set_manual_control_enabled(True)
        self.sync_client.set_auto_homing_enabled(False)
        # self.sync_client.undock_shelf()
        # self.sync_client.dock_shelf()

        # auxilary vars
        # needed vars
        self.linear = 0
        self.angular = 0
        self.target_pos = None
        self.cv_img = None
        self.dest_pose = None
        # unused vars
        self.being_controlled = False
        self.human_found_count = 0
        self.face_found_count = 0
        self.find_face_mode = False
        self.target_near = False
        self.cd = 0

        # get models
        self.face_detector = FaceDetect()
        print(f"{C.GREEN}Got{C.RESET} media pipe face detector")
        self.yolo_model = YOLO("yolov8n.pt")
        print(f"{C.GREEN}Got{C.RESET} YOLOv8")
        self.mebow_model = MEBOWFrame()
        print(f"{C.GREEN}Got{C.RESET} MEBOW model")
        self.mp_landmark_model = MPLandmark()
        print(f"{C.GREEN}Got{C.RESET} media pipe landmark detector")

        # load realsense camera
        # self.realsense = RealSenseCamera(
        #     depth_width=SCREEN_W,
        #     depth_height=SCREEN_H,
        #     depth_fps=30,
        #     rgb_width=SCREEN_W,
        #     rgb_height=SCREEN_H,
        #     rgb_fps=30,
        # )
        self.color_image = None
        self.depth_image = None

        self.nav_i = 0
        self.run = True
        self.run_nav = True
        self.navigating = False
        self.location_names = ["0","1","2","3","4","fu"]

        # vars for move_to_pos
        self.running_move_to_pose = False
        self.move_try_timer = 0
        self.to_adjust = True
        self.z_dist = None
        self.charge = False

        # var for visualization
        self.visualize_prev_locations = []
        self.graph_i = 0
        self.consecutive_z_dists = []
        directory = os.path.join(os.getcwd(),"stalk/img")
        png_files = glob.glob(os.path.join(directory, '*.png'))

        # map
        self.png_map = self.sync_client.get_png_map()
        self.png_map_img = cv2.cvtColor(np.array(Image.open(io.BytesIO(self.png_map.data))), cv2.COLOR_RGB2BGR)
        self.past_coords = []

        # Loop through the list of files and remove them
        for file in png_files:
            try:
                os.remove(file)
            except Exception as e:
                print(f"Error deleting {file}: {e}")
        logging.info(f"KachakaFrame.__init__() done")

    @log_function_data
    async def emergency_stop(self):
        """ detect and perform emergency stop using LiDAR data
        
        scans LiDAR from kachaka and determines if kachaka should perform an emergency stop

        Attibutes:
            lidar_scan: coroutine get_ros_laser_scan()
        """         
        logging.info(f"KachakaFrame.emergency_stop() start")
        lidar_scan = await self.async_client.get_ros_laser_scan()
        self.nearest_scan_dist = min([dist for dist in lidar_scan.ranges if dist > 0])
        if self.nearest_scan_dist < EMERGENCY_STOP_DISTANCE:
            self.need_to_emergency_stop = True
            print(f"ID:{self.id} has prevented a crash!")
            await self.async_client.speak("move")
            self.sync_client.set_robot_velocity(-self.linear, -self.angular)
        elif self.need_to_emergency_stop == True:
            self.need_to_emergency_stop = False
            await self.async_client.speak("re")
            print(f"ID:{self.id} has resumed moving!")
        logging.info(f"KachakaFrame.emergency_stop() done")

    async def manual_move_to_pose(self):
        dest_x, dest_y, dest_theta = self.dest_pose
        (cur_x, cur_y), cur_theta = await self.get_robot_pose()
        dest_theta, cur_theta = mod_radians(dest_theta), mod_radians(cur_theta)
        angle_diff = dest_theta - cur_theta
        angular_speed = 0
        # rotate first
        if not abs(angle_diff) < MOVE_ANGLE_THRE:
            angle_to_target = (angle_diff + np.pi) % (2 * np.pi) - np.pi
            angular_speed = np.clip(angle_to_target/4, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)
            angular_speed = np.sign(angular_speed)*max(abs(angular_speed), MIN_ANGULAR_SPEED)
        else:
            self.dest_pose = None
            # print(f"DESTINATION {dest_theta} REACHED")
        await self.async_client.set_robot_velocity(0, angular_speed)
        # print(np.rad2deg(dest_theta), np.rad2deg(cur_theta), np.rad2deg(angle_diff))

    @log_function_data
    async def _handle_dest_pose(self):
        dest_x, dest_y, dest_theta = self.dest_pose
        running_move_to_pose = await self.check_move_to_pose()
        (cur_x, cur_y), cur_theta = await self.get_robot_pose()
        if self.move_try_timer == 0:
            self.move_try_timer = time.time()
        dest_theta = mod_radians(dest_theta)
        cur_theta = mod_radians(cur_theta)
        diff = np.array([cur_x, cur_y, cur_theta])-np.array([dest_x, dest_y, dest_theta])
        euclidean_dist = np.linalg.norm(diff)
        # if destination pose reached
        if euclidean_dist < 0.1:
            self.dest_pose = None
            self.running_move_to_pose = False
            self.move_try_timer = 0
            if running_move_to_pose:
                await self.cancel_move_to_pose()
            logging.debug(f"goal reached:{np.round([dest_x, dest_y, dest_theta],3)}-{np.round([cur_x, cur_y, cur_theta],3)}={np.round(diff,4)}->{np.round(euclidean_dist,4)}")
        elif time.time() - self.move_try_timer > 120:
            self.dest_pose = None
            self.running_move_to_pose = False
            self.move_try_timer = 0
            if running_move_to_pose:
                await self.cancel_move_to_pose()
            logging.debug(f"time out:{np.round([dest_x, dest_y, dest_theta],3)}-{np.round([cur_x, cur_y, cur_theta],3)}={np.round(diff,4)}->{np.round(euclidean_dist,4)}")
        # if goal not reached
        elif not running_move_to_pose and not self.running_move_to_pose:
            self.running_move_to_pose = True
            logging.debug(f"kachakaFrame.short_move_to_pose() starting")
            await self.short_move_to_pose()

    @log_function_data
    async def move(self):
        """movement controller
        """
        if self.run:
            battery_percentage, power_supply_status = await self.get_battery()
            if battery_percentage > CHARGE_THRE[0]:
                self.charge = False
            if not self.charge:
                if battery_percentage < CHARGE_THRE[1]:
                    self.charge = True
                # navigate if nothing in priority
                if self.dest_pose is None and self.run_nav:
                    await self.short_navigate()
                # move to position (circling and navigation)
                elif self.dest_pose is not None:
                    await self._handle_dest_pose()
                # linear and angular velocity handle, used in forward/backwards adjusting
                elif not(self.linear is None and self.angular is None):
                    linear = self.linear if self.linear is not None else 0
                    angular = self.angular if self.angular is not None else 0
                    await self.async_client.set_robot_velocity(linear, angular)
                    self.linear, self.angular = None, None
            else:
                going_home = await self.check_return_home()
                if not going_home or power_supply_status != 1:
                    await self.async_client.set_speaker_volume(7)
                    await self.async_client.return_home(cancel_all=True)
                    await self.async_client.set_speaker_volume(0)
                    await asyncio.sleep(5)
                if battery_percentage > CHARGE_THRE[1]:
                    self.charge = False

    async def get_battery(self):
        result = await self.async_client.get_battery_info() # (battery percentage, power supply status)
        try:
            battery_percentage, power_supply_status = result
            return battery_percentage, power_supply_status
        except:
            logging.debug(f"[{self.id}] get_battery() couldn't access battery data")

    async def get_image_from_camera(self):
        """get image frame from kachaka's image stream

        Returns
        -------
        np.ndarray
            BGR image array suited for cv2
        """        
        image = await self.stream_i.__anext__()
        self.cv_img = cv2.imdecode(np.frombuffer(image.data, dtype=np.uint8), flags=1)
        self.cv_img = undistort(self.cv_img, *self.undistort_map)

    async def human_detection(self, image:np.ndarray):
        """detects human using kachaka's embedded model
        """
        results = self.yolo_model(image, verbose=False, conf=0.5)[0]
        results = [r for r in results if r.boxes.xywh.numel() > 0 and r.boxes.cls.numpy()[0] == 0]
        # if there are result bounding boxes in the current frame which has label==0==human
        if len(results) > 0:
            self.human_detection_result = results
            res_boxes = [r.boxes.xywh.numpy()[0] for r in results]
            res_boxes = [(n,n[2]*n[3]) for n in res_boxes]
            res_boxes.sort(key=lambda x:x[1])
            self.target_pos = [n for n in res_boxes[-1][0]]
            # since x,y is center xy, adjust
            self.target_pos = [
                int(self.target_pos[0]-self.target_pos[2]/2),
                int(self.target_pos[1]-self.target_pos[3]/2),
                int(self.target_pos[2]),
                int(self.target_pos[3])
                ]
            self.target_found = True
        else:
            self.target_found = False
            self.human_detection_result = None

    async def human_detection_annotate(self, do_draw_box=True, draw_target_marker=True):
        if self.human_detection_result:
            if do_draw_box:
                self.cv_img = draw_box(self.cv_img, self.human_detection_result)
            if draw_target_marker:
                cv2.putText(self.cv_img, "X", (self.target_pos[0]+self.target_pos[2]//2,
                                self.target_pos[1]+self.target_pos[3]//2), *lazy_cv2_txt_params)

    async def _prep_auto_control(self):
        """prepare automatic control

        Returns
        -------
        :float:
            d_linear is the change in linear speed to make w.r.t size of detected human in frame

        :float:
            d_angular is the change in angular speed to make w.r.t horizontal offset from center

        :float:
            (horizontal offset from center) / width of screen

        :float:
            area of bounding box / area of screen

        :float:
            top left x val of bbox

        :float:
            top left y val of bbox

        :float:
            width of bbox

        :float:
            height of bbox
        """        
        tx, ty, tw, th = self.target_pos
        center_tx, center_ty = tx+tw//2, ty+th//2
        x_r = (SCREEN_W/2-center_tx)/SCREEN_W
        area_r = (tw*th)/(SCREEN_W*SCREEN_H)
        d_linear, d_angular = (1-area_r)*AUTO_LINEAR_SPEED, x_r*AUTO_ANGULAR_SPEED
        return d_linear, d_angular, x_r, area_r, tx, ty, tw, th
    
    async def adjust(self):
        if self.to_adjust:
            d_linear, d_angular, x_r, area_r, tx, ty, tw, th = await self._prep_auto_control()
            cx = tx + tw/2
            d = SCREEN_W/2 - cx
            if abs(d) < SCREEN_W/10:
                self.to_adjust = False
                self.linear = 0
            else:
                self.linear = np.sign(d) * 0.05

    async def follow(self):
        d_linear, d_angular, x_r, area_r, tx, ty, tw, th = await self._prep_auto_control()
        self.angular = d_angular
        if area_r < 0.3:
            self.linear = d_linear
        else:
            self.linear, self.angular = 0, 0

    async def control(self):
        if keyboard.is_pressed(str(self.id)):
        # if True:
            self.being_controlled = True
            if self.need_to_emergency_stop == False:
                if keyboard.is_pressed("w"): self.linear += LINEAR_SPEED
                if keyboard.is_pressed("s"): self.linear -= LINEAR_SPEED
                if keyboard.is_pressed("a"): self.angular += ANGULAR_SPEED
                if keyboard.is_pressed("d"): self.angular -= ANGULAR_SPEED
            if self.target_found == True and keyboard.is_pressed("space") == True:
                await self.adjust()
        else:
            self.being_controlled = False

    async def annotate(self, st:float, show_fps = False, show_nearest_lidar = False, show_id = True):
        if show_fps:
            cv2.putText(self.cv_img, f"fps:{round(1/(time.time()-st))}", (20, 80), *lazy_cv2_txt_params)
        if show_nearest_lidar:
            cv2.putText(self.cv_img, f"{round(self.nearest_scan_dist, 3)}", (20, 140), *lazy_cv2_txt_params)
        if show_id:
            cv2.putText(self.cv_img, f"ID:{self.id}", (SCREEN_W-100,40), *lazy_cv2_txt_params)

    async def speak(self, txt:str):
        await self.async_client.speak(txt)

    async def get_locations(self, locations:str) -> list[str]:
        return [location for location in await self.async_client.get_locations() if location.name in locations]

    @log_function_data
    async def short_navigate(self):
        if self.run_nav:
            locations = await self.get_locations(self.location_names)
            self.dest_pose = (locations[self.nav_i].pose.x, locations[self.nav_i].pose.y, 0)
            self.nav_i = (self.nav_i+1)%len(self.location_names)
            logging.debug(f"[{self.id}] navigate to {self.location_names[self.nav_i]}")

    @log_function_data
    async def check_navigate(self):
        """
        returns True if its currently navigating
        """
        is_command_running, get_running_command, get_last_command_result = await self.check_command_states()
        return is_command_running and get_running_command is not None and get_running_command.move_to_location_command is not None

    async def check_return_home(self):
        is_command_running, get_running_command, get_last_command_result = await self.check_command_states()
        return is_command_running and get_running_command is not None and get_running_command.return_home_command is not None

    @log_function_data
    async def short_move_to_pose(self):
        """non-blocking move_to_pos()
        """
        result = await self.async_client.move_to_pose(*self.dest_pose)
        if result.success:
            # when function ends
            self.running_move_to_pose = False
        else:
            logging.error(f"[{self.id}]{C.RED}Fail{C.RESET} during move_to_pose(): {self.error_code[result.error_code]}")

    @log_function_data
    async def check_move_to_pose(self):
        """
        returns True if its currently move_to_pose()
        """
        is_command_running, get_running_command, get_last_command_result = await self.check_command_states()
        return is_command_running and get_running_command is not None and get_running_command.move_to_pose_command is not None

    @log_function_data
    async def cancel_move_to_pose(self):
        """Cancel move_to_pos
        """
        if await self.check_move_to_pose():
            await self.async_client.cancel_command()

    async def check_command_states(self) -> tuple:
        """
        Return -> bool, [pb2.GetCommandStateResponse, None], (Result, Command)
        """
        is_command_running = await self.async_client.is_command_running()
        get_running_command = await self.async_client.get_running_command()
        get_last_command_result = await self.async_client.get_last_command_result()
        return is_command_running, get_running_command, get_last_command_result
    
    @log_function_data
    async def cancel_navigation(self):
        """
        cancels move_to_location() if currently running
        """
        if await self.check_navigate():
            await self.async_client.cancel_command()

    async def _draw_orientation_line(self, deg, image, line_length=100):
        if self.target_found:
            d_linear, d_angular, x_r, area_r, tx, ty, tw, th = await self._prep_auto_control()
            rads = np.deg2rad(deg)
            center_x, center_y = tx+tw//2, ty+th//2
            end_x = int(center_x + line_length * np.cos(rads))
            end_y = int(center_y - line_length * np.sin(rads))
            cv2.line(image, (center_x, center_y), (end_x, end_y), BLUE, 3)
            cv2.putText(image, "deg="+str(round(deg,1)), (20,50), *lazy_cv2_txt_params)

    async def mebow_annotate(self, line_length=100):
        # only if there is a result
        deg = self.mebow_model.ori.numpy()
        print(deg)
        await self._draw_orientation_line(deg + 90, self.cv_img)

    def _find_deg_from_landmarks(self):
        m = self.mp_landmark_model
        rads, _, _ = m._get_deg_from_landmarks()
        sign = -1 if not m.facing_camera() else 1
        deg = np.rad2deg(sign*rads)
        return deg

    async def mp_landmark_annotate(self, line_length=100):
        m = self.mp_landmark_model
        # draw if, human detected, landmarks found, and if these 2 overlap
        if self.target_pos and m.result.pose_landmarks and self._is_landmark_in_bbox():
            deg = self._find_deg_from_landmarks() # estimate deg from relative z distance to specific landmarks
            await self._draw_orientation_line(deg, self.cv_img)
            await m.draw_landmarks_on_image(self.cv_img)

    def _is_landmark_in_bbox(self):
        h,w,_ = self.cv_img.shape
        # get unnormalized coords of landmarks and check if its within bounding box
        t = [self.mp_landmark_model._convert_to_ndarray(i) for i in 
             [MPLandmark.NOSE, MPLandmark.LEFT_SHOULDER, MPLandmark.RIGHT_SHOULDER]]
        t = [[norm_x*w, norm_y*h] for norm_x, norm_y, _ in t]
        return all([self._is_coord_in_bbox(*n) for n in t])

    def _is_coord_in_bbox(self, tx, ty):
        x,y,w,h = self.target_pos
        return x<=tx<=x+w and y<=ty<=y+h

    async def get_robot_pose(self):
        """
        return:
            type: [tuple[tuple[float, float], float]] -> ((x, y), theta)
        """
        pose = await self.async_client.get_robot_pose()
        return (pose.x, pose.y), pose.theta
    
    async def get_dist_to_target(self):
        m = self.mp_landmark_model
        img_h, img_w, _ = self.color_image.shape
        _, pitch, _ = self.realsense.get_orientation() # for getting horizontal dist towards person
        if not self.to_adjust and self.target_found and m.landmarks is not None and self.dest_pose is None:
            l = [
                MPLandmark.NOSE, MPLandmark.LEFT_SHOULDER, MPLandmark.RIGHT_SHOULDER, MPLandmark.LEFT_HIP, 
                MPLandmark.RIGHT_HIP, MPLandmark.LEFT_EYE, MPLandmark.RIGHT_EYE, MPLandmark.LEFT_EAR, 
                MPLandmark.RIGHT_EAR
                ]
            len_l = len(l)
            l = [m._convert_to_ndarray(i) for i in l]
            l = [self.realsense.get_depth_at_pixel(int(img_w*x), int(img_h*y), self.depth_image, self.color_image) for x,y,_ in l]
            l = filter_outliers_IQR(np.array([n for n in l if n is not None],dtype=float))
            if len(l) >= int(len_l*0.7): # continue only if valid depths can be retrieved
                z_dist = np.min(l) # distance in meter
                if 0.3 < z_dist < 3: # deny distances which are too far
                    if len(self.consecutive_z_dists) == 0:
                        self.consecutive_z_dists.append(z_dist)
                    elif max(np.absolute(np.array(self.consecutive_z_dists)) - z_dist) < 0.5:
                        self.consecutive_z_dists.append(z_dist)
                if len(self.consecutive_z_dists) >= 7: # smoothen z_dists to avoid too large of a distance
                    z_dist = np.median(self.consecutive_z_dists)
                    # get horizontal dist
                    self.z_dist = z_dist * np.cos(pitch)
                else:
                    self.z_dist = None
        else:
            self.consecutive_z_dists = []
            self.z_dist = None

    @log_function_data
    async def circle(self):
        def d(r): return round(np.rad2deg(r),3)
        if self.z_dist is not None:
            target_deg = self._find_deg_from_landmarks()
            target_rad = PI/2+2*PI-d(target_deg)
            self.consecutive_z_dists = []
            logging.debug(f"[{self.id}]KachakaFrame.circle(). self.z_dist:{self.z_dist}")
            (pose_x, pose_y), pose_theta = await self.get_robot_pose() # pose_theta is in range [-pi, pi]
            pose = np.array([pose_x, pose_y])
            camera_rad = PI/2 # kachaka to camera yaw offset
            kachaka_look_rad = pose_theta - PI/2 # angle kachaka is looking
            camera_look_rad = kachaka_look_rad - camera_rad # direction of camera
            target_rad = normalize_radians(target_rad + pose_theta) # target angle
            # get human world coordinate by using BOTH camera and human orientation
            human = pose + get_coords_from_angle(camera_look_rad, self.z_dist)
            # get relative coords
            rel_pose = get_coords_from_angle(PI + camera_look_rad, self.z_dist)
            # get theta_diff
            d_theta = normalize_radians(target_rad - (camera_look_rad + PI))
            print("pose",d(pose_theta), "kachaka",d(kachaka_look_rad), "target",d(target_rad), "camera",d(camera_look_rad), "d_theta",d(d_theta))
            # get new coords
            step_theta = np.sign(d_theta)*min(PI/8, abs(d_theta))
            new_theta = pose_theta + step_theta
            if np.sign(step_theta) == 1: # head to the opposite of where kachaka is looking
                rel_new = get_coords_from_angle(new_theta + PI, self.z_dist)
                new = pose + rel_new
                destination = (*new, new_theta)
                # reset rel_new for visualization
                rel_new = get_coords_from_angle(new_theta, self.z_dist)
            else:
                rel_new = get_coords_from_angle(new_theta, self.z_dist)
                new = pose + rel_new
                destination = (*new, new_theta)
            # finally check if kachaka is already on the front side of the human
            if abs(step_theta) < PI/24:
                self.dest_pose = None
                logging.debug(f"[{self.id}]Front-side reached")
            else:
                # self.dest_pose = destination
                pass
            self.to_adjust = True
            print("step_theta",d(step_theta), "new_theta",d(new_theta))

            def visualize():
                # visualize
                fig,ax = plt.subplots(figsize=(12,8))
                center = (0,0)
                kachaka = rel_pose
                arrow_sca = 40
                ax.add_artist(plt.Circle(center, self.z_dist/13, color="green", zorder=5, label="human")) # circle center
                ax.text(self.z_dist/10, 0, f"{(0,0)}", fontsize=12, ha='center', color='green', zorder=5)
                ax.add_artist(plt.Circle(center, self.z_dist, color="blue", fill=False, linestyle="dashed", zorder=5)) # circumference
                ax.add_artist(plt.Circle(kachaka, self.z_dist/13, color="black", zorder=5)) #kachaka obj
                ax.text(*kachaka+self.z_dist/10, f"{(round(rel_pose[0],2),round(rel_pose[1],2))}", fontsize=12, ha='center', color='black', zorder=5)
                # kachaka pov
                ax.arrow(*kachaka, *get_coords_from_angle(kachaka_look_rad, self.z_dist/2), width=self.z_dist/arrow_sca, shape="full", color="black", linestyle="", label="Kachaka")
                # camera pov
                ax.arrow(*kachaka, *get_coords_from_angle(camera_look_rad, self.z_dist/2), width=self.z_dist/arrow_sca, shape="full", color="yellow", linestyle="", label="camera")
                # past target lines
                self.visualize_prev_locations.append(target_rad)
                if len(self.visualize_prev_locations) > 5:
                    self.visualize_prev_locations.pop(0)
                for i,n in enumerate(self.visualize_prev_locations):
                    x,y,dx,dy = (*center, *get_coords_from_angle(n, self.z_dist))
                    if i == len(self.visualize_prev_locations)-1:
                        # most recent line to be draw with different colour and full alpha
                        ax.arrow(x, y, dx, dy, color="red", linestyle="solid", lw=1, width=self.z_dist/arrow_sca, shape="full", label="present human")
                        ax.text(x+dx, y+dy-self.z_dist/10, f"{round(np.rad2deg(n),1)}", fontsize=12, ha='center', color='red', zorder=5)
                    else:
                        ax.arrow(x, y, dx, dy, color="orange", linestyle="solid", lw=1, alpha=(255/5 * (i+1))/255, width=self.z_dist/arrow_sca, label=f"[{i}] past human")
                        ax.text(x+dx, y+dy-self.z_dist/10, f"{round(np.rad2deg(n),1)}", fontsize=12, ha='center', color='orange', zorder=5)
                # kachaka destination annotations
                ax.plot((kachaka[0], rel_new[0]), (kachaka[1], rel_new[1]), color="purple")
                ax.text(*rel_new+self.z_dist/10, f"{(round(rel_new[0],2), round(rel_new[1],2))}", fontsize=12, ha='center', color='purple', zorder=5)
                # ax.arrow(*kachaka, *get_coords_from_angle(step_theta, self.z_dist), color="purple", linestyle="solid", lw=1, width=self.z_dist/arrow_sca, shape="full", label="destination", alpha=.5)
                ax.text(kachaka[0]+self.z_dist/3, kachaka[1]-self.z_dist/6, f"To-Face:{round(np.rad2deg(pose_theta),1)}°\nz_dist:{round(self.z_dist,3)}m", fontsize=12, ha='center', color='purple')
                # draw some info text on center
                ax.text(0, self.z_dist/6, f'Radius = {round(self.z_dist,5)}', fontsize=12, ha='center', color='purple')
                c = 1.2
                ax.set_xlim(-self.z_dist*c, self.z_dist*c)
                ax.set_ylim(-self.z_dist*c, self.z_dist*c)
                ax.set_aspect("equal")
                ax.axis("off")
                ax.legend()
                fig.savefig("stalk/visualize.png", bbox_inches="tight")
                # save frame by frame
                fig.savefig(f"stalk/img/graph_{str(self.graph_i).zfill(3)}.png")
                plt.close(fig)
                self.graph_i += 1
            visualize()

    async def draw_map(self):
        ROBOT_SIZE = 20
        LANDMARK_SIZE = 10
        _, yaw = await self.get_robot_pose()
        map_image_2d_geometry = MapImage2DGeometry(self.png_map)
        p = map_image_2d_geometry.calculate_robot_pose_matrix_in_pixel(self.sync_client.get_robot_pose())
        map_x, map_y = p[0][2], p[1][2]
        map_coord = np.array([map_x, map_y])

        img = self.png_map_img.copy()

        # draw robot
        box = np.int32(cv2.boxPoints([map_coord, [ROBOT_SIZE, ROBOT_SIZE], 360-np.rad2deg(yaw)]))
        img = cv2.fillPoly(img, [box], color=cv2_c.black)
        
        # draw robot trail
        for i, c in enumerate(self.past_coords):
            img = cv2.circle(img, np.int32(c), int((ROBOT_SIZE/2)*i/len(self.past_coords)), color=cv2_c.red, thickness=-1)

        # add to robob trail
        if len(self.past_coords) > 50:
            self.past_coords.pop(0)
        self.past_coords.append(map_coord)

        # draw landmarks
        self.locations = await self.get_locations(self.location_names + ["充電ドック"])
        for location in self.locations:
            pose = kachaka_api.pb2.Pose()
            name, pose.x, pose.y = location.name, location.pose.x, location.pose.y
            pose.theta = 0
            p = map_image_2d_geometry.calculate_robot_pose_matrix_in_pixel(pose)
            org = np.array([p[0][2], p[1][2]], dtype=np.int32)
            img = cv2.circle(img, org, LANDMARK_SIZE, color=cv2_c.green, thickness=1)
            name = "charger" if name == "充電ドック" else name
            img = cv2.putText(img, f"{name}", org, cv2.FONT_HERSHEY_SIMPLEX, 1, color=cv2_c.black, thickness=1)

        # draw battery
        b_p, power_supply_status = await self.get_battery()
        b_p /= 100
        bar_len = 200
        img = cv2.rectangle(img, [0,0], [bar_len,30], color=cv2_c.gray, thickness=-1)
        img = cv2.rectangle(img, [0,0], [int(b_p*bar_len),30], color=cv2_c.green, thickness=-1)
        for c in CHARGE_THRE:
            img = cv2.line(img, [int(bar_len*c/100), 0], [int(bar_len*c/100), 30], color=cv2_c.black, thickness=1)

        return img

class FaceDetect():
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detector = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)

        self.results = None
        
    async def process(self, img):
        self.results = self.face_detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def is_face_present(self):
        if self.results:
            return self.results.detections
    
    def draw_landmarks(self, img):
        for detection in self.results.detections:
            self.mp_drawing.draw_detection(img, detection)

class FaceMesh():
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.face_mesh_detection = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5)

        self.results = None
        
    def process(self, img):
        self.results = self.face_mesh_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def is_face_present(self):
        if self.results:
            return self.results.multi_face_landmarks
    
    def draw_landmarks(self, img, draw_tesselation=True, draw_contours=True, draw_irises=True):
        for face_landmarks in self.results.multi_face_landmarks:
            if draw_tesselation == True:
                self.mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
            if draw_contours == True:
                self.mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_contours_style())
            if draw_irises == True:
                self.mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())

class MPLandmark():
    NOSE=0
    LEFT_EYE_INNER=1
    LEFT_EYE=2
    LEFT_EYE_OUTER=3
    RIGHT_EYE_INNER=4
    RIGHT_EYE=5
    RIGHT_EYE_OUTER=6
    LEFT_EAR=7
    RIGHT_EAR=8
    MOUTH_LEFT=9
    MOUTH_RIGHT=10
    LEFT_SHOULDER=11
    RIGHT_SHOULDER=12
    LEFT_ELBOW=13
    RIGHT_ELBOW=14
    LEFT_WRIST=15
    RIGHT_WRIST=16
    LEFT_PINKY=17
    RIGHT_PINKY=18
    LEFT_INDEX=19
    RIGHT_INDEX=20
    LEFT_THUMB=21
    RIGHT_THUMB=22
    LEFT_HIP=23
    RIGHT_HIP=24
    LEFT_KNEE=25
    RIGHT_KNEE=26
    LEFT_ANKLE=27
    RIGHT_ANKLE=28
    LEFT_HEEL=29
    RIGHT_HEEL=30
    LEFT_FOOT_INDEX=31
    RIGHT_FOOT_INDEX=32
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.model = self.mp_pose.Pose(
            static_image_mode = True,
            min_detection_confidence = 0.4
        )
        self.mp_drawer = mp.solutions.drawing_utils
        self.landmarks = None
        # options = vision.PoseLandmarkerOptions(
        #     base_options=python.BaseOptions(model_asset_path='pose_landmarker_heavy.task'),
        #     output_segmentation_masks=True)
        # self.detector = vision.PoseLandmarker.create_from_options(options)

    async def process(self, image):
        self.result = self.model.process(image)
        if self.result.pose_landmarks:
            self.landmarks = self.result.pose_landmarks.landmark
        else:
            self.landmarks = None

    async def draw_landmarks_on_image(self, rgb_image):
        if self.result.pose_landmarks:
            self.mp_drawer.draw_landmarks(
                rgb_image, self.result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )
    
    def get_distance_to_hip(self):
        l_hip = self._convert_to_ndarray(MPLandmark.LEFT_HIP)
        r_hip = self._convert_to_ndarray(MPLandmark.RIGHT_HIP)
        dist_z = abs(l_hip[2]-r_hip[2])/2
        return dist_z
    
    def get_distance_to_shoulder(self):
        l_shoulder = self._convert_to_ndarray(MPLandmark.LEFT_SHOULDER)
        r_shoulder = self._convert_to_ndarray(MPLandmark.RIGHT_SHOULDER)
        dist_z = abs(l_shoulder[2]-r_shoulder[2])/2
        return dist_z

    def _get_deg_from_landmarks(self):
        l_shoulder = self._convert_to_ndarray(MPLandmark.LEFT_SHOULDER)
        r_shoulder = self._convert_to_ndarray(MPLandmark.RIGHT_SHOULDER)
        axis_z = self._normalize((l_shoulder - r_shoulder))
        if self._vec_length(axis_z) == 0:
            axis_z = np.array((0, -1, 0))
            
        axis_x = np.cross(np.array((0, 0, 1)), axis_z)
        if self._vec_length(axis_x) == 0:
            axis_x = np.array((1, 0, 0))
        
        axis_y = np.cross(axis_z, axis_x)
        rot_matrix = np.array([axis_x, axis_y, axis_z]).transpose()
        r11, r12, r13 = rot_matrix[0]
        r21, r22, r23 = rot_matrix[1]
        r31, r32, r33 = rot_matrix[2]

        theta_x = np.arctan2(r32,r33)
        theta_y = np.arctan2(-r31,np.sqrt(r32**2+r33**2))
        theta_z = np.arctan2(r21,r11)

        return theta_x, theta_y, theta_z
    
    def _convert_to_ndarray(self, landmark_i):
        return np.array([self.landmarks[landmark_i].x, self.landmarks[landmark_i].y, self.landmarks[landmark_i].z])
    
    def _normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm
    
    def _vec_length(self, v: np.array):
        return np.sqrt(sum(i**2 for i in v))
    
    def facing_camera(self):
        """
        returns True if the person is facing the camera
        """
        # check if person if facing towards or away from camera
        try:
            l_shoulder = self._convert_to_ndarray(MPLandmark.LEFT_SHOULDER)
            r_shoulder = self._convert_to_ndarray(MPLandmark.RIGHT_SHOULDER)
            nose = self._convert_to_ndarray(MPLandmark.NOSE)
        except:
            return False
        shoulder_mid = (l_shoulder+r_shoulder)/2
        d = nose-shoulder_mid
        return np.sign(d[2]) == -1

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

async def display_kachakas(kachakas:list[KachakaFrame]):
    image = np.concatenate(([kachaka.cv_img for kachaka in kachakas]), axis=1)
    image = image_resize(image, width=SCREEN_W, height=SCREEN_H)
    return image

def process_object(objects):
    # objects = [[label, roi{x, y, height, width}, score]]
    objects = [[n.roi.x_offset,n.roi.y_offset,n.roi.width,n.roi.height] for n in objects if n.label == 1 and n.score > 0.5]
    if len(objects) > 0:
        cur_max_area = 0
        cur_max_area_i = 0
        for i,(x, y, w, h) in enumerate(objects):
            if h*w > cur_max_area:
                cur_max_area = h*w
                cur_max_area_i = i
        x, y, w, h = objects[cur_max_area_i]
        return x, y, w, h
    
def get_camera_info(client:kachaka_api.KachakaApiClient):
    camera_info = client.get_front_camera_ros_camera_info()
    mtx = np.array(camera_info.K, dtype=float).reshape(3, 3)
    dist = np.array(camera_info.D)
    height = camera_info.height
    width = camera_info.width
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (width, height), 0, (width, height)
    )
    map_x, map_y = cv2.initUndistortRectifyMap(
        mtx, dist, None, new_camera_mtx, (width, height), 5
    )
    return map_x, map_y

def undistort(img, map_x, map_y):
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

def draw_box(img, res:ultralytics.engine.results.Boxes):
    for r in res:
        r:ultralytics.engine.results.Results
        x,y,w,h = [int(n) for n in r.boxes.xywh.numpy()[0]]
        x,y = x-w//2, y-h//2
        conf = r.boxes.conf.numpy()[0]
        if conf > YOLO_CONF_THRE:
            img = cv2.rectangle(img, (x,y), (x+w,y+h), color=GREEN, thickness=2)
            img = cv2.putText(img, f"score:{round(conf,3)}", (x+20, y), FONT, 1, GREEN, 1)
    return img

async def get_map_images(kachakas:KachakaFrame):
    imgs = []
    for k in kachakas:
        png_map = await k.async_client.get_png_map()
        png_map = cv2.imdecode(np.frombuffer(png_map.data, dtype=np.uint8), flags=1)
        imgs.append(png_map)
    imgs = pad_images_to_same_shape(imgs)
    return imgs

def pad_images_to_same_shape(arrays:list[np.ndarray], before=None, after=1, value=0, tie_break=np.floor):
    shapes = np.array([x.shape for x in arrays])
    if before is not None:
        before = np.zeros_like(shapes) + before
    else:
        before = np.ones_like(shapes) - after
    max_size = shapes.max(axis=0, keepdims=True)
    margin = (max_size - shapes)
    pad_before = tie_break(margin * before.astype(float)).astype(int)
    pad_after = margin - pad_before
    pad = np.stack([pad_before, pad_after], axis=2)
    return [np.pad(x, w, mode='constant', constant_values=value) for x, w in zip(arrays, pad)]

async def show_map(kachakas):
    while True:
        imgs = await get_map_images(kachakas)
        cv2.imshow("",np.concatenate(imgs, axis=1))
        cv2.waitKey(1)

async def task_monitor_key_press(tasks:list[asyncio.Task]):
    while True:
        key = await aioconsole.ainput()
        if key.lower() == 'q':
            print("Key 'q' pressed. Terminating all tasks...")
            for task in tasks:
                task.cancel()
            break

@log_function_data
async def object_monitor_key_press(kachakas:list[KachakaFrame]):
    while True:
        key = await aioconsole.ainput()
        if key.lower() == 'q':
            print("Key 'q' pressed. Terminating all tasks...")
            for kachaka in kachakas:
                kachaka.run = False
                await kachaka.async_client.set_robot_stop()
                await kachaka.async_client.set_auto_homing_enabled(True)
            # dir_to_gif()
            break
        elif key.lower() == 'h':
            print("Key 'h' pressed. Returning to home...")
            for kachaka in kachakas:
                kachaka.run = False
                kachaka.sync_client.set_auto_homing_enabled(False)
                kachaka.sync_client.return_home()
            break

async def anext(iterator, default=None):
    try:
        return await iterator.__anext__()
    except StopAsyncIteration:
        if default is None:
            raise
        return default

def mod_radians(r:float):
    # normalize to range[-2pi, 2pi]
    return (r + 2*PI) % (4*PI) - 2*PI

def normalize_radians(r:float):
    # normalize radians to prevent jumps
    return np.arctan2(np.sin(r), np.cos(r))

def radian_subtract(a:float, b:float):
    # difference in radians between a and b, maintaining continuity
    delta = (b - a) % 360
    if delta > 180:
        delta -= 360
    return delta

def get_coords_from_angle(theta:float, r:float = 1) -> tuple[float, float]:
    """given hypotenuse and angle, return (x,y) components
    """
    # theta = theta % (2*PI)
    return np.array((r*math.cos(theta), r*math.sin(theta)))

def filter_outliers_z_score(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    z_scores = np.abs(stats.zscore(data))
    return data[z_scores < threshold]

@log_function_data
def filter_outliers_IQR(data: np.ndarray) -> np.ndarray:
    if len(data) == 0:
        return np.array([])
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)

    # Calculate IQR
    IQR = Q3 - Q1

    # Define bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter data
    return data[(data >= lower_bound) & (data <= upper_bound)]

def dir_to_gif():
    import os
    from PIL import Image
    try:
        dir = "/home/yo/Desktop/kachaka/kachaka-learn/stalk/img/"
        images = [f for f in os.listdir(dir) if f.endswith(".png")]
        images.sort()
        images = [Image.open(os.path.join(dir,f)) for f in images]
        images[0].save('output.gif',
                save_all=True,  # Save all frames
                append_images=images[1:],  # Append the rest of the images
                duration=500,  # Duration between frames in milliseconds
                loop=0)  # Loop forever
    except Exception as e:
        logging.error(e)

class cv2_c:
    black = (0, 0, 0)
    white = (255, 255, 255)
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    yellow = (0, 255, 255)
    cyan = (255, 255, 0)
    magenta = (255, 0, 255)
    gray = (128, 128, 128)
    light_gray = (211, 211, 211)
    dark_gray = (169, 169, 169)
    orange = (0, 165, 255)
    purple = (128, 0, 128)
    brown = (42, 42, 165)
class C:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    RESET = "\033[0m"