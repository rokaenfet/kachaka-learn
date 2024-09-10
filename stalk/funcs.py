import kachaka_api
import time
import asyncio
import cv2
import numpy as np
import keyboard
import mediapipe as mp
import aioconsole
import math
from mediapipe.framework.formats import landmark_pb2
import ultralytics.engine
import ultralytics.engine.results
import matplotlib.pyplot as plt
import math
import sys
import matplotlib.patches as mpatches

from mebow_model import MEBOWFrame
from ultralytics import YOLO
import ultralytics

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
WIN_W = 1280
WIN_H = 720
THRE = 30
YOLO_CONF_THRE = 0.85
lazy_cv2_txt_params = (FONT, 3, GREEN, 3)
MIN_LINEAR_SPEED = 0.1
MIN_ANGULAR_SPEED = 0.1
MAX_LINEAR_SPEED = 1
MAX_ANGULAR_SPEED = 1
MOVE_ANGLE_THRE = 0.03
MOVE_EUCLIDEAN_DIST_THRE = 0.03

SCREEN_NAME = "Fullscreen"
cv2.namedWindow(SCREEN_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(SCREEN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
SCREEN_W, SCREEN_H = cv2.getWindowImageRect(SCREEN_NAME)[2:4]
cv2.destroyWindow(SCREEN_NAME)
SCREEN_W, SCREEN_H = 1280, 720


class KachakaFrame():
    human_detection_result: ultralytics.engine.results.Boxes
    def __init__(self, IP:str, id:int):
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
        self.linear = 0
        self.angular = 0
        self.being_controlled = False
        self.human_found_count = 0
        self.face_found_count = 0
        self.find_face_mode = False
        self.target_near = False

        self.target_pos = None
        self.cv_img = None

        print(f"{C.GREEN}got{C.RESET} media pipe face detector")
        self.face_detector = FaceDetect()

        self.locations = self.get_locations(["start","end"])
        self.nav_i = 0
        self.run = True
        self.run_nav = True
        self.navigating = False

        self.cd = 0

        self.yolo_model = YOLO("yolov8n.pt")

        self.visualize_prev_locations = []
        self.dest_pose = None

        # models
        print(f"{C.GREEN}got{C.RESET} MEBOW model")
        self.mebow_model = MEBOWFrame()
        print(f"{C.GREEN}got{C.RESET} media pipe landmark detector")
        self.mp_landmark_model = MPLandmark()

    async def emergency_stop(self):
        """ detect and perform emergency stop using LiDAR data
        
        scans LiDAR from kachaka and determines if kachaka should perform an emergency stop

        Attibutes:
            lidar_scan: coroutine get_ros_laser_scan()
        """         
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

    async def move(self):
        """change kachaka's linear and angular velocity

        when self.dest_pose is not None:
            imitates client.move_to_pos() by using liner P control. This is to prevent using long-time blocking events
        """
        if self.dest_pose is not None:
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
                print(f"DESTINATION {dest_theta} REACHED")
            await self.async_client.set_robot_velocity(0, angular_speed)
            print(np.rad2deg(dest_theta), np.rad2deg(cur_theta), np.rad2deg(angle_diff))
        else:
            await self.async_client.set_robot_velocity(self.linear, self.angular)

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

    async def human_detection(self):
        """detects human using kachaka's embedded model
        """
        results = self.yolo_model(self.cv_img, verbose=False)[0]
        results = [r for r in results if r.boxes.xywh.numel() > 0]
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
        tuple
            float
                d_linear is the change in linear speed to make w.r.t size of detected human in frame
            float
                d_angular is the change in angular speed to make w.r.t horizontal offset from center
            float
                (horizontal offset from center) / width of screen
            float
                
        """        
        tx, ty, tw, th = self.target_pos
        center_tx, center_ty = tx+tw//2, ty+th//2
        x_r = (WIN_W/2-center_tx)/WIN_W
        area_r = (tw*th)/(WIN_W*WIN_H)
        d_linear, d_angular = (1-area_r)*AUTO_LINEAR_SPEED, x_r*AUTO_ANGULAR_SPEED
        return d_linear, d_angular, x_r, area_r, tx, ty, tw, th
    
    async def adjust(self):
        d_linear, d_angular, x_r, area_r, tx, ty, tw, th = await self._prep_auto_control()
        self.angular = d_angular
        if ty <= THRE:
            self.linear = -area_r*AUTO_LINEAR_SPEED
        elif tw/WIN_W < 0.2 or ty/WIN_H < 0.2 or area_r < 0.4:
            self.linear = area_r*AUTO_LINEAR_SPEED
        else:
            self.linear, self.angular = 0, 0

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
            cv2.putText(self.cv_img, f"ID:{self.id}", (WIN_W-100,40), *lazy_cv2_txt_params)

    async def speak(self, txt:str):
        await self.async_client.speak(txt)

    def get_locations(self, locations:str) -> list[str]:
        return [location for location in self.sync_client.get_locations() if location.name in locations]

    async def navigate(self):
        try:
            while self.run:
                if self.target_found == False:
                    result = await self.async_client.move_to_location(self.locations[self.nav_i].id)
                    if result.success:
                        self.nav_i = (self.nav_i+1)%len(self.locations)
                    else:
                        print(self.error_code[result.error_code])
        except asyncio.CancelledError:
            print(f'Navigation for Kachaka:{self.id} was cancelled.')

    async def short_navigate(self):
        try:
            if self.run:
                result = await self.async_client.move_to_location(self.locations[self.nav_i].id)
                if result.success:
                    self.nav_i = (self.nav_i+1)%len(self.locations)
                else:
                    print(self.error_code[result.error_code])
        except Exception as e:
            print(self.id,":",e)

    async def check_navigate(self):
        """
        returns True if its currently navigating
        """
        is_command_running, get_running_command, get_last_command_result = await self.check_command_states()
        return is_command_running and get_running_command is not None and get_running_command.move_to_location_command is not None

    async def check_command_states(self) -> tuple:
        """
        Return -> bool, [pb2.GetCommandStateResponse, None], (Result, Command)
        """
        is_command_running = await self.async_client.is_command_running()
        get_running_command = await self.async_client.get_running_command()
        get_last_command_result = await self.async_client.get_last_command_result()
        return is_command_running, get_running_command, get_last_command_result
    
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
            center_x, center_y = tx+tw//2, ty+th
            end_x = int(center_x + line_length * np.cos(rads))
            end_y = int(center_y - line_length * np.sin(rads))
            cv2.line(image, (center_x, center_y), (end_x, end_y), BLUE, 3)
            cv2.putText(image, "deg="+str(round(deg,1)), (20,50), *lazy_cv2_txt_params)

    async def mebow_annotate(self, line_length=100):
        # only if there is a result
        deg = self.mebow_model.ori + 90
        await self._draw_orientation_line(deg)

    def _find_deg_from_landmarks(self):
        m = self.mp_landmark_model
        rads, _, _ = m._get_deg_from_landmarks()
        sign = -1 if m.facing_camera() else 1
        deg = sign*rads*180/np.pi
        return deg

    async def mp_landmark_annotate(self, line_length=100):
        m = self.mp_landmark_model
        if m.result.pose_landmarks:
            deg = self._find_deg_from_landmarks()
            await self._draw_orientation_line(deg, self.cv_img)

    async def get_robot_pose(self):
        """
        return:
            type: [tuple[tuple[float, float], float]] -> ((x, y), theta)
        """
        pose = await self.async_client.get_robot_pose()
        return (pose.x, pose.y), pose.theta
    
    async def adjust_to_front(self):
        m = self.mp_landmark_model
        if self.target_found and m.landmarks is not None:
            pose_task = asyncio.create_task(self.get_robot_pose())
            # get distance to target user
            target_deg = self._find_deg_from_landmarks() # +90 for offset
            # z_dist = m.get_distance_to_hip()
            z_dist = m.get_distance_to_shoulder()
            d_step = z_dist / 4 # chord len (dist to travel between 2 vertices on the circumference)
            (pose_x, pose_y), pose_theta  = await pose_task
            pose_deg = np.rad2deg(pose_theta)
            # pose_theta = np.deg2rad(pose_deg)
            # TODO: angular offset in the x-axis between kachaka and camera, will be determined with robot arm later
            camera_to_kachaka_offset = 90-pose_deg # assume camera is placed 90deg against kachaka
            camera_deg = -90 # camera assumed to always face human
            if self.dest_pose is None: # only if dest_pose is empty, to prevent the kachaka robot overloading
                c = 2*math.asin(d_step/(2*z_dist)) # angle to turn
                if target_deg+90 > 0:
                    angle_to_turn = math.pi + c
                    new_pose = pose_theta - angle_to_turn
                else:
                    angle_to_turn = c
                    new_pose = pose_theta+np.pi+angle_to_turn
                self.dest_pose = (pose_x, pose_y, new_pose)
                """
                pose_deg = 0
                2*math.asin(d_step/(2*z_dist))
                """
                # print(f"target_deg:{round(target_deg,1)} | camera_deg:{round(camera_deg,1)} | kachaka_deg:{round(np.rad2deg(pose_theta),1)} | angle_to_turn:{round(np.rad2deg(angle_to_turn),1)}")
                # visualize
                await self._visualize_adjusting_to_front(z_dist, np.deg2rad(target_deg), np.deg2rad(camera_deg), d_step, angle_to_turn, pose_theta)
            
    async def _visualize_adjusting_to_front(self, z_dist, target_rads, camera_rads, step_distance, angle_to_turn, kachaka_theta): # assumes camera is facing towards target
        m = self.mp_landmark_model
        if m.result.pose_landmarks:
            fig,ax = plt.subplots(figsize=(12,8))
            center = (0,0)
            kachaka = (0,z_dist)
            arrow_sca = 40
            ax.add_artist(plt.Circle(center, z_dist/13, color="green", zorder=5, label="human")) # circle center
            ax.add_artist(plt.Circle(center, z_dist, color="blue", fill=False, linestyle="dashed", zorder=3)) # circumference
            ax.add_artist(plt.Circle(kachaka, z_dist/13, color="black", zorder=3)) #kachaka obj
            # kachaka pov
            kachaka_arrow = ax.arrow(kachaka[0], kachaka[1], z_dist/2 * np.cos(kachaka_theta), z_dist/2 * np.sin(kachaka_theta), width=z_dist/arrow_sca, shape="full", color="black", linestyle="", label="Kachaka")
            ax.text(kachaka[0]+z_dist/2, kachaka[1], f"Kachaka:{round(np.rad2deg(kachaka_theta),1)}°", fontsize=12, ha='center', color='black')
            # camera pov
            camera_arrow = ax.arrow(kachaka[0], kachaka[1], z_dist/2 * np.cos(camera_rads), z_dist/2 * np.sin(camera_rads), width=z_dist/arrow_sca, shape="full", color="yellow", linestyle="", label="camera")
            # past target povs
            self.visualize_prev_locations.append(((0,-z_dist*np.cos(target_rads)),(0,-z_dist*np.sin(target_rads))))
            if len(self.visualize_prev_locations) > 5:
                self.visualize_prev_locations.pop(0)
            for i,n in enumerate(self.visualize_prev_locations):
                (x,dx),(y,dy) = n
                if i == len(self.visualize_prev_locations)-1:
                    # most recent line to be draw with different colour and full alpha
                    ax.arrow(x, y, dx, dy, color="red", linestyle="solid", lw=1, zorder=4, width=z_dist/arrow_sca, shape="full", label="present human")
                else:
                    ax.arrow(x, y, dx, dy, color="orange", linestyle="solid", lw=1, zorder=4, alpha=(255/5 * i)/255, width=z_dist/arrow_sca, label=f"[{i}] past human")
            # kachaka destination
            destination_arrow = ax.arrow(*kachaka, step_distance*math.cos(angle_to_turn), step_distance*math.sin(angle_to_turn), color="purple", linestyle="solid", lw=1, zorder=4, width=z_dist/arrow_sca, shape="full", label="destination")
            # draw some info text on center
            ax.text(0, z_dist/6, f'Radius = {round(z_dist,5)}\nAngle = {round(np.rad2deg(target_rads),1)}°', 
                    fontsize=12, ha='center', color='purple')
            c = 1.2
            ax.set_xlim(-z_dist*c, z_dist*c)
            ax.set_ylim(-z_dist*c, z_dist*c)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.legend()
            fig.savefig("stalk/visualize.png", bbox_inches="tight")


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
        l_shoulder = self._convert_to_ndarray(MPLandmark.LEFT_SHOULDER)
        r_shoulder = self._convert_to_ndarray(MPLandmark.RIGHT_SHOULDER)
        nose = self._convert_to_ndarray(MPLandmark.NOSE)
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
    # print(SCREEN_W, SCREEN_H)
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

def pad_images_to_same_shape(imgs:list):
    max_w, max_h = 0, 0
    for img in imgs:
        w, h, _ = img.shape
        max_w, max_h = max(max_w, w), max(max_h, h)
    for i,img in enumerate(imgs):
        w, h, _ = img.shape
        pad_w = max_w - w if max_w > w else 0
        pad_h = max_h - h if max_h > h else 0
        padded_img = np.pad(img, ((pad_h//2, pad_h//2-pad_h%2), (pad_w//2, pad_w//2-pad_w%2), (0, 0)), mode='constant', constant_values=0)
        imgs[i] = padded_img
    return imgs

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

async def object_monitor_key_press(kachakas:list[KachakaFrame]):
    while True:
        key = await aioconsole.ainput()
        if key.lower() == 'q':
            print("Key 'q' pressed. Terminating all tasks...")
            for kachaka in kachakas:
                kachaka.run = False
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
    return r%(math.pi*2)

class C:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    RESET = "\033[0m"