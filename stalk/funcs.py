import kachaka_api
import time
import asyncio
from kachaka_api.util.vision import OBJECT_LABEL, get_bbox_drawn_image
import matplotlib.patches as patches
import cv2
import numpy as np
import keyboard
import mediapipe as mp
import aioconsole

FONT = cv2.FONT_HERSHEY_PLAIN
WHITE = (255,255,255)
RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)
BLACK = (0,0,0)
LINEAR_SPEED = 0.3
AUTO_LINEAR_SPEED = 1
ANGULAR_SPEED = 1
AUTO_ANGULAR_SPEED = 2
EMERGENCY_STOP_DISTANCE = 0.2
WIN_W = 1280
WIN_H = 720
THRE = 30
lazy_cv2_txt_params = (FONT, 3, WHITE, 2)

class KachakaFrame():
    def __init__(self, IP:str, id:int):
        self.id = id
        self.sync_client = kachaka_api.KachakaApiClient(IP)
        self.async_client = kachaka_api.aio.KachakaApiClient(IP)
        print(f"Kachaka ID:{self.id} has {C.GREEN}connected{C.RESET} with address: {IP}")
        self.stream_i = self.async_client.front_camera_ros_compressed_image.stream()
        self.stream_d = self.async_client.object_detection.stream()
        print(f"{C.GREEN}got{C.RESET} stream")
        image = self.sync_client.get_front_camera_ros_compressed_image()
        self.undistort_map = get_camera_info(self.sync_client)
        print(f"{C.GREEN}got{C.RESET} camera info")
        self.error_code = self.sync_client.get_robot_error_code()
        print(f"{C.GREEN}got{C.RESET} error code")
        self.need_to_emergency_stop = False
        self.target_found = False
        self.sync_client.set_manual_control_enabled(True)
        self.sync_client.set_auto_homing_enabled(False)
        self.linear = 0
        self.angular = 0
        self.being_controlled = False
        self.human_found_count = 0
        self.face_found_count = 0
        self.find_face_mode = False
        self.target_near = False

        self.target_pos = None
        self.cv_img = None

        self.face_detector = FaceDetect()

        self.locations = self.get_locations(["start","end"])
        self.nav_i = 0
        self.run = True

        self.cd = 0

    async def process_kachaka(self):
        st = time.time()
        self.linear, self.angular = 0, 0
        await self.move() # set speed to 0,0
        await self.control() # key and auto control
        await self.move() # move with the updated speed
        await self.emergency_stop() # detect for emergency stops
        await self.human_detection() # detect human
        self.annotate(st) # annotate img

    async def emergency_stop(self):
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
        if self.run:
            await self.async_client.set_robot_velocity(self.linear, self.angular)

    async def get_image_from_camera(self):
        image = await self.stream_i.__anext__()
        self.cv_img = cv2.imdecode(np.frombuffer(image.data, dtype=np.uint8), flags=1)
        return self.cv_img

    async def human_detection(self):
        image, (header, objects) = await asyncio.gather(anext(self.stream_i), anext(self.stream_d))
        self.cv_img = cv2.imdecode(np.frombuffer(image.data, dtype=np.uint8), flags=1) # roscompressed to img
        objects = [n for n in objects if n.label==1]
        self.cv_img = undistort(self.cv_img, *self.undistort_map)
        if len(objects) > 0:
            self.cv_img = draw_box(self.cv_img, objects)
            self.target_pos = process_object(objects) #x, y, w, h
            if self.target_pos:
                cv2.putText(self.cv_img, "X", (self.target_pos[0]+self.target_pos[2]//2, 
                                          self.target_pos[1]+self.target_pos[3]//2), *lazy_cv2_txt_params)
                self.target_found = True
            else:
                self.target_found = False
        else:
            self.target_found = False

    async def _prep_auto_control(self):
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
            cv2.putText(self.cv_img, f"ID:{self.id}", (WIN_W-20,20), *lazy_cv2_txt_params)

    async def speak(self, txt:str):
        await self.async_client.speak(txt)

    def get_locations(self, locations:str):
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


class FaceDetect():
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detector = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

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
                
async def display_kachakas(kachakas:list[KachakaFrame]):
    imgs = [cv2.resize(n.cv_img, (640, 360)) for n in kachakas]
    imgs = np.concatenate((imgs), axis=1)
    cv2.imshow("", imgs)
    cv2.waitKey(1)

def move(client:kachaka_api.KachakaApiClient, linear:float, angular:float):
    client.set_robot_velocity(linear=linear, angular=angular)

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

def draw_box(img, objects):
    for obj in objects:
        x, y, w, h = (obj.roi.x_offset, obj.roi.y_offset, obj.roi.width, obj.roi.height,)
        img = cv2.rectangle(img, (x,y), (x+w,y+h), color=GREEN, thickness=2)
        img = cv2.putText(img, f"score:{round(obj.score,3)}", (x+20, y), FONT, 1, GREEN, 1)
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

class C:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    RESET = "\033[0m"