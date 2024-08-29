import kachaka_api
import time
import asyncio
from kachaka_api.util.vision import OBJECT_LABEL, get_bbox_drawn_image
import matplotlib.patches as patches
import cv2
import numpy as np
import keyboard

FONT = cv2.FONT_HERSHEY_PLAIN
WHITE = (255,255,255)
RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)
BLACK = (0,0,0)
LINEAR_SPEED = 0.3
AUTO_LINEAR_SPEED = 0.2
ANGULAR_SPEED = 1
AUTO_ANGULAR_SPEED = 0.1
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
        print(f"Kachaka ID:{self.id} has connected with address: {IP}")
        self.stream_i = self.async_client.front_camera_ros_compressed_image.stream()
        self.stream_d = self.async_client.object_detection.stream()
        print(f"stream got")
        image = self.sync_client.get_front_camera_ros_compressed_image()
        self.undistort_map = get_camera_info(self.sync_client)
        print(f"camera info got")
        self.need_to_emergency_stop = False
        self.target_found = False
        self.sync_client.set_manual_control_enabled(True)
        self.linear = 0
        self.angular = 0
        self.being_controlled = False

        self.target_pos = None
        self.cv_img = None

    async def process_kachaka(self):
        st = time.time()
        self.linear, self.angular = 0, 0
        await self.move() # set speed to 0,0
        await self.control() # key and auto control
        await self.move() # move with the updated speed
        await self.emergency_stop() # detect for emergency stops
        await self.human_detection() # detect human
        # annotate img
        self.annotate(st)

    
    async def emergency_stop(self):
        lidar_scan = await self.async_client.get_ros_laser_scan()
        self.nearest_scan_dist = min([dist for dist in lidar_scan.ranges if dist > 0])
        if self.nearest_scan_dist < EMERGENCY_STOP_DISTANCE:
            self.need_to_emergency_stop = True
            print(f"ID:{self.id} has prevented a crash!")
            # self.sync_client.speak("move")
            self.sync_client.set_robot_velocity(0, 0)
        elif self.need_to_emergency_stop == True:
            self.need_to_emergency_stop = False
            # self.sync_client.speak("ty")
            print(f"ID:{self.id} has resumed moving!")

    async def move(self):
        await self.async_client.set_robot_velocity(self.linear, self.angular)

    async def human_detection(self):
        image, (header, objects) = await asyncio.gather(anext(self.stream_i), anext(self.stream_d))
        self.cv_img = cv2.imdecode(np.frombuffer(image.data, dtype=np.uint8), flags=1) # roscompressed to img
        self.cv_img = undistort(self.cv_img, *self.undistort_map) # undistort
        self.cv_img = draw_box(self.cv_img, objects) # draw bounding box
        if len(objects) > 0:
            self.target_pos = process_object(objects) #x, y, w, h
            if self.target_pos:
                cv2.putText(self.cv_img, "X", (self.target_pos[0]+self.target_pos[2]//2, 
                                          self.target_pos[1]+self.target_pos[3]//2), *lazy_cv2_txt_params)
                self.target_found = True
            else:
                self.target_found = False
        else:
            self.target_found = False
    
    async def adjust(self):
        tx, ty, tw, th = self.target_pos
        center_tx, center_ty = tx+tw//2, ty+th//2
        # horizontal adjustment
        self.angular = np.sign(WIN_W//2-center_tx)*AUTO_ANGULAR_SPEED
        # distance adjustment
        if ty <= THRE:
            self.linear -= AUTO_LINEAR_SPEED
        elif th <= WIN_H//3.5:
            self.linear += AUTO_LINEAR_SPEED

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

    def annotate(self, st:float):
        cv2.putText(self.cv_img, f"fps:{round(1/(time.time()-st))}", (20, 80), *lazy_cv2_txt_params)
        cv2.putText(self.cv_img, f"{round(self.nearest_scan_dist, 3)}", (20, 140), *lazy_cv2_txt_params)
        cv2.putText(self.cv_img, f"ID:{self.id}", (WIN_W-20,20), *lazy_cv2_txt_params)

async def display_kachakas(kachakas):
    highlighted_imgs = [n.cv_img+10 if n.being_controlled else n.cv_img for n in kachakas]
    resized_imgs = [cv2.resize(n, (640, 360)) for n in highlighted_imgs]
    disp_img = np.concatenate((resized_imgs), axis=1)
    cv2.imshow("", disp_img)
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