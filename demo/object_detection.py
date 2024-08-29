import math
import kachaka_api
from kachaka_api.util.vision import OBJECT_LABEL, get_bbox_drawn_image
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from IPython.display import Image, clear_output
import cv2
import asyncio
import numpy as np
import time
KACHAKA_IP = "192.168.118.158:26400"
FONT = cv2.FONT_HERSHEY_PLAIN
WHITE = (255,255,255)

async def object_detection(client:kachaka_api.aio.KachakaApiClient):
    stream_i = client.front_camera_ros_compressed_image.stream()
    stream_d = client.object_detection.stream()
    while True:
        st = time.time()
        image, (header, objects) = await asyncio.gather(anext(stream_i), anext(stream_d))
        pil_img = get_bbox_drawn_image(image, objects)
        clear_output(wait=True)
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.putText(cv_img, f"fps:{round(1/(time.time()-st))}", (20,80), FONT, 3, WHITE, 2)
        # for object in objects:
        #     cv2.putText(cv_img, f"{OBJECT_LABEL[object.label]}: {round(object.score,3)}", (200,200),
        #                 FONT, 3, WHITE, 2)
        #     print(f"{OBJECT_LABEL[object.label]}, score={object.score:.2f}")
        cv2.imshow("", cv_img)
        cv2.waitKey(1)

async def main():
    client = kachaka_api.aio.KachakaApiClient(KACHAKA_IP)
    await client.set_manual_control_enabled(True)
    await object_detection(client)

asyncio.run(main())