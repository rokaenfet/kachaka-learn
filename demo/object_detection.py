import math
import kachaka_api
from kachaka_api.util.vision import OBJECT_LABEL, get_bbox_drawn_image
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import asyncio
import numpy as np
import time

import os
from pathlib import Path
import PyQt5

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.fspath(
    Path(PyQt5.__file__).resolve().parent / "Qt5" / "plugins"
)

import cv2

KACHAKA_IP = "192.168.118.159:26400"
FONT = cv2.FONT_HERSHEY_PLAIN
WHITE = (255,255,255)

async def anext(iterator, default=None):
    try:
        return await iterator.__anext__()
    except StopAsyncIteration:
        if default is None:
            raise
        return default

async def object_detection(client:kachaka_api.aio.KachakaApiClient):
    stream_i = client.front_camera_ros_compressed_image.stream()
    stream_d = client.object_detection.stream()
    while True:
        st = time.time()
        image, (header, objects) = await asyncio.gather(anext(stream_i), anext(stream_d))
        pil_img = get_bbox_drawn_image(image, objects)
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