import kachaka_api
import matplotlib.pyplot as plt
import numpy as np
import cv2
import asyncio
# from IPython.display import Image, display
# ip changes everytime check with kachacar
KACHAKA_HOST = "192.168.118.158"
# grpc port
KACHAKA_PORT = 26400


async def main():
    client = kachaka_api.aio.KachakaApiClient(f"{KACHAKA_HOST}:{KACHAKA_PORT}")
    await client.set_manual_control_enabled(True) # If true only moves with command
    
    serial_num = await client.get_robot_serial_number()
    version_num = await client.get_robot_version()
    current_pos = await client.get_locations()
    default_pos_id = await client.get_default_location_id()
    shelf_id = await client.get_shelves()
    moving_shelf_id = await client.get_moving_shelf_id()
    current_command = await client.get_running_command()
    is_command_running = await client.is_command_running()
    last_command_res = await client.get_last_command_result()
    command_state = await client.get_command_state()
    command_history = await client.get_history_list()
    pose = await client.get_robot_pose()
    map = await client.get_png_map()
    print(map.name)
    print(map.resolution, map.width, map.height)
    print(map.origin)
    print(map.data)
    cv2.imshow("map", map.data)
    imu = await client.get_ros_imu()
    odometry = await client.get_ros_odometry()
    front_cam_ros_info = await client.get_front_camera_ros_camera_info()

    await client.speak("displaying my basic info")


if __name__ == "__main__":
    asyncio.run(main())