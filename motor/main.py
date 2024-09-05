from DM_CAN import *
import serial
import time
import math
import numpy as np
import argparse

parser = argparse.ArgumentParser(
)
parser.add_argument(
    "--port",
    "-p",
    choices=["COM5","/dev/ttyS5"],
    help="try COM[n] or /dev/ttyS[n]",
    type=str
)
args = parser.parse_args()

PORT = args.port
BAUDRATE = 921600
SLAVE_ID = 0x03
MASTER_ID = 0x13
serial_device = serial.Serial(PORT, BAUDRATE, timeout=0.5)
motor = Motor(DM_Motor_Type.DM4310, SLAVE_ID, MASTER_ID)
motor_control = MotorControl(serial_device)
motor_control.addMotor(motor)

if motor_control.switchControlMode(motor,Control_Type.POS_VEL):
    print("switch POS_VEL success")
print("sub_ver:",motor_control.read_motor_param(motor,DM_variable.sub_ver))
print("Gr:",motor_control.read_motor_param(motor,DM_variable.Gr))

# if motor_control.change_motor_param(motor,DM_variable.KP_APR,54):
#     print("write success")
print("PMAX:",motor_control.read_motor_param(motor,DM_variable.PMAX))
print("MST_ID:",motor_control.read_motor_param(motor,DM_variable.MST_ID))
print("VMAX:",motor_control.read_motor_param(motor,DM_variable.VMAX))
print("TMAX:",motor_control.read_motor_param(motor,DM_variable.TMAX))
motor_control.save_motor_param(motor)
motor_control.enable(motor)
# motor_control.enable_old(motor, Control_Type.POS_VEL)
i=0
prev_pos, prev_vel, prev_torque = None, None, None
while i<10000:
    i=i+1
    # motor_control.control_pos_force(motor, 10, 1000,100)
    # motor_control.control_Vel(motor, q*5)
    txt = ""
    target = math.sin(time.time())*50
    print(target)

    motor_control.control_Pos_Vel(motor,target,30)

    cur_pos = motor.getPosition()
    if cur_pos != prev_pos:
        txt += f"POS: {cur_pos}"
    prev_pos = cur_pos
    cur_vel = motor.getVelocity()
    if cur_vel != prev_vel:
        txt += f"VEL: {cur_vel}"
    prev_vel = cur_vel
    cur_torque = motor.getTorque()
    if cur_torque != prev_torque:
        txt += f"TORQUE: {cur_torque}"
    prev_torque = cur_torque
    if len(txt) > 0:
        print(f"{i} | {txt}")
    # motor_control.controlMIT(Motor2, 35, 0.1, 8*q, 0, 0)

    time.sleep(0.01)

serial_device.close()