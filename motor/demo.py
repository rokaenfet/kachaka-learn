import serial
import time
import math
import numpy as np
import argparse
import sys
from DM_CAN import *
from funcs import *

BAUDRATE = 921600
SLAVE_ID = 0x03
MASTER_ID = 0x13
DEVICE = DM_Motor_Type.DM4310

# COM5, /dev/tty5, /dev/ttyACM0
parser = argparse.ArgumentParser()
parser.add_argument("--port","-p",help="try COM[n] or /dev/ttyS[n]",type=str)
# load args
args = parser.parse_args()

# open serial port
try:
    serial_device = serial.Serial(args.port, BAUDRATE, timeout=0.5)
    print_success(f"Opened port at {args.port} with Baudrate:{BAUDRATE}")
except Exception as e:
    print_fail(f"Could not open port at {args.port}")
    print(e)
    sys.exit()
    
# load motors
motor = Motor(DEVICE, SLAVE_ID, MASTER_ID)
print(f"{c.OKBLUE}Success!{c.ENDC} Init Motor object with model:{DEVICE=}, motor ID:{SLAVE_ID}, host ID:{MASTER_ID}")
motor_control = MotorControl(serial_device)
motor_control.addMotor(motor)
MITControl = MITController(motor_control)

# enable device
MITControl.reset(motor)
MITControl.enable(motor)
MITControl.set_zero_position(motor)

try:
    while True:
        target_q = 2*np.pi*(time.time()%60)/60
        MITControl.position_control(motor, 30, 0.3, -target_q)
        print(motor.getPosition())
except KeyboardInterrupt:
    pass

# disable device
MITControl.reset(motor)

# close device
serial_device.close()