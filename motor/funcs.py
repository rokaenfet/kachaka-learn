# funcs to be used in MIT mode
from DM_CAN import MotorControl, Motor, Control_Type
import time
import numpy as np

"""
MotorControl.controlMIT(DM_Motor:Motor, kp:p_gain, kd:d_gain, q:pos(rads), dq:vel(rads/s), tau:torque(Nm))
"""

DEFAULT_DELAY = 0.002 #ms
POSITION_OFFSET_THRE = 0.0349066 #rads

class MITController():
    def __init__(self, dm:MotorControl):
        self.dm = dm

    def move_to_location(self, m:Motor, kp:float, kd:float, target_q:float):
        cur_q = m.getPosition()
        count = 0
        while abs(cur_q-target_q) > POSITION_OFFSET_THRE:
            cur_q = m.getPosition()
            self.position_control(m, kp, kd, target_q)
            count += 1
            if count > 100:
                print_fail(f"Failed to reach pos:{target_q} after {count} attempts")
                break

    def velocity_control(self, m:Motor, kd:float, dq:float):
        self.dm.control_delay(
            DM_Motor=m, 
            kp=0, 
            kd=kd, 
            q=0, 
            dq=dq, 
            tau=0,
            delay=DEFAULT_DELAY
            )
        
    def torque_control(self, m:Motor, tau:float):
        self.dm.control_delay(m, 0, 0, 0, 0, tau, delay=DEFAULT_DELAY)

    def position_control(self, m:Motor, kp:float, kd:float, q:float):
        self.dm.control_delay(m, kp, kd, q, 0, 0, delay=DEFAULT_DELAY)

    def enable(self, m:Motor):
        """
        checks motor is in MIT mode and has been enabled
        """
        self.dm.enable(m)
        if m.NowControlMode == Control_Type.MIT:
            print_success("Motor has been enabled!", m)
            return True
        else:
            print_fail("Motor is not in MIT mode", m)
            return False
        
    def disable(self, m:Motor):
        self.dm.disable(m)
        print_success("Motor has been DISABLED...")
        time.sleep(DEFAULT_DELAY)

    def reset(self, m:Motor):
        self.dm.enable(m)
        time.sleep(DEFAULT_DELAY)
        self.dm.control_delay(m, 0, 0, 0, 0, 0, delay=DEFAULT_DELAY)
        time.sleep(DEFAULT_DELAY)
        self.dm.disable(m)
        time.sleep(DEFAULT_DELAY)
        print_success("Motor RESET", m)

    def set_zero_position(self, m:Motor):
        self.dm.set_zero_position(m)
        time.sleep(DEFAULT_DELAY)


def deg2rads(deg:float) -> float:
    return deg*np.pi/180

def rads2deg(rads:float) -> float:
    return rads*180/np.pi

def get_motor_state(m:Motor) -> tuple[float, float, float]:
    """
    param:
        [Motor]:m
    return
        tuple[float, float, float]:pos, vel, torque
    """
    return m.getPosition(), m.getVelocity(), m.getTorque()

def print_success(txt:str, m: Motor | None = None) -> str:
    if m is not None:
        print(f"ID[{m.SlaveID}] {c.OKGREEN}Success!{c.ENDC} {txt}")
    else:
        print(f"{c.OKGREEN}Success!{c.ENDC} {txt}")

def print_fail(txt:str, m: Motor | None = None) -> str:
    if m is not None:
        print(f"ID[{m.SlaveID}] {c.FAIL}FAIL{c.ENDC} {txt}")
    else:
        print(f"{c.FAIL}FAIL!{c.ENDC} {txt}")

class c:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
