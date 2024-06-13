import numpy as np
import enum as enum
import opencv_camera
import pathFinder
import Translator


class phases(enum):
    Startup_phase = 1,
    Calibration_phase = 2,
    robot_phase = 3,
    emergency_phase = 4

def main():
    current_phase = phases.Startup_phase
    running = True
    while(running):
        if current_phase == phases.Startup_phase:
            startup
            current_phase = phases.robot_phase

        elif current_phase == phases.Calibration_phase:
            calibration

        elif current_phase == phases.robot_phase:
            run_robot

        elif current_phase == phases.emergency_phase:
            emergency_stop






def startup():
    print("numse")
    opencv_camera.run_image

def calibration():
    print("implement calibration")


def run_robot():
    print("implement robot running")

def emergency_stop():
    print("implement emergency phase")