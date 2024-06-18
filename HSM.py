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

'''
def calibration(first_position, second_position): #first_position og second er ikke rigtige frames
    # skal erstattes af rigtige frames

    first_triangle, first_point = opencv_camera.find_triangle(first_position)
    a, b, first_tip_of_tri = opencv_camera.find_abc(first_point)
    #Giver mig det første punkt C i spidsen af trekanten

    run_robot_calibration()

    second_triangle, second_point = opencv_camera.find_triangle(second_position)
    a2, b2, second_tip_of_tri = opencv_camera.find_abc(second_point)
    #giver mig andet punkt C i spidsen af trekanten

    first_tip_of_tri - second_tip_of_tri/ #størrelse af square = mængden af squares rykket

    return None
'''


def calibration():
    print("Calibrating...")


def run_robot_calibration():
    print("drive2cm and turn 90")


def calib_turn():
    print("turn 90")


def run_robot():
    print("implement robot running")


def emergency_stop():
    print("implement emergency phase")