import numpy as np
from enum import Enum
import opencv_camera
from autodrive import calibration_move
import pathFinder
import Translator


class phases(Enum):
    Startup_phase = 1,
    Calibration_phase = 2,
    robot_phase = 3,
    emergency_phase = 4

def main():
    current_phase = phases.Startup_phase
    running = True
    while(running):
        if current_phase == phases.Startup_phase:
            startup()
            current_phase = phases.robot_phase

        elif current_phase == phases.Calibration_phase:
            pass

        elif current_phase == phases.robot_phase:
            run_robot()

        elif current_phase == phases.emergency_phase:
            emergency_stop()


def startup():
    opencv_camera.run_image


def calibration(first_frame, second_frame): #first_position og second er ikke rigtige frames
    # skal erstattes af rigtige frames
    opencv_camera.detect_Objects(first_frame)
    first_triangle, first_points = opencv_camera.find_triangle(first_frame)
    a1, b1, first_tip_of_tri = opencv_camera.find_abc(first_points)
    #Giver mig det første punkt C i spidsen af trekanten
    cell_width = opencv_camera.cell_width
    second_triangle, second_points = opencv_camera.find_triangle(second_frame)
    a2, b2, second_tip_of_tri = opencv_camera.find_abc(second_points)
    #giver mig andet punkt C i spidsen af trekanten

    calibration_difference = cell_width / ((first_tip_of_tri[0] - first_tip_of_tri[1]) - (second_tip_of_tri[0] - second_tip_of_tri[1]))
    abs(calibration_difference)
    print(calibration_difference)
    return calibration_difference


#def calibration():
    #print("Calibrating...")


def run_robot_calibration():
    firstframe = opencv_camera.take_picture()
    calibration_move()
    secondframe = opencv_camera.take_picture()
    calibration(firstframe, secondframe, 20)


def run_robot():
    print("implement robot running")


def emergency_stop():
    print("implement emergency phase")