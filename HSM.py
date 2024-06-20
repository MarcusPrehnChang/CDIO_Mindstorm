import numpy as np
from enum import Enum
import opencv_camera
import server
from autodrive import calibration_move
import pathFinder
import Translator

robot = None
server_socket = None


class phases(Enum):
    Startup_phase = 1,
    Calibration_phase = 2,
    Robot_phase = 3,
    emergency_phase = 4


def main():
    current_phase = phases.Startup_phase
    running = True
    while(running):
        if current_phase == phases.Startup_phase:
            startup()
            current_phase = phases.Calibration_phase

        elif current_phase == phases.Calibration_phase:
            run_robot_calibration()
            print("changing to robot phase")
            current_phase = phases.Robot_phase

        elif current_phase == phases.Robot_phase:
            run_robot()

        elif current_phase == phases.emergency_phase:
            emergency_stop()


def startup():
    global robot
    global server_socket
    robot, server_socket = server.startup_sequence()


def calibration(first_frame, second_frame):

    opencv_camera.detect_Objects(first_frame)
    first_triangle, first_points = opencv_camera.find_triangle(first_frame)
    print("first points: " + str(first_points))
    print("first triangle: " + str(first_triangle))
    a1, b1, first_tip_of_tri = opencv_camera.find_abc(first_points)
    print(first_tip_of_tri)

    cell_width = opencv_camera.cell_width

    opencv_camera.detect_Objects(second_frame)
    second_triangle, second_points = opencv_camera.find_triangle(second_frame)
    a2, b2, second_tip_of_tri = opencv_camera.find_abc(second_points)

    first_tip_difference = first_tip_of_tri[0] - first_tip_of_tri[1]
    second_tip_difference = second_tip_of_tri[0] - second_tip_of_tri[1]
    print(str(first_tip_difference))
    print(str(second_tip_difference))
    print(str(cell_width))
    calibration_difference = abs(cell_width / (first_tip_difference - second_tip_difference))


    print("Calibration function, calibration difference " + str(calibration_difference))
    return calibration_difference


def get_robot_info():
    vector_list, robot_heading = opencv_camera.get_info_from_camera()
    robot_heading = str(robot_heading)
    vector_list = str(vector_list)
    square_size = str(20)
    return robot_heading, vector_list, square_size


def run_robot_calibration():
    first_frame, second_frame = server.run_calibration_sequence(robot)
    calibration_difference = calibration(first_frame, second_frame)
    server.send_message("calibration done", robot)
    server.receive_message(robot)
    server.send_message(str(calibration_difference), robot)
    server.receive_message(robot)



def run_robot():
    server.send_message("robot phase", robot)
    server.receive_message(robot)
    robot_heading, vector_list, square_size = get_robot_info()
    print("run robot vectorlist: ", vector_list)
    vector_list = eval(vector_list)
    vector_list = [vector_list]
    robot_heading = eval(robot_heading)
    square_size = int(square_size)
    iterator = 0
    server.start_of_run_sequence(str(robot_heading), str(vector_list[iterator]), str(square_size), robot)
    iterator += 1
    while True:
        server.run_sequence(str(vector_list[iterator]), str(square_size), robot)
        iterator += 1


def emergency_stop():
    print("implement emergency phase")

main()