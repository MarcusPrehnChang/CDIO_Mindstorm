import math

import numpy as np
from enum import Enum
import opencv_camera
import server

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
    while (running):
        if current_phase == phases.Startup_phase:
            startup()
            current_phase = phases.Calibration_phase

        elif current_phase == phases.Calibration_phase:
            run_robot_calibration()
            run_robot_calibration_angle()
            current_phase = phases.Robot_phase

        elif current_phase == phases.Robot_phase:
            run_robot()

        elif current_phase == phases.emergency_phase:
            emergency_stop()


def startup():
    global robot
    global server_socket
    robot, server_socket = server.startup_sequence()


def calibration_distance(first_frame, second_frame):
    cell_width, cell_height, bounding_box = opencv_camera.find_box(first_frame)
    first_triangle, first_points, contour = opencv_camera.find_triangle(first_frame)

    a1, b1, first_tip_of_tri = opencv_camera.find_abc(first_points)
    print(first_tip_of_tri)

    second_triangle, second_points, contour = opencv_camera.find_triangle(second_frame)
    a2, b2, second_tip_of_tri = opencv_camera.find_abc(second_points)

    first_tip_difference = first_tip_of_tri[0] - first_tip_of_tri[1]
    second_tip_difference = second_tip_of_tri[0] - second_tip_of_tri[1]

    calibration_difference = abs(cell_width / (first_tip_difference - second_tip_difference))

    return calibration_difference


def calculate_turn(first_frame, second_frame):
    # Detect objects first frame
    opencv_camera.detect_Objects(first_frame)

    # Find Triangle and Vector for first frame
    triangle1, points1, contour1 = opencv_camera.find_triangle(first_frame)
    vec1 = opencv_camera.get_orientation(first_frame, points1)

    # Detect objects second frame
    opencv_camera.detect_Objects(second_frame)

    # Find Triangle and Vector for second frame
    triangle2, points2, contour2 = opencv_camera.find_triangle(second_frame)
    vec2 = opencv_camera.get_orientation(second_frame, points2)

    # Calculate turn angle between two vectors
    # Vector Product
    vector_product = (vec1[0] * vec2[0]) + (vec1[1] * vec2[1])

    # Vector distance
    vec1_distance = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
    vec2_distance = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2)

    # Vector Angle
    true_vector_angle = math.degrees(math.acos(vector_product / (vec1_distance * vec2_distance)))

    print("True Vector Angle0", true_vector_angle)

    calibration_difference = abs(90 / true_vector_angle)

    print("Calibration difference: ", calibration_difference)

    return calibration_difference


def calibration_turn(f1_left, f2_left, f1_right, f2_right):
    calibration_left = calculate_turn(f1_left, f2_left)
    calibration_right = calculate_turn(f1_right, f2_right)

    return calibration_left, calibration_right


def get_robot_info():
    vector_list, robot_heading = opencv_camera.get_info_from_camera()
    robot_heading = str(robot_heading)
    vector_list = str(vector_list)
    square_size = str(20)
    return robot_heading, vector_list, square_size


def run_robot_calibration():
    first_frame, second_frame = server.run_calibration_sequence(robot)
    calibration_difference = calibration_distance(first_frame, second_frame)
    server.send_message("calibration done", robot)
    server.receive_message(robot)
    server.send_message(str(calibration_difference), robot)
    server.receive_message(robot)


def run_robot_calibration_angle():
    f1_left, f2_left, f1_right, f2_right = server.run_calibration_angle_sequence(robot)
    calibration_left, calibration_right = calibration_turn(f1_left, f2_left, f1_right, f2_right)
    server.send_message("calibration done", robot)
    if server.receive_message(robot) == "Received":
        server.send_message(str(calibration_left), robot)
        if server.receive_message(robot) == "Received":
            server.send_message(str(calibration_right), robot)
            if server.receive_message(robot) == "Received":
                server.send_message("Done with calibration", robot)
                if server.receive_message(robot) == "Done applying angles":
                    server.send_message("Received", robot)



def run_robot():
    server.send_message("robot phase", robot)
    message = server.receive_message(robot)
    if message.lower().strip() == "received robot phase":
        robot_heading, vector_list, square_size = get_robot_info()
        print("run robot vectorlist: ", vector_list)
        vector_list = eval(vector_list)
        vector_list = [vector_list]
        robot_heading = eval(robot_heading)
        square_size = int(square_size)
        iterator = 0
        server.start_of_run_sequence(str(robot_heading), str(vector_list[iterator]), str(square_size), robot)
        while iterator != len(vector_list):
            iterator += 1
            server.run_sequence(str(vector_list[iterator]), str(square_size), robot)
            if(iterator != len(vector_list)):
                server.send_message("continue", robot)
        server.send_message("run is done", robot)
        server.receive_message(robot)


def emergency_stop():
    print("implement emergency phase")


main()
