import socket
import threading

import opencv_camera
from opencv_camera import get_info_from_camera as get_inf

stop_flag = False


def run_server():
    # Get hostname and port
    host = "192.168.23.184"
    port = 5000

    # Get the instance of socket and start listening on host and port.
    server_socket = socket.socket()
    server_socket.bind((host, port))

    # Max accepting one connection at once.
    server_socket.listen(1)

    # When connect is accepted, save address and conn in variables
    robot, address = server_socket.accept()

    return robot, address, server_socket


def send_message(message, conn):
    print("Sending message: " + str(message))
    conn.send(message.encode())


def receive_message(conn):
    data = conn.recv(4096).decode()
    print("Received message: " + str(data))
    return data


def startup_sequence():
    # Run the server and get necessary information
    robot, address, server_socket = run_server()

    # receive startup message from the robot
    message = receive_message(robot)

    # If the message is "ready", send the necessary information to the robot
    if message.lower().strip() == "ready":
        send_message("ready", robot)
        return robot, server_socket
    else:
        send_message("ready", robot)
        return robot, server_socket


def start_of_run_sequence(robot_heading, vector_list, square_size, robot):
    send_message(robot_heading, robot)
    message = receive_message(robot)
    if message.lower().strip() == "received":
        send_message(vector_list, robot)
        message = receive_message(robot)
        if message.lower().strip() == "received":
            send_message(square_size, robot)
            message = receive_message(robot)
            if message.lower().strip() == "received":
                send_message("done with info", robot)
                message = receive_message(robot)
                new_robot_heading_loop(robot)


def run_sequence(vector_list, square_size, robot):
    send_message(vector_list, robot)
    message = receive_message(robot)
    if message.lower().strip() == "received":
        send_message(square_size, robot)

        new_robot_heading_loop(robot)


def new_robot_heading_loop(robot):
    in_middle_of_run = True
    while in_middle_of_run:
        message = receive_message(robot)
        if message.lower().strip() == "get new robot heading":
            robot_heading = str(opencv_camera.get_robot_heading())
            print("sending heading back to: " + robot_heading)
            send_message(robot_heading, robot)
            message = receive_message(robot)
            if message.lower().strip() == "run is done":
                in_middle_of_run = False
                break
    
    send_message("continue", robot)
    message = receive_message(robot)


# send emergency stop message to robot to make it stop
def emergency_stop():
    global stop_flag
    stop_flag = True


def run_calibration_sequence(robot):
    send_message("calibration phase", robot)
    message = receive_message(robot)
    if (message.lower().strip() == "calibrate ready"):
        first_frame = opencv_camera.take_picture()
        send_message("calibration move", robot)
        message = receive_message(robot)
        if (message.lower().strip() == "calibration move done"):
            second_frame = opencv_camera.take_picture()
            return first_frame, second_frame


# Task is to take frames for left and right angle with the robot
def run_calibration_angle_sequence(robot):
    message = receive_message(robot)
    if message.lower().strip() == "calibrate ready":
        f1_left = opencv_camera.take_picture()
        send_message("calibration left", robot)
        message = receive_message(robot)
        if message.lower().strip() == "calibration left done":
            f2_left = opencv_camera.take_picture()
            send_message("calibration right", robot)
            message = receive_message(robot)
            if message.lower().strip() == "calibration right done":
                f2_right = opencv_camera.take_picture()

                return f1_left, f2_left, f2_left, f2_right


def send_new_robot_heading(robot, new_heading):
    send_message(new_heading, robot)
    message = receive_message(robot)
    if message.lower().strip() == "received":
        return


def emergency_stop_listener():
    global stop_flag
    global robot
    while True:
        if stop_flag:
            send_message("emergency stop", robot)
            break


