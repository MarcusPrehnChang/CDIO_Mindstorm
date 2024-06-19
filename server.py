import socket
import threading

import HSM
import opencv_camera
from opencv_camera import get_info_from_camera as get_inf

robot = None
stop_flag = False


def run_server():
    global robot
    # Get hostname and port
    host = "192.168.98.209"
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
    global robot
    global stop_flag
    # Run the server and get necessary information
    robot, address, server_socket = run_server()

    # receive startup message from the robot
    message = receive_message(robot)

    # If the message is "ready", send the necessary information to the robot
    if message.lower().strip() == "ready":
        send_message("ready", robot)
        run_calibration_sequence(robot)
        robot_heading, vector_list, square_size = get_drive_info()
        send_message(robot_heading, robot)
        message = receive_message(robot)
        if message.lower().strip() == "received":
            iterator = 0
            while running:
                 
                send_message(vector_list[iterator], robot)
                message = receive_message(robot)
                if message.lower().strip() == "received":
                    send_message(square_size, robot)
                    message = receive_message(robot)
                    if message.lower().strip() == "received":
                        print("Startup sequence complete")
                        run_thread = threading.Thread(target=run_sequence)
                        emergency_stop_thread = threading.Thread(target=emergency_stop_listener)
                        emergency_stop_thread.start()
                        run_thread.start()
                        run_thread.join()
                message = receive_message(robot)
                if message.lower().strip() == "ready":
                    pass
                else:
                    running = False
                iterator += 1
            

    server_socket.close()


# run_sequence() needs to be implemented
def run_sequence():
    pass


# send emergency stop message to robot to make it stop
def emergency_stop():
    global stop_flag
    stop_flag = True


def run_calibration_sequence(robot):
    message = receive_message(robot)
    if (message.lower().strip() == "calibrate ready"):
        firstframe = opencv_camera.take_picture()
        send_message("calibration move", robot)
        message = receive_message(robot)
        if (message.lower().strip() == "calibration done"):
            secondframe = opencv_camera.take_picture()
            calibration_difference = HSM.calibration(firstframe, secondframe)
            print(calibration_difference)
            send_message("calibration done", robot)
            send_message(str(calibration_difference), robot)


def emergency_stop_listener():
    global stop_flag
    global robot
    while True:
        if stop_flag:
            send_message("emergency stop", robot)
            break


# get_drive_info() needs to be implemented
def get_drive_info():
    vector_list, robot_heading = get_inf()
    robot_heading = str(robot_heading)  # needs to receive from function
    vector_list = vector_list[0] # needs to receive from function
    vector_list = [vector_list]
    vector_list = str(vector_list)
    square_size = str(20)  # needs to receive from function

    return robot_heading, vector_list, square_size


def start_server():
    server_thread = threading.Thread(target=startup_sequence)
    server_thread.start()


start_server()