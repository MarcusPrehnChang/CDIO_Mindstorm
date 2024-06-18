#!/usr/bin/env pybricks-micropython
import socket

import autodrive
from autodrive import auto_drive


def connect_to_server(hostname):
    # Name and port of the host
    host = hostname
    port = 5000

    # Get the socket instance and make connection to server
    client_socket = socket.socket()
    client_socket.connect((host, port))

    return client_socket


def send_message(message, client_socket):
    print("Sending message: " + str(message))
    client_socket.send(message.encode())


def receive_message(client_socket):
    data = client_socket.recv(1024).decode()
    print("Received message: " + str(data))
    return data


# Global variable for the autodrive thread
autodrive_thread = None
# Global variable for the emergency stop flag
stop_flag = False
# Global variable to stop the emergency stop listener
emergency_stop_listener = True
# Global variable to stop the run
run_is_not_done = True


def listen_for_emergency_stop(client_socket):
    global autodrive_thread
    global stop_flag
    # Listen for the emergency stop message
    while emergency_stop_listener:
        message = receive_message(client_socket)
        if message.lower().strip() == "emergency stop":
            if autodrive_thread is not None:
                # Stop the autodrive thread
                stop_flag = True
                # Get new information
                robot_heading, vector_list, square_size = get_info(client_socket)
                # Restart the autodrive thread
                #autodrive_thread = threading.Thread(target=autodrive.auto_drive, args=(vector_list, square_size, robot_heading))
                #autodrive_thread.start()


def startup_sequence(hostname):
    global autodrive_thread
    global emergency_stop_listener
    global run_is_not_done
    global stop_flag

    # Connect to the server
    client_socket = connect_to_server(hostname)
    # Send ready message
    send_message("ready", client_socket)
    message = receive_message(client_socket)

    if message.lower().strip() == "ready":
        run_calibration(client_socket)
        while run_is_not_done:
            robot_heading, vector_list, square_size = get_info(client_socket)

            # Start the listen_for_emergency_stop thread
            # listen_thread = threading.Thread(target=listen_for_emergency_stop, args=(client_socket,))
            # Start the autodrive thread
            # autodrive_thread = threading.Thread(target=autodrive.auto_drive, args=(vector_list, square_size, robot_heading,))
            # listen_thread.start()
            # autodrive_thread.start()

            # Wait for the autodrive thread to finish
            # autodrive_thread.join()
            # Stop the listen_for_emergency_stop thread to receive new messages
            auto_drive(vector_list, square_size, robot_heading)
            emergency_stop_listener = False

            send_message("run is done", client_socket)
            continuation_message = receive_message(client_socket)
            if continuation_message == "run is done":
                send_message("received", client_socket)
                run_is_not_done = False

            if continuation_message == "continue":
                send_message("received", client_socket)
                # Reset the emergency stop listener
                stop_flag = False

    client_socket.close()


# Get the robot heading, vector list, and square size
def get_info(client_socket):
    robot_heading = eval(receive_message(client_socket))
    send_message("received", client_socket)
    vector_list = eval(receive_message(client_socket))
    send_message("received", client_socket)
    square_size = int(receive_message(client_socket))
    send_message("received", client_socket)

    square_size = square_size * autodrive.calibration_variable
    return robot_heading, vector_list, square_size


def run_calibration(client_socket):
    send_message("calibrate ready", client_socket)
    message = receive_message(client_socket)
    if message.lower().strip() == "calibration move":
        autodrive.calibration_move()
        send_message("calibration done", client_socket)
        message = receive_message(client_socket)
        if message.lower().strip() == "calibration done":
            calibration_difference = receive_message(client_socket)
            autodrive.set_calibration_variable(float(calibration_difference))

# Run the client
def run_client():
    startup_sequence("192.168.98.209")
    # startup_thread = threading.Thread(target=startup_sequence, args=("192.168.23.184",))
    # startup_thread.start()
    # startup_thread.join()


run_client()
