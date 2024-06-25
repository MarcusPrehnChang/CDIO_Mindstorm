#!/usr/bin/env pybricks-micropython
import socket

import autodrive
from autodrive import calibration_move
from autodrive import set_calibration_variable_drive
from autodrive import set_calibration_variable_angle
from autodrive import calibration_turn_right
from autodrive import calibration_turn_left


# Simon (s224277) - 10%
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

client_socket = None


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


def phase_switcher(client_socket):
    received_message = receive_message(client_socket)
    if received_message.lower().strip() == "calibration phase":
        pass
        # run_calibration(client_socket)
        # run_calibration_angle(client_socket)
    elif received_message.lower().strip() == "robot phase":
        run_loop_sequence(client_socket)
    elif received_message.lower().strip() == "emergency phase":
        pass


def startup_sequence(hostname):
    global autodrive_thread
    global emergency_stop_listener
    global run_is_not_done
    global stop_flag
    global client_socket

    # Connect to the server
    client_socket = connect_to_server(hostname)
    # Send ready message
    send_message("ready", client_socket)
    message = receive_message(client_socket)

    if message.lower().strip() == "ready":
        return client_socket
    else:
        return client_socket


def run_loop_sequence(client_socket):
    global run_is_not_done
    send_message("received robot phase", client_socket)
    robot_heading, vector_list, square_size = get_info(client_socket)
    message = receive_message(client_socket)
    if message.lower().strip() == "done with info":
        send_message("received", client_socket)
        for vectors in vector_list:
            autodrive.pick_up_ball()
            navigate_to_ball(vectors, square_size, robot_heading)
        send_message("run is done", client_socket)
        continuation_message = receive_message(client_socket)
        if continuation_message == "continue":
            send_message("received", client_socket) #kør til resten
            while run_is_not_done:
                message = receive_message(client_socket)
                vector_list = eval(message)
                send_message("received", client_socket)
                message = receive_message(client_socket)
                square_size = int(message)
                for vectors in vector_list:
                    autodrive.pick_up_ball()
                    navigate_to_ball(vectors, square_size, robot_heading)


                # Start the listen_for_emergency_stop thread
                # listen_thread = threading.Thread(target=listen_for_emergency_stop, args=(client_socket,))
                # Start the autodrive thread
                # autodrive_thread = threading.Thread(target=autodrive.auto_drive, args=(vector_list, square_size, robot_heading,))
                # listen_thread.start()
                # autodrive_thread.start()

                # Wait for the autodrive thread to finish
                # autodrive_thread.join()
                # Stop the listen_for_emergency_stop thread to receive new messages

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

        else:
            # here is where it ends, start ending sequence
            print("Ending of whole run")


# Get the robot heading, vector list, and square size
def get_info(client_socket):
    robot_heading = eval(receive_message(client_socket))
    send_message("received", client_socket)
    vector_list = eval(receive_message(client_socket))
    send_message("received", client_socket)
    square_size = int(receive_message(client_socket))
    send_message("received", client_socket)

    #square_size = square_size * autodrive.calibration_variable_drive
    square_size = square_size * 1.01
    return robot_heading, vector_list, square_size


def run_calibration(client_socket):
    send_message("calibrate ready", client_socket)
    message = receive_message(client_socket)
    if message.lower().strip() == "calibration move":
        calibration_move()
        send_message("calibration move done", client_socket)
        message = receive_message(client_socket)
        if message.lower().strip() == "calibration done":
            send_message("Received", client_socket)
            calibration_difference = receive_message(client_socket)
            send_message("Received", client_socket)
            set_calibration_variable_drive(float(calibration_difference))


# Simon (s224277) - 100%
def run_calibration_angle(client_socket):
    send_message("calibrate ready", client_socket)
    message = receive_message(client_socket)
    if message.lower().strip() == "calibration left":
        print("Before Calibration Left")
        calibration_turn_left()
        print("After Calibration Left")
        send_message("calibration left done", client_socket)
        message = receive_message(client_socket)
        if message.lower().strip() == "calibration right":
            calibration_turn_right()
            send_message("calibration right done", client_socket)
            message = receive_message(client_socket)
            if message.lower().strip() == "calibration done":
                send_message("Received", client_socket)
                angle_left = receive_message(client_socket)
                send_message("Received", client_socket)
                angle_right = receive_message(client_socket)
                send_message("Received", client_socket)
                if receive_message(client_socket) == "Done with calibration":
                    set_calibration_variable_angle(angle_right, angle_left)
                    send_message("Done applying angles", client_socket)
                    if receive_message(client_socket) == "Received":
                        phase_switcher(client_socket)


def get_new_robot_heading():
    global client_socket
    send_message("get new robot heading", client_socket)
    robot_heading = eval(receive_message(client_socket))
    send_message("received", client_socket)
    return robot_heading


def turn_till_precise(vector):
    turn_again = True
    while turn_again:
        new_heading = get_new_robot_heading()
        autodrive.wait(200)
        print("current heading: ", new_heading)
        angle_to_turn = autodrive.get_angle_to_turn(new_heading, vector)
        if - 2.5 < angle_to_turn < 2.5:
            print("AGNLE GOOD ENOUGH")
            break
        else:
            print("turning : ", angle_to_turn)
            if abs(angle_to_turn) < 10:
                autodrive.turn(angle_to_turn, 50)
            else:
                autodrive.turn(angle_to_turn, 75)


def navigate_to_ball(vector_list, square_size, robot_heading):
    for vector in vector_list:
        distance_to_drive = autodrive.run_one_vector_turn(robot_heading, vector)
        turn_till_precise(vector)
        autodrive.drive(distance_to_drive, 75)


def auto_drive(list_of_list_of_vectors, square_size, robot_heading):
    for list_of_vectors in list_of_list_of_vectors:
        autodrive.pick_up_ball()
        navigate_to_ball(list_of_vectors, square_size, robot_heading)
        # if stop_flag:
        # break


# Run the client
def run_client():
    client_socket = startup_sequence("192.168.23.184")
    phase_switcher(client_socket)
    # startup_thread = threading.Thread(target=startup_sequence, args=("192.168.23.184",))
    # startup_thread.start()
    # startup_thread.join()


run_client()
