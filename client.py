#!/usr/bin/env pybricks-micropython
import ast
import socket
import autodrive
import threading


def connect_to_server(hostname):
    # Name and port of the host
    host = hostname
    port = 5000

    # Get the socket instance and make connection to server
    client_socket = socket.socket()
    client_socket.connect((host, port))

    return client_socket


def send_message(message, client_socket):
    print("Sending message: " + message)
    client_socket.send(message.encode())


def receive_message(client_socket):
    data = client_socket.recv(1024).decode()
    print("Received message: " + data)
    return data


def startup_sequence(hostname):
    client_socket = connect_to_server(hostname)
    # Take input
    send_message("ready", client_socket)
    message = receive_message(client_socket)

    if message.lower().strip() == "ready":
        robot_heading, vector_list, square_size = get_info(client_socket)

        autodrive_thread = threading.Thread(target=autodrive.auto_drive, args=(vector_list, square_size, robot_heading))
        autodrive_thread.start()
        autodrive_thread.join()

    client_socket.close()


def get_info(client_socket):
    robot_heading = ast.literal_eval(receive_message(client_socket))
    send_message("received", client_socket)
    vector_list = ast.literal_eval(receive_message(client_socket))
    send_message("received", client_socket)
    square_size = int(receive_message(client_socket))
    send_message("received", client_socket)

    return robot_heading, vector_list, square_size


def run_client():
    startup_thread = threading.Thread(target=startup_sequence, args=(socket.gethostname(),))
    startup_thread.start()
    startup_thread.join()


run_client()