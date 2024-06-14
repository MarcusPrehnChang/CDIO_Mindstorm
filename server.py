import socket


def run_server():
    # Get hostname and port
    host = socket.gethostname()
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
    print("Sending message: " + message)
    conn.send(message.encode())


def receive_message(conn):
    data = conn.recv(1024).decode()
    print("Received message: " + data)
    return data


def startup_sequence():
    robot, address, server_socket = run_server()

    # Take input
    message = receive_message(robot)

    if message.lower().strip() == "ready":
        send_message("ready", robot)
        robot_heading, vector_list, square_size = get_drive_info()
        send_message(robot_heading, robot)
        message = receive_message(robot)
        if message.lower().strip() == "received":
            send_message(vector_list, robot)
            message = receive_message(robot)
            if message.lower().strip() == "received":
                send_message(square_size, robot)
                message = receive_message(robot)

    server_socket.close()


def get_drive_info():
    robot_heading = "[0, 1]"  # needs to receive from function
    vector_list = "[[[0, 1], [1, 0], [0, -1], [-1, 0]],[[0, 1], [1, 0], [0, -1], [-1, 0]]]"  # needs to receive from function
    square_size = "200"  # needs to receive from function
    return robot_heading, vector_list, square_size


def main():
    startup_sequence()


if __name__ == "__main__":
    main()
