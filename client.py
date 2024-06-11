import socket


def client_program():
    # Name and port of the host
    host = socket.gethostname()
    port = 5000

    # Get the socket instance and make connection to server
    client_socket = socket.socket()
    client_socket.connect((host, port))

    # Take input
    message = input(" -> ")

    while message.lower().strip() != 'bye':
        # Send message and receive response
        client_socket.send(message.encode())
        data = client_socket.recv(1024).decode()

        # Show in terminal
        print('Received from server: ' + data)

        # Taking input again
        message = input(" -> ")

    # Close the connection when receiving 'bye'
    client_socket.close()


def main():
    client_program()


if __name__ == "__main__":
    main()
