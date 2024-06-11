import socket


def server_program():
    # Get hostname and port
    host = socket.gethostname()
    port = 5000

    # Get the instance of socket and start listening on host and port.
    server_socket = socket.socket()
    server_socket.bind((host, port))

    # Max accepting two connections at once.
    server_socket.listen(2)

    # When connect is accepted, save address and conn in variables
    conn, address = server_socket.accept()

    print("Connection from: " + str(address))

    while True:
        # Receive data stream. Won't accept data packet greater than 1024 bytes
        data = conn.recv(1024).decode()
        if not data:
            # If data is not received break
            break
        print("from connected client: " + str(data))
        data = input(' -> ')
        conn.send(data.encode())

    conn.close()  # Close the connection


def main():
    server_program()


if __name__ == "__main__":
    main()
