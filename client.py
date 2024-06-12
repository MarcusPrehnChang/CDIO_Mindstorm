import socket
import time


import pywifi
from pywifi import const
import comtypes

def wifi_connect(ssid, password):
    wifi = pywifi.PyWiFi()

    # Get the first wireless interface.
    iface = wifi.interfaces()[0]

    # Disconnect all connections.
    iface.disconnect()
    time.sleep(1)

    # Create a new profile with the given ssid and password.
    profile = pywifi.Profile()
    profile.ssid = ssid
    profile.auth = const.AUTH_ALG_OPEN
    profile.akm.append(const.AKM_TYPE_WPA2PSK)
    profile.cipher = const.CIPHER_TYPE_CCMP
    profile.key = password

    # Remove all other profiles and add the new one.
    iface.remove_all_network_profiles()
    iface.add_network_profile(profile)

    # Connect to the network.
    iface.connect(profile)
    time.sleep(10)

    if iface.status() == const.IFACE_CONNECTED:
        print(f"Connected to {ssid}")
    else:
        print("Failed to connect")

    # List available networks.
    print("Available networks:")
    networks = iface.scan_results()
    for network in networks:
        print(network.ssid)


def client_program(hostname):
    # Name and port of the host
    host = hostname
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
    wifi_connect("OnePlus 11 5G", "s92iynbj")
    client_program("192.168.23.124")


if __name__ == "__main__":
    main()
