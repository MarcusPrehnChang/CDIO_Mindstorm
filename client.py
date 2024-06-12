#!/usr/bin/env pybricks-micropython
import socket

from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor, GyroSensor
from pybricks.parameters import Port
from pybricks.robotics import DriveBase
from pybricks.tools import wait

# Initialize the EV3 Brick
ev3 = EV3Brick()

# Initialize the motors connected to the wheels
left_motor = Motor(Port.A)
right_motor = Motor(Port.C)
small_motor = Motor(Port.B)

# Create a DriveBase object with the initialized motors
# Adjust the wheel diameter and axle track according to your robot design
robot = DriveBase(left_motor, right_motor, wheel_diameter=40, axle_track=110)
# robot.settings(straight_speed=200, straight_acceleration=100, turn_rate=100)

gyro_sensor = GyroSensor(Port.S1)


# angle = degrees to turn, speed = mm/s
def turn(angle, speed):
    gyro_sensor.reset_angle(0)
    if angle < 0:
        while gyro_sensor.angle() > angle:
            left_motor.run(speed=(-1 * speed))
            right_motor.run(speed=speed)
            wait(10)
    elif angle > 0:
        while gyro_sensor.angle() < angle:
            left_motor.run(speed=speed)
            right_motor.run(speed=(-1 * speed))
            wait(10)
    else:
        print("Error: no angle chosen")

    left_motor.brake()
    right_motor.brake()


# distance = mm, robotSpeed = mm/s
def drive(distance, robot_speed):
    robot.reset()
    gyro_sensor.reset_angle(0)

    PROPORTIONAL_GAIN = 1.1
    if distance < 0:  # move backwards
        while robot.distance() > distance:
            reverse_speed = -1 * robot_speed
            angle_correction = -1 * (PROPORTIONAL_GAIN * gyro_sensor.angle())
            robot.drive(reverse_speed, angle_correction)
            wait(10)
    elif distance > 0:  # move forwards
        while robot.distance() < distance:
            angle_correction = -1 * PROPORTIONAL_GAIN * gyro_sensor.angle()
            robot.drive(robot_speed, angle_correction)
            wait(10)
    robot.stop()

def client_program(hostname):
    # Name and port of the host
    host = hostname
    port = 5000

    # Get the socket instance and make connection to server
    client_socket = socket.socket()
    client_socket.connect((host, port))

    # Take input
    message = "ready"

    while message.lower().strip() != 'bye':
        # Send message and receive response
        client_socket.send(message.encode())
        data = client_socket.recv(1024).decode()

        # Show in terminal
        print('Received from server: ' + data)
        if (data == "drive"):
            drive(100, 50)
        # Taking input again
        message = "recieved"

    # Close the connection when receiving 'bye'
    client_socket.close()


def connect():
    client_program("192.168.23.124")


connect()
