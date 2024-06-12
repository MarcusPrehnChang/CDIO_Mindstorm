#!/usr/bin/env pybricks-micropython
import math

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
robot.settings(straight_speed=200, straight_acceleration=100, turn_rate=100)

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


def navigate_to_ball(vectors, square_size):
    robot_heading = get_robot_heading()
    for vector in vectors:
        angle_to_turn = get_angle_to_turn(robot_heading, vector)
        distance_to_drive = get_distance_to_drive(vector, square_size)

        # Turn the robot to the correct angle
        turn(angle_to_turn, 200)

        # Drive the robot to the target distance
        drive(distance_to_drive, 200)

        # Update the robot's position
        robot_heading = vector

def get_robot_heading():
    return get_current_heading()

def get_distance_to_drive(vector, square_size):
    distance_to_drive = math.sqrt(vector[0] ** 2 + vector[1] ** 2) * square_size
    return distance_to_drive

def get_angle_to_turn(robot_heading, pointer_vector):
    robot_heading_distance = math.sqrt(robot_heading[0] ** 2 + robot_heading[1] ** 2)
    pointer_vector_distance = math.sqrt(pointer_vector[0] ** 2 + pointer_vector[1] ** 2)
    vector_product = robot_heading[0] * pointer_vector[0] + robot_heading[1] * pointer_vector[1]
    print("Vector product: ", vector_product)
    print("Robot heading: ", robot_heading_distance)
    print("pointer vector: ", pointer_vector_distance)
    angle_to_turn_radian = math.acos(vector_product/(robot_heading_distance*pointer_vector_distance))
    angle_to_turn = math.degrees(angle_to_turn_radian)
    if(robot_heading[0] * pointer_vector[1] - robot_heading[1] * pointer_vector[0] > 0):
        angle_to_turn = -angle_to_turn
    return angle_to_turn

def auto_drive(list_of_list_of_vectors, square_size):
    for list_of_vectors in list_of_list_of_vectors:
        navigate_to_ball(list_of_vectors, square_size)
        small_motor.run(-300)
        pick_up_ball()


def pick_up_ball():
    small_motor.run(-500)

def get_current_heading(): # Replace with real function, this is a placeholder
    return [0, 1]
# List of vectors representing the positions of the golf balls
vectors = [[[2, 0] , [1 , 3] , [-1, -2], [-1,3]]]

# Starting position of the robot
robot_position = (0, 0)

auto_drive(vectors, 200)

# Stop the robot
robot.stop()
