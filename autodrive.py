#!/usr/bin/env pybricks-micropython
import math
# from client import stop_flag

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
# Adjust the wheel diameter and axle track according to our robot design
robot = DriveBase(left_motor, right_motor, wheel_diameter=40, axle_track=115)
# robot.settings(straight_speed=200, straight_acceleration=100, turn_rate=100)


gyro_sensor = GyroSensor(Port.S2)

square_size = 20
calibration_variable_drive = 1
calibration_variable_angle_left = 1
calibration_variable_angle_right = 1


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


def navigate_to_ball(vector_list, square_size, robot_heading):
    for vector in vector_list:
        angle_to_turn = get_angle_to_turn(robot_heading, vector)
        # Based on it what way it should turn
        if angle_to_turn < 0:
            angle_to_turn = angle_to_turn * calibration_variable_angle_left
        else:
            angle_to_turn = angle_to_turn * calibration_variable_angle_right

        distance_to_drive = get_distance_to_drive(vector, square_size)
        # if stop_flag:
        # break
        # Turn the robot to the correct angle
        turn(angle_to_turn, 50)
        # if stop_flag:
        # break
        # Drive the robot to the target distance
        drive(distance_to_drive, 50)

        # Update the robot's position
        robot_heading = vector

    return robot_heading


def get_distance_to_drive(vector, square_size):
    distance_to_drive = math.sqrt(vector[0] ** 2 + vector[1] ** 2) * square_size
    return distance_to_drive


def get_angle_to_turn(robot_heading, pointer_vector):
    robot_heading_distance = math.sqrt(robot_heading[0] ** 2 + robot_heading[1] ** 2)
    pointer_vector_distance = math.sqrt(pointer_vector[0] ** 2 + pointer_vector[1] ** 2)
    vector_product = robot_heading[0] * pointer_vector[0] + robot_heading[1] * pointer_vector[1]
    angle_to_turn_radian = math.acos(vector_product / (robot_heading_distance * pointer_vector_distance))
    angle_to_turn = math.degrees(angle_to_turn_radian)

    if robot_heading[0] * pointer_vector[1] - robot_heading[1] * pointer_vector[0] > 0:
        angle_to_turn = -angle_to_turn

    return angle_to_turn


def auto_drive(list_of_list_of_vectors, square_size, robot_heading):
    for list_of_vectors in list_of_list_of_vectors:
        pick_up_ball()
        new_heading = navigate_to_ball(list_of_vectors, square_size, robot_heading)
        # if stop_flag:
        # break
    return new_heading


def calculate_square_size(amount_of_squares_length, amount_of_squares_width):
    length_size = 180
    width_size = 120
    square_size_length = length_size / amount_of_squares_length
    square_size_width = width_size / amount_of_squares_width
    square_size = (square_size_length + square_size_width) / 2

    return square_size


def calibration_move():
    drive(20, 50)


def calibration_turn_left():
    turn(get_angle_to_turn([1, 0], [0, 1]), 50)


def calibration_turn_right():
    turn(get_angle_to_turn([0, 1], [1, 0]), 50)


def pick_up_ball():
    small_motor.run(400)
    wait(100)


def offload_ball():
    small_motor.run(-400)
    wait(100)


def set_calibration_variable_drive(new_calibration_variable):
    global calibration_variable_drive
    calibration_variable_drive = new_calibration_variable


def set_calibration_variable_angle(angle_variable_right, angle_variable_left):
    global calibration_variable_angle_right
    global calibration_variable_angle_left
    calibration_variable_angle_right = float(angle_variable_right)
    calibration_variable_angle_left = float(angle_variable_left)



# Stop the robot
robot.stop()
