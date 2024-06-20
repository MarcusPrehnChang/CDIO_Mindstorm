import math

import cv2
import numpy as np


import pathFinder
from Translator import GridTranslator
from pathFinder import find_path_to_multiple

index = ["color", "color_name", "hex", "R", "G", "B"]
columns = 90
rows = 60
arr = [[0] * columns for _ in range(rows)]
walls = []
balls = []
highprio = []
gooseEgg = []
number_of_minimum_balls = 11
robot_identifier = []
cell_width = 0


def detect_Objects(frame):
    find_ball(frame)
    bounding_box = find_walls(frame)
    frame = map_objects(bounding_box, frame)
    
    return frame


def find_highprio(frame, min_radius=5, max_radius=20):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([18, 10, 0])
    upper_orange = np.array([50, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity > 0.7:
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            r, b, g = get_pixel_color(frame, int(x), int(y))
            center = (int(x), int(y))
            radius = int(radius)
            if min_radius < radius < max_radius:
                cv2.circle(frame, center, radius, (0, 0, 0), 2)
                highprio.append(contour)


def find_ball(frame, min_radius=4, max_radius=20):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    find_highprio(frame)


    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity > 0.7:
            print("circularity passed: ", i , " times")
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            r, b, g = get_pixel_color(frame, int(x), int(y))
            x, y = int(x), int(y)
            center = (x, y)
            radius = int(radius)

            print("radius of the " , i, "th ball is: ", radius)
            if min_radius < radius < max_radius and isValidColorBall(r, b, g):
                cv2.circle(frame, center, radius, (0, 255, 255), 2)
                balls.append(contour)
            elif radius > max_radius:
                cv2.circle(frame, center, radius, (255, 0, 0), 2)
                gooseEgg.append(contour)

    
    return balls, highprio
 

def robot_builder(robot_size):
    robot_length = 30
    robot_width = 20
    triangle_height = 7
    triangle_bottom = 6
    robot_grid_height = math.ceil((robot_length / triangle_height) * robot_size)
    robot_grid_width = math.ceil((robot_width / triangle_bottom) * robot_size)

    return robot_grid_height, robot_grid_width

def map_objects(box_dimensions, output_image):
    x, y, w, h = box_dimensions
    cell_width = w // columns
    cell_height = h // rows

    print("cell width: ", cell_width)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.imshow('Image given to map_objects', output_image)

    for wall in walls:
        cv2.drawContours(mask, [wall], -1, 255, thickness=cv2.FILLED, offset=(-x, -y))
    for ball in balls:
        cv2.drawContours(mask, [ball], -1, 100, thickness=cv2.FILLED, offset=(-x, -y))
    for ball in highprio:
        cv2.drawContours(mask, [ball], -1, 155, thickness=cv2.FILLED, offset=(-x, -y))
    for ball in gooseEgg:
        cv2.drawContours(mask, [ball], -1, 20, thickness=cv2.FILLED, offset=(-x, -y))
    for robot in robot_identifier:
        cv2.drawContours(mask, [robot], -1, 75, thickness=cv2.FILLED, offset=(-x, -y))
    robot_size = 0

    for i in range(rows + 1):
        start_point = (0, i * cell_height)
        end_point = (w, i * cell_height)
        cv2.line(mask, start_point, end_point, (143),1)

    for j in range(columns + 1):
        start_point = (j * cell_height, 0)
        end_point = (j * cell_height, h)
        cv2.line(mask, start_point, end_point, (143), 1)


    cv2.imshow('Shape masked grid', mask)


    cv2.waitKey(0)
    cv2.destroyAllWindows
    for row in range(rows):
        for col in range(columns):
            cell_x_start = col * cell_width
            cell_y_start = row * cell_height
            cell_x_end = cell_x_start + cell_width
            cell_y_end = cell_y_start + cell_height

            # Check if any part of the cell is within the mask
            if np.any(mask[cell_y_start:cell_y_end, cell_x_start:cell_x_end] == 255):
                arr[row][col] = 1
            if np.any(mask[cell_y_start:cell_y_end, cell_x_start:cell_x_end] == 100):
                arr[row][col] = 2
            if np.any(mask[cell_y_start:cell_y_end, cell_x_start:cell_x_end] == 155):
                arr[row][col] = 3
            if np.any(mask[cell_y_start:cell_y_end, cell_x_start:cell_x_end] == 20):
                arr[row][col] = 1
            if np.any(mask[cell_y_start:cell_y_end, cell_x_start:cell_x_end] == 75):
                robot_size += 1
                arr[row][col] = 5
                


    return output_image#, robot_size


def get_width():
    return cell_width

def find_walls(frame):
    minimum_size = 100
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([170, 200, 0])
    upper_red = np.array([185, 255, 255])
    lower_red2 = np.array([0, 100, 0])
    upper_red2 = np.array([5, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask, mask2)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    highest_size = 0
    largest_contour = None
    cv2.imshow("masked wall image", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows

    for contour in contours:
        x2, y2, w2, h2 = cv2.boundingRect(contour)
        size = w2 * h2
        if size > highest_size:
            highest_size = size
            largest_contour = contour
    for contour in contours:
        x2, y2, w2, h2 = cv2.boundingRect(contour)
        size = w2 * h2
        if size > minimum_size and size < highest_size / 2:
            walls.append(contour)
    return cv2.boundingRect(largest_contour)


def find_triangle(
        frame,
        area_size=1000,
        #105,145,135
        #134,191,156
        lower_green=np.array([125, 135, 95]),
        upper_green=np.array([165, 205, 145])
):
    # Modifying the image and removing all other color than green to highlight the shape of the triangle
    mask = cv2.inRange(frame, lower_green, upper_green)
    points = []

    # Finding based on shape
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in contours:
        approx = cv2.approxPolyDP(i, 0.1 * cv2.arcLength(i, True), True)
        if len(approx) == 3:
            [area, triangle] = cv2.minEnclosingTriangle(i)
            if area > area_size:
                print(area)
                cv2.imshow('triangle masked Image', mask)
                frame = cv2.drawContours(frame, [i], -1, (255, 0, 0), 3)
                robot_identifier.append(i)
                points = triangle

    return frame, points


def find_abc(points):
    # Init the points to x and y
    y1, x1 = points[0][0]
    y2, x2 = points[1][0]
    y3, x3 = points[2][0]

    # Calculate the distance between points using the Afstandsformlen
    length1 = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    length2 = math.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
    length3 = math.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)

    print("length1: " + length1.__str__())
    print("length2: " + length2.__str__())
    print("length3: " + length3.__str__())

    # Determine which point is C (the return is in this order A, B, C)
    if math.isclose(length1, length2, abs_tol=10):
        # print("1")
        return points[1][0], points[2][0], points[0][0]
    elif math.isclose(length3, length2, abs_tol=10):
        # print("2")
        return points[0][0], points[1][0], points[2][0]
    elif math.isclose(length1, length3, abs_tol=10):
        # print("3")
        return points[0][0], points[2][0], points[1][0]
    else:
        return None, None, None


def get_orientation(frame, points):
    # Find A,B and C points
    A, B, C = find_abc(points)

    # Init the points to x and y
    y1, x1 = A
    y2, x2 = B
    y3, x3 = C
    print(x1, y1, x2, y2, x3, y3)

    # Calculate the point between A and B
    Mx = (x2 + x1) / 2
    My = (y2 + y1) / 2

    #print("Mx:", Mx)
    #print("My:", My)

    #print("x3:", x3)
    #print("y3:", y3)

    # Calculate the vector (direction the robot is going)
    # Multiplying with -1 to switch the y coordinate to a normal coordinate system.
    V = [float(Mx - x3), float((My - y3) * -1)]
    return V
    #except:
    #    print("increasing sensitivity")
    #    return inc_sen_triangle(frame)


def get_array():
    return arr


def get_pixel_color(image, x, y):
    b, g, r = image[y, x]
    return r, b, g


def isValidColorBall(R, G, B):
    return R > 100 and G > 100 and B > 100


def isHighPrioBall(R, G, B):
    return R > 200 and G < 100 and B > 150


def isValidColorWall(R, G, B):
    return R < 190 and G > 10 and B > 25


'''
# Increases the search for the triangle by increasing the sensitivity over 10 iterations.
def inc_sen_triangle(frame):
    for i in range(10):
        global robot_identifier
        robot_identifier = []
        new_frame, points = find_triangle(
            frame,
            area_size=(1400 - (i * 10)),
            lower_green=np.array([105 - (i * 5), 110 - (i * 5), 85 - (i * 5)]),
            upper_green=np.array([140 + (i * 5), 145 + (i * 5), 110 + (i * 5)])
        )
        if bool(robot_identifier):
            print("Triangle Found in: " + str(i))
            return get_orientation(frame, points)
            break
'''

# Increases the search for the balls by increasing the sensitivity over 10 iterations.
def inc_sen_balls(frame):
    for i in range(10):
        global balls
        global highprio
        balls = []
        highprio = []
        find_ball(
            frame,
            min_radius=(150 - (i * 10)),
            max_radius=(300 + (i * 10))
        )
        find_highprio(
            frame,
            min_radius=(150 - (i * 10)),
            max_radius=(300 + (i * 10))
        )
        if bool(balls or highprio):
            print("Ball Found")
            break


# def create_sparse_map(bounding_box_size, balls):

def print_grid(grid):
    for row in grid:
        print(" ".join(map(str, row)))


def take_picture():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    ret, frame = cap.read()

    return frame


def get_info_from_camera():
    balls = []
    # Image Capture
    input_image = cv2.resize(cv2.imread('images/real_map.jpg'), (1280,720))

    newFrame, points = find_triangle(input_image)
    if points is not None:
        vec = get_orientation(input_image, points)
    else:
        print("error finding triangle")
    print(points)

    if input_image is None:
        print("Error: Could not open or read the image")
        return
    frame = detect_Objects(input_image)
    grid_translator = GridTranslator(arr)
    grid_translator.translate()
    translated_goals, translated_high, translated_start = grid_translator.get_info()
    object_size = (2, 2)
    path = find_path_to_multiple(arr, translated_start, translated_goals, object_size)
    print("Fullpath:", path)
    vectors = grid_translator.make_list_of_lists(path)
    print("vectors made", vectors)
    vectorlist = grid_translator.make_vectors(vectors)
    return vectorlist, vec


def test():
    #frame = cv2.resize(cv2.imread('images/Triangletest2.jpg'), (1000, 1025))

    frame = cv2.imread('images/thisistheone.jpg')

    new_frame, points = find_triangle(frame)
    vec = get_orientation(frame, points)
    print("Vector:" + str(vec))
    print(robot_identifier)

    grid_translator = GridTranslator(arr)
    grid_translator.translate()
    translated_goals, translated_high, translated_start = grid_translator.get_info()
    object_size = (2, 2)
    path = find_path_to_multiple(arr, translated_start, translated_goals, object_size)
    print("Path:", path)
    vectors = grid_translator.make_list_of_lists(path)
    print("work?", vectors)
    vectorlist = grid_translator.make_vectors(vectors)
    print("vectors", vectorlist)
    cv2.imshow('frame', new_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


