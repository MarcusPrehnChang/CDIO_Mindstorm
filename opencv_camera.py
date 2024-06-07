import math

import cv2
import numpy as np

# import pandas as pd

index = ["color", "color_name", "hex", "R", "G", "B"]
columns = 50
rows = 50
arr = [[0] * columns for _ in range(rows)]
walls = []
balls = []
highprio = []
number_of_minimum_balls = 11


def detect_Objects(frame):
    find_ball(frame)
    box_dimensions = find_outer_walls(frame)
    return frame


def find_ball(frame, min_radius=5, max_radius=20):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity > 0.8:
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            r, b, g = get_pixel_color(frame, int(x), int(y))
            center = (int(x), int(y))
            radius = int(radius)
            if min_radius < radius < max_radius and isValidColorBall(r, b, g):
                cv2.circle(frame, center, radius, (0, 255, 255), 2)
                balls.append((center, radius))
            elif min_radius < radius < max_radius and isHighPrioBall(r, b, g):
                cv2.circle(frame, center, radius, (0, 0, 0), 2)
                highprio.append((center, radius))
    return balls, highprio


def map_objects(balls, highprio, box_dimensions, output_image):
    x, y, w, h = box_dimensions
    cell_width = w // rows
    cell_height = h // columns

    for i in range(1, rows):
        cv2.line(output_image, (x + cell_height * i, y), (x + cell_height * i, y + h), (255, 0, 0))
    for i in range(1, columns):
        cv2.line(output_image, (x, y + cell_width * i), (x + w, y + cell_width * i), (255, 0, 0))

    counter = 0

    for ball in balls:
        counter = counter + 1
        center, _ = ball

        cell_x = (center[0] - x) // cell_width
        cell_y = (center[1] - y) // cell_height
        if 0 <= cell_x < columns and 0 <= cell_y < rows:
            arr[cell_x][cell_y] = 2
        else:
            print("Warning: Out of bounds for cell:", cell_x, cell_y)
    for ball in highprio:
        counter = counter + 1
        center, _ = ball

        cell_x = (center[0] - x) // cell_width
        cell_y = (center[1] - y) // cell_height
        if 0 <= cell_x < columns and 0 <= cell_y < rows:
            arr[cell_x][cell_y] = 3
        else:
            print("Warning: Out of bounds for cell:", cell_x, cell_y)

    return output_image


def find_outer_walls(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)

    for contour in contours:
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        cv2.boundingRect(contour)
        if contour is not max_contour:
            walls.append(contour)

    contour_image = cv2.drawContours(frame.copy(), [max_contour], -1, (255, 255, 0), 2)
    x, y, w, h = cv2.boundingRect(max_contour)

    return contour_image, (x, y, w, h)


def find_triangle(frame):
    # Modifying the image and removing all other color than green to highlight the shape of the triangle
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    points = []

    # Finding based on shape
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of contours detected: ", len(contours))

    for i in contours:
        approx = cv2.approxPolyDP(i, 0.01 * cv2.arcLength(i, True), True)
        if len(approx) == 3:
            frame = cv2.drawContours(frame, [i], -1, (0, 0, 0), 3)
            for n in approx:
                print(n[0])
                points.append(n[0])

    return frame, points


def get_orientation(points):
    # Init the points to x and y
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]

    # Calculate the distance between points using the Afstandsformlen
    length1 = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    length2 = math.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
    length3 = math.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)

    # Determine which point is C
    if math.isclose(length1, length2, rel_tol=1):
        print(points[0])
    elif math.isclose(length3, length2, rel_tol=1):
        print(points[2])
    elif math.isclose(length1, length2, rel_tol=1):
        print(points[1])

def get_array():
    return arr


def get_pixel_color(image, x, y):
    b, g, r = image[y, x]
    return r, b, g


def isValidColorBall(R, G, B):
    return R > 200 and G > 150 and B > 100


def isHighPrioBall(R, G, B):
    return R > 200 and G < 100 and B > 150


def isValidColorWall(R, G, B):
    return R < 190 and G > 10 and B > 25


# def create_sparse_map(bounding_box_size, balls):

def main():
    balls = []
    # Image Capture
    input_image = cv2.resize(cv2.imread('images/wallplusballs.jpg'), (1000, 1000))

    input2 = cv2.resize(cv2.imread('images/dark_green_triangle.png'), (1000, 1000))

    newFrame, points = find_triangle(input2)

    cv2.imshow('Example', newFrame)

    get_orientation(points)

    print(points)

    # input_image = cv2.imread('images/whiteball.jpg')

    # input_image = cv2.resize(cv2.imread('images/gulvbillede.jpg'), (600, 750))

    if input_image is None:
        print("Error: Could not open or read the image")
        return
    frame = detect_Objects(input_image)

    cv2.imshow('Output Image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Video Capture
    some_value = 0
    amount_correct = 0

    """
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

    print("vidcap")
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        print("in while")
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = find_ball(frame)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    """


if __name__ == "__main__":
    main()
