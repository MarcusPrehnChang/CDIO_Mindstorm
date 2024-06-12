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
gooseEgg = []
number_of_minimum_balls = 11


def detect_Objects(frame):
    find_ball(frame)
    bounding_box = find_walls(frame)
    frame = map_objects(bounding_box, frame)
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
                balls.append(contour)
            elif min_radius < radius < max_radius and isHighPrioBall(r, b, g):   
                cv2.circle(frame, center, radius, (0,0,0), 2)
                highprio.append(contour)
            elif radius > max_radius:
                cv2.circle(frame, center, radius, (255,0,0), 2)
                gooseEgg.append(contour)

                
    return balls, highprio

def map_objects(box_dimensions, output_image):
    x, y, w, h = box_dimensions
    cell_width = w // rows
    cell_height = h // columns

    counter = 0
    mask = np.zeros((h,w), dtype=np.uint8)
    for wall in walls:
        cv2.drawContours(mask, [wall], -1, 255, thickness=cv2.FILLED, offset=(-x, -y))
    for ball in balls:
        cv2.drawContours(mask, [ball], -1, 100, thickness=cv2.FILLED, offset=(-x, -y))
    for ball in highprio:
        cv2.drawContours(mask, [ball], -1, 155, thickness=cv2.FILLED, offset=(-x, -y))
    for ball in gooseEgg:
        cv2.drawContours(mask, [ball], -1, 20, thickness=cv2.FILLED, offset=(-x, -y))

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
   
    return output_image

def find_walls(frame):
    minimum_size = 100
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    highest_size = 0
    largest_contour = None

    for contour in contours:
        x2, y2, w2, h2 = cv2.boundingRect(contour)
        size = w2 * h2
        if size > highest_size:
            highest_size = size
            largest_contour = contour
    for contour in contours:
        x2, y2, w2, h2 = cv2.boundingRect(contour)
        size = w2 * h2
        if size > minimum_size and size < highest_size/2:
            walls.append(contour)


    return cv2.boundingRect(largest_contour)

def find_triangle(frame, area_size=600):
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
            [area, triangle] = cv2.minEnclosingTriangle(i)
            if area > area_size:
                frame = cv2.drawContours(frame, [i], -1, (255, 0, 0), 3)
                points = triangle
                print(area)
                print(triangle)

    return frame, points


def find_abc(points):
    # Init the points to x and y
    x1, y1 = points[0][0]
    x2, y2 = points[1][0]
    x3, y3 = points[2][0]

    # Calculate the distance between points using the Afstandsformlen
    length1 = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    length2 = math.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
    length3 = math.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)

    print("length1: " + length1.__str__())
    print("length2: " + length2.__str__())
    print("length3: " + length3.__str__())

    # Determine which point is C (the return is in this order A, B, C)
    if math.isclose(length1, length2, abs_tol=10):
        #print("1")
        return points[1][0], points[2][0], points[0][0]
    elif math.isclose(length3, length2, abs_tol=10):
        #print("2")
        return points[0][0], points[1][0], points[2][0]
    elif math.isclose(length1, length3, abs_tol=10):
        #print("3")
        return points[0][0], points[2][0], points[1][0]
    else:
        return None, None, None


def get_orientation(points):
    # Find A,B and C points
    A, B, C = find_abc(points)

    # Init the points to x and y
    x1, y1 = A
    x2, y2 = B
    x3, y3 = C

    # Calculate the point between A and B
    Mx = (x2 + x1) / 2
    My = (y2 + y1) / 2

    #print("Mx:", Mx)
    #print("My:", My)

    #print("x3:", x3)
    #print("y3:", y3)

    # Calculate the vector (direction the robot is going)
    # Multiplying with -1 to switch the y coordinate to a normal coordinate system.
    V = [x3 - Mx, (y3 - My) * -1]
    return V


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

def print_grid(grid):
    for row in grid:
        print(" ".join(map(str, row)))

def main():
    balls = []
    # Image Capture
    input_image = cv2.resize(cv2.imread('images/wallplusballs.jpg'), (1000, 1000))

    input2 = cv2.resize(cv2.imread('images/triangle_robot_balls.jpg'), (1000, 1000))

    newFrame, points = find_triangle(input2)

    cv2.imshow('Example', newFrame)

    vec = get_orientation(points)
    print('Vector Direction:', vec)

    # input_image = cv2.imread('images/whiteball.jpg')

    # input_image = cv2.resize(cv2.imread('images/gulvbillede.jpg'), (600, 750))

    if input_image is None:
        print("Error: Could not open or read the image")
        return
    frame = detect_Objects(input_image)

    cv2.imshow('Output Image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print_grid(arr)

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
