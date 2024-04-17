import cv2
import numpy as np
#import pandas as pd

index = ["color", "color_name", "hex", "R", "G", "B"]


def find_ball(frame, min_radius=10, max_radius=200):
    print("find ball")
    balls = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    cv2.imshow('Output Image', edges)
    cv2.waitKey(0)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(contours.count)

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            break
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity > 0.7:
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            r, b, g = get_pixel_color(frame, int(x), int(y))
            center = (int(x), int(y))
            radius = int(radius)

            if min_radius < radius < max_radius and isValidColorBall(r, b, g):
                print("drawing circle")
                cv2.circle(frame, center, radius, (0, 255, 255), 2)
                balls.append((center, radius))    


    return frame, balls

def find_outer_walls(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    mask = cv2.inRange(hsv, lower_red, upper_red)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)


    contour_image = cv2.drawContours(frame.copy(), [max_contour], -1, (0, 255, 0), 2)
    x, y, w, h = cv2.boundingRect(max_contour)

    return contour_image, (x, y, w ,h)


def get_pixel_color(image, x, y):
    b, g, r = image[y, x]
    return r, b, g


def isValidColorBall(R, G, B):
    return R > 200 and G > 150 and B > 100


def isValidColorWall(R, G, B):
    return R < 190 and G > 10 and B > 25

#def create_sparse_map(bounding_box_size, balls):

def main():
    balls = []
    # Image Capture
    input_image = cv2.resize(cv2.imread('images/board.png'), (1000, 1000))

    # input_image = cv2.imread('images/whiteball.jpg')

    # input_image = cv2.resize(cv2.imread('images/gulvbillede.jpg'), (600, 750))

    if input_image is None:
        print("Error: Could not open or read the image")
        return


    mask, box_dimensions = find_outer_walls(input_image)

    output_image, balls = find_ball(mask)



    cv2.imshow('Output Image', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Video Capture

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
