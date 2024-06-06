import cv2
import numpy as np
import heapq
import math

# Define the classes and functions from the second file

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.g = float('inf')  # Distance from start node
        self.h = None  # Heuristic distance to end node
        self.f = None  # Total cost
        self.parent = None

    def __lt__(self, other):
        return self.f < other.f

def heuristic(node, goal):
    # Euclidean distance heuristic
    return ((node.x - goal.x) ** 2 + (node.y - goal.y) ** 2) ** 0.5

def astar(grid, start, goals, obstacles):
    open_set = []
    closed_set = set()
    reached_goals = set()  # Keep track of reached goals

    start_node = Node(start[0], start[1])
    start_node.g = 0
    start_node.h = min(heuristic(start_node, Node(goal[0], goal[1])) for goal in goals)
    start_node.f = start_node.g + start_node.h

    heapq.heappush(open_set, start_node)

    while open_set:
        current = heapq.heappop(open_set)

        if (current.x, current.y) in goals:
            reached_goals.add((current.x, current.y))  # Add reached goal to the set

            if reached_goals == set(goals):  # Check if all goals reached
                path = []
                while current:
                    path.append((current.x, current.y))
                    current = current.parent
                return path[::-1]

        closed_set.add((current.x, current.y))

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
            new_x, new_y = current.x + dx, current.y + dy
            cost = 1 if abs(dx) + abs(dy) == 1 else math.sqrt(2)  # Adjust movement cost for diagonals

            if 0 <= new_x < len(grid) and 0 <= new_y < len(grid[0]) and grid[new_x][new_y] != 1 and (new_x, new_y) not in closed_set and (new_x, new_y) not in obstacles:
                neighbor = Node(new_x, new_y)
                neighbor.g = current.g + cost  # Update movement cost
                neighbor.h = min(heuristic(neighbor, Node(goal[0], goal[1])) for goal in goals)  # Calculate heuristic for each goal
                neighbor.f = neighbor.g + neighbor.h
                neighbor.parent = current

                if (neighbor.x, neighbor.y) not in {(node.x, node.y) for node in open_set}:  # Check membership in open_set efficiently
                    heapq.heappush(open_set, neighbor)

    return None

# Define functions from the first file

index = ["color", "color_name", "hex", "R", "G", "B"]
columns = 50
rows = 50
arr = [[0]*columns for _ in range(rows)]

def find_ball(frame, min_radius=5, max_radius=20):
    print("find ball")
    balls = []
    highprio = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    cv2.imshow('Output Image', edges)
    cv2.waitKey(0)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(frame, contours, -1, (255, 0, 0), 2)

    for i, contour in enumerate(contours):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(frame, str(i+1), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        print("Proccessing contour ", i+1, "of", len(contours))
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        print("Area:", area)  # Print area for debugging
        print("Perimeter:", perimeter)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(frame, f"({circularity:.2f})", (cX, cY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        print(circularity)

        if circularity > 0.8:
            print("ball found")
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            r, b, g = get_pixel_color(frame, int(x), int(y))
            center = (int(x), int(y))
            radius = int(radius)
            print(r,b,g)
            if min_radius < radius < max_radius and isValidColorBall(r, b, g):
                print("drawing circle")
                cv2.circle(frame, center, radius, (0, 255, 255), 2)
                balls.append((center, radius))
            elif min_radius < radius < max_radius and isHighPrioBall(r, b, g):   
                print("high prio found")
                cv2.circle(frame, center, radius, (0,0,0), 2)
                highprio.append((center, radius))
                
    print("done finding balls")
    return balls, highprio, frame

def get_pixel_color(frame, x, y):
    (B, G, R) = frame[y, x]
    return R, G, B

def isHighPrioBall(r, g, b):
    if r > 150 and g > 100 and b < 100:
        return True
    return False

def isValidColorBall(r, g, b):
    if r > 150 and g < 100 and b < 100:
        return True
    return False

def display(frame, balls):
    for ball in balls:
        center, radius = ball
        cv2.circle(frame, center, radius, (0, 255, 255), 2)
    cv2.imshow('Output Image', frame)
    cv2.waitKey(0)

def create_grid(columns, rows, balls, highprio):
    arr = [[0]*columns for _ in range(rows)]
    for ball in balls:
        center, radius = ball
        x, y = center
        arr[y][x] = 1
    for ball in highprio:
        center, radius = ball
        x, y = center
        arr[y][x] = 2
    return arr

# Main function
def main():
    # Open the camera
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Initialize an empty list to store detected balls
    balls = []

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Find outer walls
        mask, box_dimensions = find_outer_walls(frame)

        # Find balls
        output_image, balls, highprio = find_ball(mask)

        # Update goals with the coordinates of detected balls
        goals = [(ball[0][0], ball[0][1]) for ball in balls]

        # Perform pathfinding considering balls as obstacles
        grid = create_grid(output_image, balls, box_dimensions)
        start = (2, 2)
        path = astar(grid, start, goals)

        # Draw path on the output image
        if path:
            for i in range(len(path) - 1):
                cv2.line(output_image, path[i], path[i + 1], (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Camera Feed', output_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def create_grid(output_image, balls, box_dimensions):
    # Create grid based on output image and ball coordinates
    # You can use box_dimensions to adjust the grid based on the bounding box of the walls
    # This function should return a grid where obstacles are marked as 1 and free spaces are marked as 0
    return grid

if __name__ == "__main__":
    main()