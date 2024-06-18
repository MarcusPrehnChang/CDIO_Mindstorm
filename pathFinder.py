import heapq

class Node:
    def __init__(self, position, g=0,h=0,parent=None):
        self.position = position
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

def is_valid_position(grid, position, object_size):
    rows, cols = len(grid), len(grid[0])
    for i in range(object_size[0]):
        for j in range(object_size[1]):
            x, y = position[0] + i, position[1] + j
            if x >= rows or y >= cols or x < 0 or y < 0 or grid[x][y] == 1:
                return False
    return True

def get_neighbours(grid, node, object_size):
    neighbours = []
    for dx, dy in [(-1,0), (1,0), (0,-1),(0,1)]:
        neighbour_pos = (node.position[0] + dx, node.position[1] + dy)
        if is_valid_position(grid, neighbour_pos, object_size):
            neighbours.append(neighbour_pos)
    return neighbours

def pyth(a,b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(grid, start, goal, object_size):
    open_set = []
    start_node = Node(start, 0, pyth(start, goal))
    heapq.heappush(open_set, start_node)
    closed_set = set()
    came_from = {}
    
    while open_set:
        current_node = heapq.heappop(open_set)
        current_pos = current_node.position

        if current_pos == goal:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            path.reverse()
            return path
        
        closed_set.add(current_pos)

        for neighbour_pos in get_neighbours(grid, current_node, object_size):
            if neighbour_pos in closed_set:
                continue
            
            g_score = current_node.g + 1
            neighbour_node = Node(
                neighbour_pos,
                g_score,
                pyth(neighbour_pos, goal),
                current_node
            )

            if neighbour_node.position in {n.position for n in open_set}:
                continue

            heapq.heappush(open_set, neighbour_node)

    return None

def find_path_to_multiple(grid, start, goals, object_size):
    full_path = []
    current_start = start
    remaining_goals = set(goals)
    print("goals:", remaining_goals)
    index = 0
    while remaining_goals:
        paths = []
        for goal in remaining_goals:
            print("Running a_star with: ", current_start, goal, object_size)
            path = a_star(grid, current_start, goal, object_size)
            print("path:", path)
            if path:
                paths.append((path, goal))

        if not paths:
            return None

        shortest_path, reached_goal = min(paths, key=lambda x: len(x[0]))
        full_path.extend(shortest_path[1:])
        current_start = reached_goal
        remaining_goals.remove(reached_goal)
        index += 1

    return full_path