import numpy as np


class GridTranslator:
    def __init__(self, grid):
        self.grid = grid
        self.start_point = None
        self.goals = []
        self.highprio = []
        self.visited = set()

    def translate(self):
        for row_index in range(len(self.grid)):
            for col_index in range(len(self.grid[0])):
                value = self.grid[row_index][col_index]
                if value == 5 and (row_index, col_index) not in self.visited:
                    start_area = self.find_area(row_index, col_index, value)
                    self.start_point = self.find_center(start_area)
                elif value == 2 and (row_index, col_index) not in self.visited:
                    start_area = self.find_area(row_index, col_index, value)
                    self.goals.append(self.find_center(start_area))
                elif value == 3 and (row_index, col_index) not in self.visited:
                    start_area = self.find_area(row_index, col_index, value)
                    self.highprio.append(self.find_center(start_area))

    def find_area(self, start_row, start_col, value):
        area = []
        stack = [(start_row, start_col)]
        while stack:
            row, col = stack.pop()
            if (row, col) not in self.visited and self.grid[row][col] == value:
                self.visited.add((row, col))
                area.append((row, col))
                if row > 0:
                    stack.append((row - 1, col))
                if row < len(self.grid) - 1:
                    stack.append((row + 1, col))
                if col > 0:
                    stack.append((row, col - 1))
                if col < len(self.grid[0]) - 1:
                    stack.append((row, col + 1))
        return area

    def find_center(self, area):
        row_sum = sum(row for row, _ in area) / len(area)
        col_sum = sum(col for _, col in area) / len(area)
        center_row = int(round(row_sum))
        center_col = int(round(col_sum))
        return (center_row, center_col)

    def make_list_of_lists(self, path):
        goals = self.goals
        list_of_lists = []
        current_path = []
        prev_path = 0
        for i in range(0, len(path)):
            if path[i] in goals:
                current_path = path[prev_path:i + 1]
                list_of_lists.append(current_path)
                prev_path = i + 1

        return list_of_lists

    def make_vectors(self, list_of_lists):
        current_x = 0
        current_y = 0
        prev_x = 0
        prev_y = 0
        vector_list = []
        for list in list_of_lists:
            vectors = []
            for tuple in list:
                small_list = []
                current_x = tuple[0]
                current_y = tuple[1]
                if current_x != prev_x and prev_x != 0:
                    vector = [current_x - prev_x, 0]
                    vectors.append(vector)
                elif current_y != prev_y and prev_y != 0:
                    vector = [0, current_y - prev_y]
                    vectors.append(vector)
                prev_x = tuple[0]
                prev_y = tuple[1]
            vector_list.append(vectors)
        return vector_list

    def get_info(self):
        return self.goals, self.highprio, self.start_point

    def get_goals(self):
        return self.goals

    # [[[]]]
    def convert_to_longer_strokes(self, vectorList, grid, robotPosition):
        longerStrokes = []
        longerX, longerY = 0, 0
        xAxisLength = len(grid)
        yAxisLength = len(grid[0])

        for i in range(len(vectorList)):
            if len(vectorList[i]) != 1:
                longerX += vectorList[i][0][0]
                longerY += vectorList[i][0][1]
                prev = vectorList[i][0]
                for j in range(len(vectorList[i]) - 1):
                    if vectorList[i][j + 1][0] != prev[0] * - 1 or vectorList[i][j + 1][1] != prev[1] * - 1:
                        longerX += vectorList[i][j + 1][0]
                        longerY += vectorList[i][j + 1][1]
                        prev = vectorList[i][j + 1]
                    else:
                        if (robotPosition[0] <= (xAxisLength * 4.5 / 12) or robotPosition[0] >= (
                                xAxisLength * 7.5 / 12)) and (
                                robotPosition[1] <= (yAxisLength * 7.5 / 18) or robotPosition[0] >= (
                                yAxisLength * 10.5 / 12)):
                            print("I go X")
                            if longerX != 0:
                                longerStrokes.append([longerX, 0])
                            if longerY != 0:
                                longerStrokes.append([0, longerY])
                        else:
                            if longerX != 0:
                                longerStrokes.append([0, longerY])
                            if longerY != 0:
                                longerStrokes.append([longerX, 0])
                        longerX = vectorList[i][j + 1][0]
                        longerY = vectorList[i][j + 1][1]
                        prev = vectorList[i][j + 1]
                if (robotPosition[0] <= (xAxisLength * 4.5 / 12) or robotPosition[0] >= (
                        xAxisLength * 7.5 / 12)) and (robotPosition[1] <= (yAxisLength * 7.5 / 18) or
                                                      robotPosition[0] >= (yAxisLength * 10.5 / 12)):
                    if longerX != 0:
                        longerStrokes.append([longerX, 0])
                    if longerY != 0:
                        longerStrokes.append([0, longerY])
                else:
                    if longerX != 0:
                        longerStrokes.append([0, longerY])
                    if longerY != 0:
                        longerStrokes.append([longerX, 0])
                longerX = 0
                longerY = 0

        longerStrokes = [longerStrokes]
        print("longer vectors:" + str(longerStrokes))
        return longerStrokes


