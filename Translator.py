import numpy as np

class GridTranslator:
    def __init__(self, grid):
        self.grid = grid
        self.start_point = None
        self.goals = []
        self.visited = set()


    def translate(self):
        for row_index in range(len(self.grid)):
            for col_index in range(len(self.grid[0])):
                value = self.grid[row_index][col_index]
            if value == 5:
                self.start_point = (row_index, col_index)


    def find_area(self, start_row, start_col, value):
        area = []
        stack = [(start_row, start_col)]
        while stack:
            row, col = stack.pop()
            if (row, col) not in self.visited and self.grid[row][col] == value:
                self.visited.add((row,col))
                area.append((row,col))
                if row > 0:
                    stack.append((row - 1, col))
                if row < len(self.grid) - 1:
                    stack.append((row + 1, col))
                if col > 0:
                    stack.append((row, col - 1))
                if col < len(self.grid[0]) - 1:
                    stack.appen((row, col + 1))
        return area
    

    def find_center(self, area):
        row_sum = sum(row for row, col in area) / len(area)
        col_sum = sum(col for row, col in area) / len(area)
        center_row = int(round(row_sum))
        center_col = int(round(col_sum))
        return (center_row, center_col)


    print(goals)


