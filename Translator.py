import numpy as np

class GridTranslator:
    def __init__(self, grid):
        self.grid = grid
        self.start_point = None
        self.goals = []
        self.highprio = []
        self.visited = set()


    def translate(self):
        for row in self.grid:
            print(" ".join(map(str, row)))
        for row_index in range(len(self.grid)):
            for col_index in range(len(self.grid[0])):
                value = self.grid[row_index][col_index]
                if value == 5 and (row_index, col_index) not in self.visited:
                    start_area = self.find_area(row_index,col_index,value)
                    self.start_point = self.find_center(start_area)
                elif value == 2 and (row_index, col_index) not in self.visited:
                    start_area = self.find_area(row_index,col_index,value)
                    self.goals.append(self.find_center(start_area))
                elif value == 3 and (row_index, col_index) not in self.visited:
                    start_area = self.find_area(row_index,col_index,value)
                    self.highprio.append(self.find_center(start_area))


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
                    stack.append((row, col + 1))
        return area
    

    def find_center(self, area):
        row_sum = sum(row for row, _ in area) / len(area)
        col_sum = sum(col for _, col in area) / len(area)
        center_row = int(round(row_sum))
        center_col = int(round(col_sum))
        return (center_row, center_col)


    def get_shit(self):
        return self.goals, self.highprio, self.start_point
    
    def get_goals(self):
        return self.goals

