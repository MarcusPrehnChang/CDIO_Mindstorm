import numpy as np

def translate(array):
    goals = set()
    for i in range(array[0]):
        for j in range(array[1]):
            if array[i][j] == 2 or array[i][j] == 3:
                goals.add(i,j)
    return goals

