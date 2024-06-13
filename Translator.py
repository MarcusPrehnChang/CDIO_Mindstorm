import numpy as np

def translate(array):
    goals = []
    highprio_goal = []
    for i in range(len(array)):
        for j in range(len(array[i])):
            if array[i][j] == 2:
                goals.append((i,j))
            elif array[i][j] == 3:
                highprio_goal.append((i,j))
                
    return goals, highprio_goal


