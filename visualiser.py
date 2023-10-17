import math
import numpy as np

def visualiser():
    width = 9
    height= 5

    rows = 4 * height + 1
    columns = 6 * width + 1
    empty_display = np.full((rows, columns), ' ', dtype=str)

    for row_number in range(rows):
        for column_number in range(columns):
            if column_number % 6 == 0 or row_number % 4 == 0:
                if row_number <= 4:
                    empty_display[row_number][column_number] = '#'
                else:
                    empty_display[row_number][column_number] = '·'
    for r in range(-1, 2):
        for c in range(-2, 3):
            empty_display[row_loc(height-1) + r][col_loc(math.floor(width / 2)) + c] = '$'
    # empty_display[row_loc(height-1)][col_loc(math.floor(width / 2))] = "G"

    # print_grid(empty_display)
    # state = initial_state
    human_row = 2
    human_column = 8
    boxes = {0, 3}
    while True:
        print()
        display = empty_display.copy()
        display = show_belt(display, boxes)
        display = show_robot(display, 4, 0)
        display = show_human(display, human_row, human_column)
        print_grid(display)
        print()
        next_move = input("Next Move: ")
        print()
        if next_move == "up":
            human_row = max(1, human_row - 1)
        elif next_move == "down":
            human_row = min(4, human_row + 1)
        elif next_move == "left":
            human_column = max(0, human_column - 1)
        elif next_move == "right":
            human_column = min(8, human_column + 1)
        boxes = update_boxes(boxes)


def update_boxes(boxes):
    new_boxes = set()
    for box in boxes:
        if box < 8:
            new_boxes.add(box + 1)
    return new_boxes



def show_belt(display, boxes):
    for box in boxes:
        for j in range(2, 5):
            display[1][box*6 + j] = "X"
            display[2][box*6 + 2] = "X"
            display[2][box * 6 + 4] = "X"
            display[3][box * 6 + j] = "X"

    return display

def show_robot(display, robot_row, robot_column):
    for i in range(-1, 2):
        for j in range(-1, 2):
            if not (abs(i) == abs(j)):
                display[row_loc(robot_row) + i][col_loc(robot_column) + j] = 'R'
    # display[row_loc(robot_row)][col_loc(robot_column)] = 'R'
    return display

def show_human(display, human_row, human_column):
    for i in range(-1, 2):
        for j in range(-1, 2):
            if abs(i) == abs(j):
                display[row_loc(human_row) + i][col_loc(human_column) + j] = 'H'
    # display[row_loc(robot_row)][col_loc(robot_column)] = 'R'
    return display





# def show_robot(display, state):
#     row = row_loc(state.robot_row)
#     col = col_loc(state.robot_column)
#     for i in range(-1, 2):
#         for j in range(-1, 2):
#             if not (i == 0 or j == 0):
#                 display[row + i][col + j] = 'r'
#     return display



def print_grid(display):
    for row in display:
        str = ''
        for char in row:
            str = str + char
        print(str)

def row_loc(x):
    return 2 + 4*x

def col_loc(x):
    return 3 + 6*x


if __name__ == "__main__":
    visualiser()