import math
import numpy as np

from mcts import Position
from mcts import VacuumEnvironmentState
from mcts import Human
from mcts import Robot
from mcts import belt_positions
from mcts import belt_to_index



def show_state(state: VacuumEnvironmentState):
    width = len(state.belt)
    height = 4

    rows = 4 * height + 1
    columns = 6 * width + 1

    empty_display = np.full((rows, columns), ' ', dtype=str)

    for row_number in range(rows):
        for column_number in range(columns):
            if column_number % 6 == 0 or row_number % 4 == 0:
                if row_number <= 4:
                    empty_display[row_number][column_number] = '#'
                elif row_number <= 8:
                    empty_display[row_number][column_number] = '·'
                elif column_number >= columns/2 - 4 and column_number <= columns/2 + 3:
                    empty_display[row_number][column_number] = '·'
                elif row_number == rows - 1 or column_number == 0 or column_number == columns - 1:
                    empty_display[row_number][column_number] = '·'

            if column_number >= columns/2 - 3 and column_number <= columns/2 + 2:
                if row_number >= rows - 4 and row_number != rows - 1 and column_number:
                    empty_display[row_number][column_number] = '$'

    # for r in range(-2, 3):
    #     for c in range(-3, 4):
    #         empty_display[row_loc(height-1) + r][col_loc(math.floor(width / 2)) + c] = '$'


    human_position = state.human.position
    human_row, human_column = position_to_coords(human_position, True, width)
    robot_position = state.robot.position
    robot_row, robot_column = position_to_coords(robot_position, False, width)

    display = empty_display.copy()
    display = show_human(display, human_row, human_column, state.human.holding_box)
    display = show_robot(display, robot_row, robot_column, state.robot.holding_box)
    display = show_boxes(display, state.belt)
    # print("packed: "+state.packed)
    print_grid(display)
    return


def show_boxes(display, belt):
    for i in range(len(belt)):
        if belt[i] ==1:
            for j in range(2, 5):
                display[1][i * 6 + j] = "X"
                display[2][i * 6 + 2] = "X"
                display[2][i * 6 + 4] = "X"
                display[3][i * 6 + j] = "X"
    return display
def position_to_coords(position, is_human, width):
    if position in belt_positions:
        index = belt_to_index(position)
        return 1, index
    elif position == Position.DROP_OFF:
        return 2, int(width/2)
    elif position == Position.REST_POSITION:
        if is_human:
            return 3, 0
        else:
            return 3, width - 1

def show_robot(display, robot_row, robot_column, holding):
    for i in range(-1, 2):
        for j in range(-1, 2):
            if not (abs(i) == abs(j)):
                display[row_loc(robot_row) + i][col_loc(robot_column) + j] = 'R'
            elif i == 0 and j == 0 and holding:
                display[row_loc(robot_row) + i][col_loc(robot_column) + j] = 'X'


    # display[row_loc(robot_row)][col_loc(robot_column)] = 'R'
    return display

def show_human(display, human_row, human_column, holding):
    for i in range(-1, 2):
        for j in range(-1, 2):
            if not abs(i) == abs(j):
                display[row_loc(human_row) + i][col_loc(human_column) + j] = 'H'
            elif i == 0 and j == 0 and holding:
                display[row_loc(human_row) + i][col_loc(human_column) + j] = 'X'


    # display[row_loc(robot_row)][col_loc(robot_column)] = 'R'
    return display


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


# if __name__ == "__main__":
#     # visualiser()
#     human = Human(Position.PICKUP_3, False, 1, 3)
#     robot = Robot(Position.PICKUP_5, False)
#     state = VacuumEnvironmentState(human, robot, np.array([0,1,0,1,0]), 0,0, "whatevs", 1)
#
#     show_state(state)