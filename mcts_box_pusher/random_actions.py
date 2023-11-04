import math
import time
import random
import numpy as np
import copy

exploration_constant = 10

STEPS_PER_TRIAL = 30

random_box_order = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0,
       1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1,
       1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1]
    

class Human():
    def __init__(self, position, holding_box, tiredness, lr_bias):
        self.position = position
        self.holding_box = holding_box
        self.tiredness = tiredness
        self.lr_bias = lr_bias

    def __hash__(self):
        return hash((self.position,
                     self.holding_box,
                     self.tiredness,
                     self.lr_bias))

    def __eq__(self, other):
        if isinstance(other, Human):
            return self.position == other.position\
                and self.holding_box == other.holding_box\
                and self.tiredness == other.tiredness\
                and self.lr_bias == other.lr_bias
        return False

    def __str__(self):
        return f"Human(position={self.position}, holding_box={self.holding_box}, tiredness={self.tiredness}, lr_bias={self.lr_bias})"


class Robot():
    def __init__(self, position, holding_box):
        self.position = position
        self.holding_box = holding_box

    def __hash__(self):
        return hash((self.position,
                     self.holding_box))

    def __eq__(self, other):
        if isinstance(other, Human):
            return self.position == other.position\
                and self.holding_box == other.holding_box
        return False

    def __str__(self):
        return f"Robot(position={self.position}, holding_box={self.holding_box})"

# In transition fn, update tiredness if we rested previously

class Position():
    PICKUP_1 = 0
    PICKUP_2 = 1
    PICKUP_3 = 2
    PICKUP_4 = 3
    PICKUP_5 = 4
    DROP_OFF = 5
    REST_POSITION = 6


belt_positions = [Position.PICKUP_1, Position.PICKUP_2, Position.PICKUP_3, Position.PICKUP_4, Position.PICKUP_5]


# class Action():
#     def __init__(self, human_action, robot_action):
#         self.human_action = human_action
#         self.robot_action = robot_action

#     def __hash__(self):
#         return hash((self.human_action,
#                      self.robot_action))

#     def __eq__(self, other):
#         if isinstance(other, Action):
#             return self.human_action == other.human_action\
#                 and self.robot_action == other.robot_action
#         return False

#     def __str__(self):
#         return f"Action(human_action={self.human_action}, robot_action={self.robot_action})"

class AtomicAction():
    GOTO_P1 = 0
    GOTO_P2 = 1
    GOTO_P3 = 2
    GOTO_P4 = 3
    GOTO_P5 = 4
    GOTO_DROPOFF = 5
    REST = 6
    PICKUP = 7
    PUTDOWN = 8

go_to_belt_actions = [AtomicAction.GOTO_P1, AtomicAction.GOTO_P2, AtomicAction.GOTO_P3, AtomicAction.GOTO_P4, AtomicAction.GOTO_P5]

atomic_actions = [AtomicAction.GOTO_P1, AtomicAction.GOTO_P2, AtomicAction.GOTO_P3, AtomicAction.GOTO_P4, AtomicAction.GOTO_P5, AtomicAction.GOTO_DROPOFF, AtomicAction.PICKUP, AtomicAction.PUTDOWN, AtomicAction.REST]

index_to_actions = ['GOTO_P1','GOTO_P2','GOTO_P3','GOTO_P4','GOTO_P5','GOTO_DROPOFF','REST', 'PICKUP','PUTDOWN']
class VacuumEnvironmentState:
    """State representation for a vacuum cleaning robot in a grid environment."""
    def __init__(self, human, robot, belt, packed, missed, time_step, time_since_rest):
        self.human = human
        self.robot = robot
        self.belt = belt
        self.packed = packed # count of packed packages
        self.missed = missed # count of missed missed
        self.time_step = time_step
        self.time_since_rest = time_since_rest


    def __hash__(self):
        return hash((self.human,
                     self.robot,
                     tuple(self.belt.tolist()),
                     self.packed,
                     self.missed))

    def __eq__(self, other):
        if isinstance(other, VacuumEnvironmentState):
            return self.human == other.human\
                and self.robot == other.robot\
                and self.belt == other.belt\
                and self.packed == self.packed\
                and self.missed == other.missed\
                and self.time_step == other.time_step\
                and self.time_since_rest == other.time_since_rest


        return False

    def __str__(self):
        return f"State(human={self.human}, robot={self.robot}, belt={self.belt}, packed={self.packed}, missed={self.missed}, time={self.time_step}, since_rest={self.time_since_rest})"


    def is_terminal(self):
        return False
        # if self.human.holding_box or self.robot.holding_box or (1 in self.belt):
        #     return False
        # return True


    def get_legal_actions(self):
        """Computes and returns a list of legal actions from the current state."""
        legal_human = list(range(len(self.belt) + 2))
        legal_robot = list(range(len(self.belt) + 2))

        if self.human.holding_box and self.human.position == Position.DROP_OFF:
            legal_human.append(AtomicAction.PUTDOWN)
        if self.human.position <= 4 and not self.human.holding_box and self.belt[self.human.position] == 1:
            legal_human.append(AtomicAction.PICKUP)
        if self.robot.holding_box and self.robot.position == Position.DROP_OFF:
            legal_robot.append(AtomicAction.PUTDOWN)
        if self.robot.position <= 4 and not self.robot.holding_box and self.belt[self.robot.position] == 1:
            legal_robot.append(AtomicAction.PICKUP)
        return [(a,b) for a in legal_human for b in legal_robot]


    def take_action(self, action, is_executing):
        """Returns a new state after applying the given action."""

        human_action = action[0]
        robot_action = action[1]
        human_rest_prob = self.human.tiredness / 10

        resultant_state = VacuumEnvironmentState(copy.copy(self.human), copy.copy(self.robot), copy.copy(self.belt), copy.copy(self.packed), copy.copy(self.missed), self.time_step + 1, copy.copy(self.time_since_rest))
        
        if human_action == AtomicAction.REST:
            resultant_state.human.position = Position.REST_POSITION
        elif human_action == AtomicAction.GOTO_DROPOFF:
            rand = np.random.rand()
            if not rand <= human_rest_prob:
            #     resultant_state.human.position = Position.REST_POSITION
            # else:
                resultant_state.human.position = Position.DROP_OFF

        elif human_action == AtomicAction.PUTDOWN:
            if not (self.human.holding_box and self.human.position == Position.DROP_OFF):
                pass
                # resultant_state.human.position = Position.REST_POSITION
            else:
                rand = np.random.rand()
                if not rand <= human_rest_prob:
                #     resultant_state.human.position = Position.REST_POSITION
                # else:
                    resultant_state.human.position = Position.DROP_OFF
                    resultant_state.human.holding_box = False
                    resultant_state.packed += 1

        elif human_action == AtomicAction.PICKUP:
            if not (self.human.position in belt_positions and not self.human.holding_box):
                pass
                # resultant_state.human.position = Position.REST_POSITION
            else:
                index = belt_to_index(self.human.position)
                if not self.belt[index] == 0:
                #     resultant_state.human.position = Position.REST_POSITION
                # else:
                    resultant_state.human.holding_box = True
                    resultant_state.belt[index] = 0

        elif human_action in go_to_belt_actions:
            rand = np.random.rand()
            if rand <= human_rest_prob:
                pass
                # resultant_state.human.position = Position.REST_POSITION
            else:
                intended = human_action

                bias_difference = self.human.lr_bias - intended

                if bias_difference == 0:
                    resultant_state.human.position = belt_positions[intended]
                elif bias_difference > 0:
                    probs = []
                    prob = 0.0
                    for i in range(intended, self.human.lr_bias + 1):
                        if i == intended:
                            prob += 0.5
                        prob += 0.5*(1 / (abs(bias_difference) + 1))
                        probs.append((i, prob))
                    rand = np.random.rand()

                    for i in range(len(probs) - 1):
                        if rand <= probs[i][1]:
                            resultant_state.human.position = belt_positions[probs[i][0]]


                elif bias_difference < 0:
                    probs = []
                    prob = 0
                    for i in range(self.human.lr_bias, intended + 1):
                        if i == intended:
                            prob += 0.5
                        prob += 0.5 * (1 / (abs(bias_difference) + 1))
                        probs.append((i, prob))
                    rand = np.random.rand()

                    for i in range(len(probs) - 1):
                        if rand <= probs[i][1]:
                            resultant_state.human.position = belt_positions[probs[i][0]]



        # ROBOT TIME:
        if robot_action == AtomicAction.REST:
            resultant_state.robot.position = Position.REST_POSITION
        elif robot_action == AtomicAction.GOTO_DROPOFF:
            if resultant_state.human.position == Position.DROP_OFF:
                resultant_state.robot.position = Position.REST_POSITION
            else:
                resultant_state.robot.position = Position.DROP_OFF

        elif robot_action == AtomicAction.PICKUP:
            if not (self.robot.position in belt_positions and not self.robot.holding_box):
                resultant_state.robot.position = Position.REST_POSITION
            else:
                index = belt_to_index(self.robot.position)
                # No box at belt location:
                if self.belt[index] == 0 or resultant_state.human.position == belt_positions[index]:
                    resultant_state.robot.position = Position.REST_POSITION
                else:
                    resultant_state.robot.holding_box = True
                    resultant_state.belt[index] = 0

        elif robot_action == AtomicAction.PUTDOWN:
            if not (self.robot.holding_box and self.robot.position == Position.DROP_OFF) \
                    or resultant_state.human.position == Position.DROP_OFF:
                resultant_state.robot.position = Position.REST_POSITION
            else:
                resultant_state.robot.holding_box = False
                resultant_state.packed += 1

        elif robot_action in go_to_belt_actions:
            index = go_to_belt_actions.index(robot_action)
            rand = np.random.rand()
            if index == 0:
                if rand <= 0.9:
                    if resultant_state.human.position == belt_positions[index]:
                        resultant_state.robot.position = Position.REST_POSITION
                    else:
                        resultant_state.robot.position = belt_positions[index]
                else:
                    if resultant_state.human.position == belt_positions[index + 1]:
                        resultant_state.robot.position = Position.REST_POSITION
                    else:
                        resultant_state.robot.position = belt_positions[index + 1]
            elif index == len(belt_positions) - 1:
                if rand <= 0.9:
                    if resultant_state.human.position == belt_positions[index]:
                        resultant_state.robot.position = Position.REST_POSITION
                    else:
                        resultant_state.robot.position = belt_positions[index]
                else:
                    if resultant_state.human.position == belt_positions[index - 1]:
                        resultant_state.robot.position = Position.REST_POSITION
                    else:
                        resultant_state.robot.position = belt_positions[index - 1]
            else:
                if rand <= 0.8:
                    if resultant_state.human.position == belt_positions[index]:
                        resultant_state.robot.position = Position.REST_POSITION
                    else:
                        resultant_state.robot.position = belt_positions[index]
                elif rand <= 0.9:
                    if resultant_state.human.position == belt_positions[index - 1]:
                        resultant_state.robot.position = Position.REST_POSITION
                    else:
                        resultant_state.robot.position = belt_positions[index - 1]
                else:
                    if resultant_state.human.position == belt_positions[index + 1]:
                        resultant_state.robot.position = Position.REST_POSITION
                    else:
                        resultant_state.robot.position = belt_positions[index + 1]

        # Ensuring the missed count has been updated correctly:
        if resultant_state.belt[0] == 1:
            # New missed package:
            resultant_state.missed += 1

        # print(resultant_state.belt)
        # Shift belt to the left:
        for i in range(len(resultant_state.belt) - 1):
            resultant_state.belt[i] = resultant_state.belt[i + 1]

        if is_executing:
            resultant_state.belt[-1] = random_box_order[self.time_step]


        else:
            prob = random.random()
            if prob <= 0.8:
                resultant_state.belt[-1] = 1
            else:
                resultant_state.belt[-1] = 0

        # Update tiredness of human:
        if resultant_state.human.position == Position.REST_POSITION:
            resultant_state.human.tiredness = max(0 , resultant_state.human.tiredness - 3)
            resultant_state.time_since_rest = 0
        else:
            l = 0.2
            resultant_state.human.tiredness = int(round(4 * (1 - np.e**(-l * self.time_since_rest))))
            resultant_state.time_since_rest += 1

        return resultant_state


    def get_reward_for_action(self, action, next_state):
        """Computes the reward for taking an action that leads to a new state."""
        reward = -1
        # if next_state.packed > self.packed:
        #     reward += 100
        reward += 10*next_state.packed / self.time_step
        # if next_state.human.holding_box and not self.human.holding_box:
        #     reward += 50
        # if next_state.robot.holding_box and not self.robot.holding_box:
        #     reward += 50
        return reward
    


    def get_reward_for_terminal(self):
        """Returns the reward when all cells are clean."""
        return 50

    
    
def simulate_mcts_run(planning_duration, trial_num):
    # Initialise grid with randomly placed obstacles

    human1 = Human(Position.PICKUP_1,False, 0, 0)
    robot = Robot(Position.REST_POSITION, False)
    init_true_state = VacuumEnvironmentState(human1, robot, np.array([0,0,0,0,0]), 0,0, 1, 0)

    # Initialise the type of flooring for each grid cell (all set to 'VINYL').
    
    cumulative_reward = 0
    # Create an initial state for the vacuum environment.
    current_state = init_true_state
    steps = 0
    
    print(f"\nTrial {trial_num + 1} for belt size 5:")
    i = 0
    while i < STEPS_PER_TRIAL:
        i += 1 
        # Initialise a new tree node with the current state.
        root = current_state

        print(current_state)
        show_state(current_state)
        # best_action = root.get_best_action()

        # Printing breakdown of actions:
        # tuple_actions = [(a,b) for a in range(9) for b in range(9)]
        # aggregated_visits = dict(zip(tuple_actions, [0 for _ in range(81)]))
        # for n in root.children:
        #     aggregated_visits[n.action] += n.visit_count
        # print(aggregated_visits)

        a1 = np.random.randint(0, len(current_state.belt) + 4)
        a2 = np.random.randint(0, len(current_state.belt) + 4)
        best_action = (a1, a2)
        
        print("Human action taken:", index_to_actions[best_action[0]])
        print("Robot action taken:", index_to_actions[best_action[1]])
        
        next_state = current_state.take_action(best_action, True)
        reward = current_state.get_reward_for_action(best_action, next_state )
        print("Reward: ", reward)
        cumulative_reward += reward 

        current_state = next_state

        # # Ensuring the missed count has been updated correctly:
        # if current_state.belt[0] == 1:
        #     # New missed package:
        #     print(current_state.belt)

        steps += 1

    print("End of trial.")
    print("Cumulative reward: ", cumulative_reward)
    return cumulative_reward

def run_experiments(planning_duration=5, num_trials=50):
    results = []

    for n in range(1):
        # Run multiple trials for each grid size and store the data.
        trials_data = [simulate_mcts_run(planning_duration, i) for i in range(num_trials)]
        
        # Extract the number of steps and time taken for each trial.

        results.append(trials_data)

    print(results)

def display_results(results):
    print("\nResults:")
    print(f"{'Size':<10}{'Median Steps':<15}{'Median Time Taken':<25}{'Std Dev Steps':<20}{'Std Dev Time':<20}")
    print("-" * 90)
    for n, median_steps, median_time, std_steps, std_time in results:
        print(f"{n:<10}{median_steps:<15.2f}{median_time:<25.2f}{std_steps:<20.2f}{std_time:<20.2f}")

def main():
    run_experiments()


def belt_to_index(position):
    for i in range(len(belt_positions)):
        if position == belt_positions[i]:
            return i
    return -1
#
#


# VISUALISATION THINGS:
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
                elif column_number >= columns / 2 - 4 and column_number <= columns / 2 + 3:
                    empty_display[row_number][column_number] = '·'
                elif row_number == rows - 1 or column_number == 0 or column_number == columns - 1:
                    empty_display[row_number][column_number] = '·'

            if column_number >= columns / 2 - 3 and column_number <= columns / 2 + 2:
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
        if belt[i] == 1:
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
        return 2, int(width / 2)
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
    return 2 + 4 * x


def col_loc(x):
    return 3 + 6 * x




if __name__ == '__main__':
    main()










