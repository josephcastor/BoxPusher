import math
import time
import random
import numpy as np
import copy

exploration_constant = 1000

class TreeNode:
    """A node class for Monte Carlo Tree Search."""
    
    def __init__(self, state, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.action_values = self.initial_action_values()
        self.action_visits = self.initial_action_visits()
        self.children = []
        self.cumulative_reward = 0
        self.visit_count = 1

    def initial_action_values(self):
        action_values = dict()
        for action in self.state.get_legal_actions():
            action_values[action] = 0
        return action_values

    def initial_action_visits(self):
        action_visits = dict()
        for action in self.state.get_legal_actions():
            action_visits[action] = 0
        return action_visits

    def is_fully_expanded(self):
        """Checks if all possible child nodes (actions) have been explored."""
        
        return len(self.children) == len(self.state.get_legal_actions())

    def select_action(self):
        max_score = None
        chosen_action = None
        # Pick the action with the maximum score using UCB1
        for a in self.action_visits.keys():
            visited = self.action_visits[a]
            score = self.action_values[a]
            if visited == 0:
                exploitation = 0
            else:
                exploitation = score / visited
            # UCB1 exploration and exploitation formula
            exploration = exploration_constant * math.sqrt(math.log(self.visit_count) / max(float(self.action_visits[a]), 0.1))
            score = exploitation + exploration
            if max_score is None or score > max_score:
                max_score = score
                chosen_action = a

        return chosen_action
    def get_best_action(self):
        max_score = None
        chosen_action = None
        # Pick the action with the maximum score using UCB1
        for a in self.action_visits.keys():
            visited = self.action_visits[a]
            score = self.action_values[a]
            if visited == 0:
                exploitation = 0
            else:
                exploitation = score / visited

            if max_score is None or exploitation > max_score:
                max_score = exploitation
                chosen_action = a

        return chosen_action


    
    def traverse_tree(self):
        """Traverses the tree to select a child node."""
        
        # Start from current node
        # Selection and Expansion:
        found_new_child = False
        while not (found_new_child or self.state.is_terminal()):
            # SELECT an action edge to explore using Multi-Arm Bandit:
            action = self.select_action()
            # Simulate applying the action:
            resultant_state = self.state.take_action(action)

            child_node = None
            for child in self.children:
                if child.parent_action == action and child.state == resultant_state:
                    child_node = child
                    break
            # SELECTED action has not been expanded from this state, so EXPAND:
            if child_node is None:
                child_node = TreeNode(resultant_state, self, action)
                self.children.append(child_node)
                found_new_child = True

            node = child_node

        return node


    def update_rewards(self, reward, discount):
        """Backpropagates and updates the reward values."""

        # Start from current node and move up the tree

        current_node = self
        while current_node.parent != None:
            # Update visit count and cumulative reward for the node
            current_node.parent.action_values[current_node.parent_action] += reward
            current_node.parent.visit_count += 1
            current_node.parent.action_visits[current_node.parent_action] += 1

            current_node = current_node.parent # Move to the parent node
            reward *= discount

class MonteCarloTreeSearch:
    """
    A class that represents the Monte Carlo Tree Search algorithm.
    """
    
    def __init__(self, root_node, planning_duration=5, heuristic=False):
        """
        Initialises the MCTS with a root node, planning duration, and an exploration factor.
        """
        self.root_node = root_node
        self.planning_duration = planning_duration
        self.heuristic = heuristic

    def run_search(self):
        """
        Executes the MCTS algorithm for a specified duration.
        """
        end_time = time.time() + self.planning_duration
        while time.time() < end_time:
            # Step 1: Selection
            selected_node = self.root_node.traverse_tree()



            # Check if selected node is terminal
            if not selected_node.state.is_terminal():
                # Step 2: Expansion
                # new_child_node = selected_node.expand_child_node()

                # Step 3: Simulation
                simulation_reward = self.run_simulation(selected_node, self.heuristic)

                # Step 4: Backpropagation
                selected_node.update_rewards(simulation_reward, 0.85)
            # else:
            #     simulation_reward = 0
            #     # If node is terminal, just backpropagate its reward
            #     # terminal_reward = selected_node.state.get_reward_for_terminal()
            #     # selected_node.update_rewards(terminal_reward)
        
        # After MCTS completes, return the best child node of the root based on value function
        print("Joseph")
        # Printing action exploration:
        for a in self.root_node.action_visits.keys():
            print("visits: "+str(index_to_actions[a[0]])+","+str(index_to_actions[a[1]])+":   "+str(self.root_node.action_visits[a]))
            print("value: "+str(index_to_actions[a[0]])+","+str(index_to_actions[a[1]])+":   "+str(self.root_node.action_values[a]))
            print("score: "+str(index_to_actions[a[0]])+","+str(index_to_actions[a[1]])+":   "+str(self.root_node.action_values[a] / self.root_node.action_visits[a]))


        return self.root_node.get_best_action()


    def get_greedy_action(self, state):

        if state.human.tiredness >= 3:
            human_action = AtomicAction.REST
        if state.human.position in belt_positions and not state.human.holding_box and state.belt[state.human.position] == 1:
            human_action = AtomicAction.PICKUP
        elif state.human.holding_box and not state.human.position == Position.DROP_OFF:
            human_action = AtomicAction.GOTO_DROPOFF
        elif state.human.position == Position.DROP_OFF and state.human.holding_box:
            human_action = AtomicAction.PUTDOWN
        else:
            # Best future pickup
            future_boxes = [x for x in state.belt[1:]]
            future_boxes.append(1)
            # Picking closest to lr bias:
            closest = None
            closest_difference = None
            for i in range(len(future_boxes)):
                if future_boxes[i] == 1:
                    difference = abs(i - state.human.lr_bias)
                    if closest_difference is None or difference < closest_difference:
                        closest = i
                        closest_difference = difference
            if closest is not None:
                human_action = go_to_belt_actions[closest]
            else:
                human_action = len(belt_positions) - 1

        # Robot time:
        # Best future pickup
        # Picking closest to lr bias:

        if state.robot.position in belt_positions and not state.robot.holding_box and state.belt[
            state.robot.position] == 1:
            robot_action = AtomicAction.PICKUP
        elif state.robot.holding_box and not state.robot.position == Position.DROP_OFF:
            robot_action = AtomicAction.GOTO_DROPOFF
        elif state.robot.position == Position.DROP_OFF and state.robot.holding_box:
            robot_action = AtomicAction.PUTDOWN
        else:
            future_boxes = [x for x in state.belt[1:]]
            future_boxes.append(1)
            for i in range(len(belt_positions)):
                if i != human_action and future_boxes[i] == 1:
                    robot_action = i
            else:
                for i in reversed(range(len(belt_positions))):
                    if i != human_action:
                        robot_action = i

        return human_action, robot_action






    def run_simulation(self, starting_node, heuristic):
        """
        Simulates a random trajectory from the given node until a terminal state is reached.
        """
        current_state = starting_node.state
        cumulative_reward = 0

        # Randomly traverse the game tree until a terminal state
        for _ in range(20):
            # Select a random action
            # legal_actions = current_state.get_legal_actions()

            if heuristic:
                chosen_action = self.get_greedy_action(current_state)
            else:
                chosen_action = random.choice(current_state.get_legal_actions())
            # Transition to the next state
            next_state = current_state.take_action(chosen_action) 
            # Get the reward for the transition
            transition_reward = current_state.get_reward_for_action(chosen_action, next_state) 
            # Accumulate rewards
            cumulative_reward += transition_reward 
            current_state = next_state
        return cumulative_reward
    

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
    PICKUP_6 = 5
    PICKUP_7 = 6
    DROP_OFF = 7
    REST_POSITION = 8


belt_positions = [Position.PICKUP_1, Position.PICKUP_2, Position.PICKUP_3, Position.PICKUP_4, Position.PICKUP_5, Position.PICKUP_6, Position.PICKUP_7]



class AtomicAction():
    GOTO_P1 = 0
    GOTO_P2 = 1
    GOTO_P3 = 2
    GOTO_P4 = 3
    GOTO_P5 = 4
    GOTO_P6 = 5
    GOTO_P7 = 6
    GOTO_DROPOFF = 7
    REST = 8
    PICKUP = 9
    PUTDOWN = 10

go_to_belt_actions = [AtomicAction.GOTO_P1, AtomicAction.GOTO_P2, AtomicAction.GOTO_P3, AtomicAction.GOTO_P4, AtomicAction.GOTO_P5, AtomicAction.GOTO_P6, AtomicAction.GOTO_P7]

atomic_actions = [AtomicAction.GOTO_P1, AtomicAction.GOTO_P2, AtomicAction.GOTO_P3, AtomicAction.GOTO_P4, AtomicAction.GOTO_P5, AtomicAction.GOTO_P6, AtomicAction.GOTO_P7, AtomicAction.GOTO_DROPOFF, AtomicAction.PICKUP, AtomicAction.PUTDOWN, AtomicAction.REST]

index_to_actions = ['GOTO_P1','GOTO_P2','GOTO_P3','GOTO_P4','GOTO_P5','GOTO_P6', 'GOTO_P7', 'GOTO_DROPOFF','REST', 'PICKUP','PUTDOWN']
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


    def take_action(self, action):
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
                resultant_state.human.position = Position.DROP_OFF

        elif human_action == AtomicAction.PUTDOWN:
            if not (self.human.holding_box and self.human.position == Position.DROP_OFF):
                pass
            else:
                rand = np.random.rand()
                if not rand <= human_rest_prob:
                    resultant_state.human.position = Position.DROP_OFF
                    resultant_state.human.holding_box = False
                    resultant_state.packed += 1

        elif human_action == AtomicAction.PICKUP:
            if not (self.human.position in belt_positions and not self.human.holding_box):
                pass
            else:
                index = belt_to_index(self.human.position)
                if not self.belt[index] == 0:
                    resultant_state.human.holding_box = True
                    resultant_state.belt[index] = 0

        elif human_action in go_to_belt_actions:
            rand = np.random.rand()
            if rand <= human_rest_prob:
                pass
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
        resultant_state.belt[-1] = np.random.randint(0,2)

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

    # Set the initial robot position to the first row and a random column.

    human1 = Human(Position.PICKUP_1,False, 0, 0)
    robot = Robot(Position.REST_POSITION, False)
    init_true_state = VacuumEnvironmentState(human1, robot, np.array([0,1,0,0,0,1,0]), 0, 0, 1, 0)

    # Initialise the type of flooring for each grid cell (all set to 'VINYL').
    
    # Create an initial state for the vacuum environment.
    current_state = init_true_state
    steps = 0
    
    print(f"\nTrial {trial_num + 1}")
    i = 0
    while not current_state.is_terminal() and i < 10:
        i += 1 
        # Initialise a new tree node with the current state.
        root = TreeNode(current_state)

        # Initialise the MCTS with the root node, a specified planning duration, and exploration factor.
        mcts = MonteCarloTreeSearch(root, planning_duration=planning_duration, heuristic=True)

        
        print(current_state)
        show_state(current_state)
        # best_action = root.get_best_action()

        # Printing breakdown of actions:
        # tuple_actions = [(a,b) for a in range(9) for b in range(9)]
        # aggregated_visits = dict(zip(tuple_actions, [0 for _ in range(81)]))
        # for n in root.children:
        #     aggregated_visits[n.action] += n.visit_count
        # print(aggregated_visits)

        best_action = mcts.run_search()
        print("Human action taken:", index_to_actions[best_action[0]])
        print("Robot action taken:", index_to_actions[best_action[1]])
        current_state = current_state.take_action(best_action)

        # Ensuring the missed count has been updated correctly:
        if current_state.belt[0] == 1:
            # New missed package:
            print(current_state.belt)

        steps += 1

    print("Terminal State Reached:", current_state)
    return steps, planning_duration * steps

def run_experiments(planning_duration=5, num_trials=5):
    results = []

    for n in range(4, 8):
        # Run multiple trials for each grid size and store the data.
        trials_data = [simulate_mcts_run(planning_duration, i) for i in range(num_trials)]
        
        # Extract the number of steps and time taken for each trial.
        steps_list, time_list = zip(*trials_data)

        # Calculate the median number of steps and time taken for each trial.
        median_steps = sorted(steps_list)[num_trials // 2]
        median_time = sorted(time_list)[num_trials // 2]

        # Calculate the standard deviation of the number of steps and time taken.
        std_steps = (sum([(x - median_steps) ** 2 for x in steps_list]) / num_trials) ** 0.5
        std_time = (sum([(x - median_time) ** 2 for x in time_list]) / num_trials) ** 0.5
        
        results.append((n, median_steps, median_time, std_steps, std_time))

    display_results(results)

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

# VISUALISER STUFF:


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










