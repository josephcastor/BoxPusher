
class Human():
    def __init__(self, position, holding_box, tiredness, productivity, lr_bias):
        self.position = position
        self.holding_box = holding_box
        self.tiredness = tiredness
        self.productivity =productivity
        self.lr_bias = lr_bias

    def __hash__(self):
        return hash((self.position, 
                     self.holding_box,
                     self.tiredness,
                     self.productivity,
                     self.lr_bias))
    
    def __eq__(self, other):
        if isinstance(other, Human):
            return self.position == other.position\
                and self.holding_box == other.holding_box\
                and self.tiredness == other.tiredness\
                and self.productivity == self.productivity\
                and self.lr_bias == other.lr_bias
        return False
    
    def __str__(self):
        pass

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
        pass


class State():
    def __init__(self, human, robot, belt, packed, missed):
        self.human = human
        self.robot = robot
        self.belt = belt
        self.packed = packed
        self.missed = missed

    def __hash__(self):
        return hash((self.human, 
                     self.robot,
                     self.belt,
                     self.packed,
                     self.missed))
    
    def __eq__(self, other):
        if isinstance(other, State):
            return self.human == other.human\
                and self.robot == other.robot\
                and self.belt == other.belt\
                and self.packed == self.packed\
                and self.missed == other.missed
        return False
    
    def __str__(self):
        pass


class Action():
    def __init__(self, human_action, robot_action):
        self.human_action = human_action
        self.robot_action = robot_action
class AtomicAction():
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    PICKUP = 4
    PUTDOWN = 5

is_movement_action = dict()
is_movement_action[AtomicAction.UP] = True
is_movement_action[AtomicAction.RIGHT] = True
is_movement_action[AtomicAction.DOWN] = True
is_movement_action[AtomicAction.LEFT] = True
is_movement_action[AtomicAction.PICKUP] = False
is_movement_action[AtomicAction.PUTDOWN] = False


# TODO:
# Make observation match the structure of State
class Observation():

    def __init__(self, tiredness, productivity, lr_bias):
        self.tiredness = tiredness
        self.productivity = productivity
        self.lr_bias = lr_bias

    def __hash__(self):
        return hash((self.tiredness, self.productivity, self.lr_bias))

    def __eq__(self, other):
        if isinstance(other, Observation):
            return self.tiredness == other.tiredness\
                and self.productivity == other.productivity\
                and self.lr_bias == other.lr_bias
        return False


class Transition():
    def __init__(self, gridmap):
        self.gridmap = gridmap
    def probability(self, current_state, resultant_state, action):
        robot_action = action.robot_action
        human_action = action.human_action

        # If human action does not return deterministic result, return 0

        # Robot
        # If action is a movement action
        if (is_movement_action[robot_action]):
            if (self.get_intended(current_state.robot_position, robot_action) == resultant_state.robot_position):
                robot_prob = 0.8
            elif (current_state.robot_position == resultant_state.robot_position):
                robot_prob = 0.2
            else:
                robot_prob = 0
        else:
            if robot_get_intended_pickup():
                robot_prob = 1
            else:
                robot_prob = 0

        return robot_prob

    def sample(self, current_state, action):

        # Generate number between 0 and 1

        robot_action = action.robot_action
        human_action = action.human_action

        # Apply the human action to the current state (deterministic)
        next_human_state = apply_human_action(current_state)

        # Sample next robot state based on underlying distribution

        # Generate no. between 0 and 1

        # If action is a movement

            # If no. <= 0.2
            future_robot_state = current_robot_state

            # If > 0.8, return same robo
            future_robot_state = get_intended(...)

        # If action is a pickup or putdown

        future_robot_state = apply_pickupputdown(current-robot_state)

        return combination of future robot andf future human


    def apply_human_action(current_state, human_action):

        if is_movement_action[human_action]




    def get_intended(self, robot_position, action):
        if action == AtomicAction.UP:
            if robot_position[1] != self.gridmap.rows - 1:
                return (robot_position[0], robot_position[1] + 1)
            else:
                return robot_position

        elif action == AtomicAction.DOWN:
            if robot_position[1] != 0:
                return (robot_position[0], robot_position[1] - 1)
            else:
                return robot_position


        elif action == AtomicAction.RIGHT:
            if robot_position[0] != self.gridmap.columns - 1:
                return (robot_position[0] + 1, robot_position[1])
            else:
                return robot_position

        elif action == AtomicAction.LEFT:
            if robot_position[0] != 0:
                return (robot_position[0] - 1, robot_position[1])
            else:
                return robot_position










        # Probability of resultant_state.robot_position = intended_next_state(current_state.robot_position) = 0.8
        # Probability of resultant_state.robot_position = current_state.robot_position = 0.2

        # No uncertainty in pickup and putdown

        # Now we have robot probability

        # Human
















