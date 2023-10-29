

# Fixed positions
#C1...CN
#REST
# DROPOFF


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
        self.packed = packed # count of packed packages 
        self.missed = missed # count of missed missed

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
    GOTO_P1 = 0
    GOTO_P2 = 1
    GOTO_P3 = 2
    GOTO_P4 = 3
    GOTO_P5 = 4
    GOTO_DROPOFF = 5
    GOTO_REST = 6
    PICKUP = 7
    PUTDOWN = 8

# TODO:
# Make observation match the structure of State
class Observation():

    def __init__(self, tiredness, lr_bias):
        self.tiredness = tiredness
        self.lr_bias = lr_bias

    def __hash__(self):
        return hash((self.tiredness, self.lr_bias))

    def __eq__(self, other):
        if isinstance(other, Observation):
            return self.tiredness == other.tiredness\
                and self.lr_bias == other.lr_bias
        return False

class Transition():
    def __init__(self, gridmap):
        self.gridmap = gridmap
    def probability(self, current_state, resultant_state, action):
        robot_action = action.robot_action
        human_action = action.human_action

        # Robot actions

        # NON DETERMINISM IN ROBOT's MOVEMENTS
        #P(s'|s,a)
        # If the human is at [beltcell] in the s':
            # P(robot goes to waiting) = 1
            # P(robot at [beltcell]) = 0

        # If the action is of type GOTO_[beltcell]
        # 80 % chance of landing up in GO_TO[beltcell]
        # 10 % chanceo f landing up in GO_TO[beltcell - 1]
        # 10 % chance of landing up in GO_TO[beltcell + 1]
        # Note: probability mass moves to TRUE cell if +1 or -1 is beyond boundary

        # SET DETERMINISM FOR OTHER ACTIONS
        # If action is dropoff:
            # If robot position is adjacent to dropoff box and robot is holding box:
                # P(true next state) = 1
                # True state will need to have an updated packed count
            # Else 
                # P(true next state) = 0

        # If action is pickup:
            # If robot position is adjacent to belt box and robot is not holding box:
                # P(true next state) = 1
            # Else 
                # P(true next state) = 0
        # Note that true next state will need to include an adjustment to belt

        # If action is GOTO_REST:
            # P(true next state) = 1
             # P(not true next state) = 0
        # ELse
            

        # If action is GOTO_DROPOFF:
            # If human not in dropoff:
             # P(true state at dropoff) = 1
            #Else:
            # P(true state at dropoff) = 0
            # P( robot goes to waiting  = 1)

        
        # UPDATE HUMAN CHARACTERISTICS
        # DETERMINISTIC
        # P(s' human tiredness = somefunction(s human tiredness)) = 1
        # P(s' lr bias) = somepiecewisefunction(s lr bias) = 1
        # Note: potential idea to change it to depend on action taken from s
        
        # HUMAN TRANSITIONS

        # Humans

        # MOVEMENT ACTIONS
        # If the human action is of type GOTO_[beltcell]

        # some_function(tiredness in s) chance of landing up in GO_TO[beltcell]

        # If we end up taking action:
            # lr bias number between 0 and n
            # x% chance of going to specified state (x% calculated as a function of current position and lr bias number)
            # 1 - x% chance of going to some other state determined as a function of (lr bias, current position)
        
        # For dropoff, pickup, putdown:
        # All tiredness related
        # No consideration of whether robot is in dropoff or not
        # Need to consider human position

        # BELT DYNAMICS

        # DETERMINISTIC NATURAL BELT MOVEMENT
        # P(BELT ARRAY AT s' = UPDATE(BELT_ARRAY at s)) = 1

        # Somewhere here update missed if belt array index number goes past n

    
        # 1 - some_function(tiredness in s) chance of landing up in GO_TO_REST

        # If 

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




# Observation FN

P(o | s', a)
  
  # Only want PO in terms of robot's perception of human characteristics

  # Take true s' and apply noise to human chaaracteristics component

  # Return obs








