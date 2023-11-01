

# Fixed positions
#C1...CN
#REST
# DROPOFF

class Position():
    PICKUP_1 = 0
    PICKUP_2 = 1
    PICKUP_3 = 2
    PICKUP_4 = 3
    PICKUP_5 = 4
    DROP_OFF = 5
    REST_POSITION = 6


belt_positions = {Position.PICKUP_1, Position.PICKUP_2, Position.PICKUP_3, Position.PICKUP_4, Position.PICKUP_5}

def belt_to_index(position):
    if position == Position.PICKUP_1:
        return 1
    elif position == Position.PICKUP_2:
        return 2
    elif position == Position.PICKUP_3:
        return 3
    elif position == Position.PICKUP_4:
        return 4
    else:
        return 5

total_belt_positions = 5



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
    REST = 6
    PICKUP = 7
    PUTDOWN = 8

go_to_belt_actions = {Position.GOTO_P1, Position.GOTO_P2, Position.GOTO_P3, Position.GOTO_P4, Position.GOTO_P5}

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


        boxes_removed = set()

        successful_drop_off = False

        # Update tiredness (super basic for now):
        tiredness_prob = 0
        if current_state.human.tiredness == resultant_state.human.tiredness:
            tiredness_prob += 0.8
        if min(10, current_state.human.tiredness + 1) == resultant_state.human.tiredness:
            tiredness_prob += 0.1
        if min(10, current_state.human.tiredness + 2) == resultant_state.human.tiredness:
            tiredness_prob += 0.1


        # Update lr bias:
        # TODO

        # Calculating probability of human action resulting in rsultant_state:
        human_action_prob = 0

        # If resting, human goes to rest position.
        if human_action == AtomicAction.REST:
            if resultant_state.human.position == Position.REST_POSITION:
                human_action_prob = 1
            else:
                return 0


        elif human_action == AtomicAction.GOTO_DROPOFF:
            if resultant_state.human.position == Position.REST_POSITION:
                human_action_prob = current_state.human.tiredness / 10
            elif resultant_state.human.position == Position.DROP_OFF:
                human_action_prob = (1 - current_state.human.tiredness) / 10
            else:
                return 0

        elif human_action == AtomicAction.PUTDOWN:
            # Ensure state is applicable for put down action:
            if not (current_state.human.holding_box and current_state.human.position == Position.DROP_OFF):
                # If we can't put down, rest instead.
                if resultant_state.human.position == Position.REST_POSITION:
                    human_action_prob = 1
                else:
                    return 0
            # If the human is holding a box and is at drop-off location:
            else:
                if resultant_state.human.position == Position.REST_POSITION:
                    human_action_prob = current_state.human.tiredness / 10
                # Successfully placed box:
                elif resultant_state.human.position == Position.DROP_OFF and not resultant_state.human.holding_box:
                    human_action_prob = (1 - current_state.human.tiredness) / 10
                    successful_drop_off = True
                else:
                    return 0

        elif human_action == AtomicAction.PICKUP:
            # Ensure state is applicable for pick-up action
            if not (current_state.human.position in belt_positions and not current_state.human.holding_box):
                # Can't pick up, so rest
                if resultant_state.human.position == Position.REST_POSITION:
                    human_action_prob = 1
                else:
                    return 0
            # If we are at the belt and not holding a box
            else:
                index = belt_to_index(current_state.human.position)
                # No box at belt location:
                if current_state.belt[index] == 0:
                    if resultant_state.human.position == Position.REST_POSITION:
                        human_action_prob = 1
                    else:
                        return 0
                # Box at our location:
                else:
                    if resultant_state.human.position == Position.REST_POSITION:
                        human_action_prob = current_state.human.tiredness / 10
                    elif resultant_state.human.position == current_state.human.position\
                        and resultant_state.human.holding_box == True\
                        and resultant_state.belt[index] == 0:
                        human_action_prob = (1 - current_state.human.tiredness) / 10
                        boxes_removed.add(index)
                    else:
                        return 0


        elif human_action in go_to_belt_actions:
            if resultant_state.human.position == Position.REST_POSITION:
                human_action_prob = current_state.human.tiredness / 10
            else:
                human_action_prob = (1 - current_state.human.tiredness) / 10

                if human_action == AtomicAction.GOTO_P1:
                    intended = 1
                elif human_action == AtomicAction.GOTO_P2:
                    intended = 2
                elif human_action == AtomicAction.GOTO_P3:
                    intended = 3
                elif human_action == AtomicAction.GOTO_P4:
                    intended = 4
                else:
                    intended = 5

                bias_difference = current_state.human.lr_bias - intended

                if resultant_state.human.position == Position.PICKUP_1:
                    gap = 1 - intended
                elif resultant_state.human.position == Position.PICKUP_2:
                    gap = 2 - intended
                elif resultant_state.human.position == Position.PICKUP_3:
                    gap = 3 - intended
                elif resultant_state.human.position == Position.PICKUP_4:
                    gap = 4 - intended
                elif resultant_state.human.position == Position.PICKUP_5:
                    gap = 5 - intended
                else:
                    return 0  # Did not end up in pickup or rest locations

                # If we have a rightward bias, and we ended up right of intended but not right of bias:
                if 0 < gap <= bias_difference:
                    human_action_prob *= 0.5 * (1 / abs(bias_difference) + 1)
                # If we have a leftward bias, and we ended up left of intended but not left of bias:
                elif 0 > gap >= bias_difference:
                    human_action_prob *= 0.5 * (1 / abs(bias_difference) + 1)
                # If we ended up at the intended cell:
                elif gap == 0:
                    human_action_prob *= 0.5 + 0.5 * (1 / abs(bias_difference) + 1)
                # lr-bias conflicts with gap
                else:
                    return 0

        # Calculating probability of robot action resulting in the resultant state
        robot_action_prob = 0
        # TODO: Add non-determinism
        if robot_action == AtomicAction.REST:
            if resultant_state.robot.position == Position.REST_POSITION:
                robot_action_prob = 1
            else:
                return 0

        elif robot_action == AtomicAction.GOTO_DROPOFF:
            if resultant_state.human.position == Position.DROP_OFF:
                # Robot goes to rest if it would clash with human position
                if resultant_state.robot.position == Position.REST_POSITION:
                    robot_action_prob = 1
                else:
                    return 0
            else:
                if resultant_state.robot.position == Position.DROP_OFF:
                    robot_action_prob = 1
                else:
                    return 0

        elif robot_action == AtomicAction.GOTO_P1:
            if resultant_state.human.position == Position.PICKUP_1:
                # Robot goes to rest if it would clash with human position
                if resultant_state.robot.position == Position.REST_POSITION:
                    robot_action_prob = 1
                else:
                    return 0
            else:
                if resultant_state.robot.position == Position.PICKUP_1:
                    robot_action_prob = 1
                else:
                    return 0
        elif robot_action == AtomicAction.GOTO_P2:
            if resultant_state.human.position == Position.PICKUP_2:
                # Robot goes to rest if it would clash with human position
                if resultant_state.robot.position == Position.REST_POSITION:
                    robot_action_prob = 1
                else:
                    return 0
            else:
                if resultant_state.robot.position == Position.PICKUP_2:
                    robot_action_prob = 1
                else:
                    return 0
        elif robot_action == AtomicAction.GOTO_P3:
            if resultant_state.human.position == Position.PICKUP_3:
                # Robot goes to rest if it would clash with human position
                if resultant_state.robot.position == Position.REST_POSITION:
                    robot_action_prob = 1
                else:
                    return 0
            else:
                if resultant_state.robot.position == Position.PICKUP_3:
                    robot_action_prob = 1
                else:
                    return 0
        elif robot_action == AtomicAction.GOTO_P4:
            if resultant_state.human.position == Position.PICKUP_4:
                # Robot goes to rest if it would clash with human position
                if resultant_state.robot.position == Position.REST_POSITION:
                    robot_action_prob = 1
                else:
                    return 0
            else:
                if resultant_state.robot.position == Position.PICKUP_4:
                    robot_action_prob = 1
                else:
                    return 0
        elif robot_action == AtomicAction.GOTO_P5:
            if resultant_state.human.position == Position.PICKUP_5:
                # Robot goes to rest if it would clash with human position
                if resultant_state.robot.position == Position.REST_POSITION:
                    robot_action_prob = 1
                else:
                    return 0
            else:
                if resultant_state.robot.position == Position.PICKUP_5:
                    robot_action_prob = 1
                else:
                    return 0

        elif robot_action == AtomicAction.PUTDOWN:
            # Ensure state is applicable for put down action:
            if not (current_state.robot.holding_box and current_state.robot.position == Position.DROP_OFF) \
                    or resultant_state.human.position == Position.DROP_OFF:
                # If we can't put down, rest instead.
                if resultant_state.human.position == Position.REST_POSITION:
                    robot_action_prob = 1
                else:
                    return 0
            # If the human is holding a box and is at drop-off location:
            else:
                # Successfully placed box:
                if resultant_state.robot.position == Position.DROP_OFF and not resultant_state.robot.holding_box:
                    robot_action_prob = 1
                    successful_drop_off = True
                else:
                    return 0

        elif robot_action == AtomicAction.PICKUP:
            # Ensure state is applicable for pick-up action
            if not (current_state.robot.position in belt_positions and not current_state.robot.holding_box):
                # Can't pick up, so rest
                if resultant_state.robot.position == Position.REST_POSITION:
                    robot_action_prob = 1
                else:
                    return 0
            # If robot is at the belt and not holding a box
            else:
                index = belt_to_index(current_state.robot.position)
                # No box at belt location:
                if current_state.belt[index] == 0:
                    if resultant_state.robot.position == Position.REST_POSITION:
                        robot_action_prob = 1
                    else:
                        return 0
                # Box at robot's location:
                else:
                    if resultant_state.robot.position == current_state.robot.position \
                            and resultant_state.robot.holding_box == True \
                            and resultant_state.belt[index] == 0:
                        robot_action_prob = 1
                        boxes_removed.add(index)
                    else:
                        return 0


        # Update boxes:
        expected_belt = current_state.belt.copy()
        # Remove the boxes:
        for index in boxes_removed:
            expected_belt[index] = 0
        # Shift belt to the left:
        for i in range(len(expected_belt) - 1):
            expected_belt[i] = expected_belt[i+1]
        expected_belt[-1] = 0

        if not expected_belt == resultant_state.belt:
            return 0

        # Finally, return total probability of achieving resultant_state:
        return tiredness_prob * human_action_prob * robot_action_prob




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








