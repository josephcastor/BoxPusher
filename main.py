import random
import numpy as np
import pomdp_py
import copy
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


belt_positions = [Position.PICKUP_1, Position.PICKUP_2, Position.PICKUP_3, Position.PICKUP_4, Position.PICKUP_5]

def belt_to_index(position):
    for i in range(len(belt_positions)):
        if position == belt_positions[i]:
            return i
    return -1

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


class State(pomdp_py.State):
    def __init__(self, human, robot, belt, packed, missed, name):
        self.human = human
        self.robot = robot
        self.belt = belt
        self.packed = packed # count of packed packages
        self.missed = missed # count of missed missed
        self.name = name

    def __hash__(self):
        return hash((self.human,
                     self.robot,
                     tuple(self.belt.tolist()),
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
        return f"State(human={self.human}, robot={self.robot}, belt={self.belt}, packed={self.packed}, missed={self.packed})"

# In transition fn, update tiredness if we rested previously

class Action(pomdp_py.Action):
    def __init__(self, human_action, robot_action):
        self.human_action = human_action
        self.robot_action = robot_action

    def __hash__(self):
        return hash((self.human_action,
                     self.robot_action))

    def __eq__(self, other):
        if isinstance(other, Action):
            return self.human_action == other.human_action\
                and self.robot_action == other.robot_action
        return False

    def __str__(self):
        return f"Action(human_action={self.human_action}, robot_action={self.robot_action})"

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

# TODO:
# Make observation match the structure of State
class Observation(pomdp_py.Observation):

    def __init__(self, human, robot, belt, packed, missed, name):
        self.human = human
        self.robot = robot
        self.belt = belt
        self.packed = packed # count of packed packages
        self.missed = missed # count of missed missed
        self.name = name

    def __hash__(self):
        return hash((self.human,
                     self.robot,
                     tuple(self.belt.tolist()),
                     self.packed,
                     self.missed))

    def __eq__(self, other):
        if isinstance(other, Observation):
            return self.human == other.human\
                and self.robot == other.robot\
                and self.belt == other.belt\
                and self.packed == other.packed\
                and self.missed == other.missed\
                and self.name == other.name
        return False

class TransitionModel(pomdp_py.TransitionModel):


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
                    intended = 0
                elif human_action == AtomicAction.GOTO_P2:
                    intended = 1
                elif human_action == AtomicAction.GOTO_P3:
                    intended = 2
                elif human_action == AtomicAction.GOTO_P4:
                    intended = 3
                else:
                    intended = 4

                bias_difference = current_state.human.lr_bias - intended

                if resultant_state.human.position == Position.PICKUP_1:
                    gap = 0 - intended
                elif resultant_state.human.position == Position.PICKUP_2:
                    gap = 1 - intended
                elif resultant_state.human.position == Position.PICKUP_3:
                    gap = 2 - intended
                elif resultant_state.human.position == Position.PICKUP_4:
                    gap = 3 - intended
                elif resultant_state.human.position == Position.PICKUP_5:
                    gap = 4 - intended
                else:
                    return 0  # Did not end up in pickup or rest locations

                # If we have a rightward bias, and we ended up right of intended but not right of bias:
                if 0 < gap <= bias_difference:
                    human_action_prob *= 0.5 * (1 / (abs(bias_difference) + 1))
                # If we have a leftward bias, and we ended up left of intended but not left of bias:
                elif 0 > gap >= bias_difference:
                    human_action_prob *= 0.5 * (1 / (abs(bias_difference) + 1))
                # If we ended up at the intended cell:
                elif gap == 0:
                    human_action_prob *= 0.5 + 0.5 * (1 / (abs(bias_difference) + 1))
                # lr-bias conflicts with gap
                else:
                    return 0

        # ROBOT:
        # Calculating probability of robot action resulting in the resultant state
        robot_action_prob = 0
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

        # Non-determinism in robot movement actions
        if robot_action in go_to_belt_actions:
            index = go_to_belt_actions.index(robot_action)
            if resultant_state.robot.position == belt_positions[index] and \
                resultant_state.human.position != resultant_state.robot.position:
                robot_action_prob = 0.8
                if index == 0 or index == len(belt_positions) - 1:
                    robot_action_prob = 0.9

            elif index > 0:
                # Chance of going one location to the left of intended:
                if resultant_state.robot.position == belt_positions[index - 1] and \
                resultant_state.human.position != resultant_state.robot.position:
                    robot_action_prob = 0.1
            elif index < len(belt_positions) - 1:
                # Chance of going one location to the right of intended
                if resultant_state.robot.position == belt_positions[index + 1] and \
                    resultant_state.human.position != resultant_state.robot.position:
                    robot_action_prob = 0.1

            # Probability of going to rest state from collisions with human:
            if resultant_state.robot.position == Position.REST_POSITION:
                robot_action_prob = 0
                if resultant_state.human.position == belt_positions[index]:
                    robot_action_prob += 0.8
                    if index == 0 or index == len(belt_positions) - 1:
                        robot_action_prob += 0.1
                elif index > 0 and resultant_state.human.position == belt_positions[index - 1]:
                    robot_action_prob += 0.1
                elif index < len(belt_positions) - 1 and resultant_state.human.position == belt_positions[index + 1]:
                    robot_action_prob += 0.1



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
                if current_state.belt[index] == 0 or resultant_state.human.position:
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

        # Ensuring the missed count has been updated correctly:
        if expected_belt[0] == 1:
            # New missed package:
            if not resultant_state.missed == current_state.missed + 1:
                return 0
        # No new missed packages:
        elif not resultant_state.missed == current_state.missed:
                return 0

        # Shift belt to the left:
        for i in range(len(expected_belt) - 1):
            expected_belt[i] = expected_belt[i+1]
        expected_belt[-1] = 0

        # Update total collected and missed:
        if successful_drop_off:
            if not resultant_state.packed == current_state.packed + 1:
                return 0
        else:
            if not resultant_state.packed == current_state.packed:
                return 0


        if not expected_belt == resultant_state.belt:
            return 0

        # Finally, return total probability of achieving resultant_state:
        return tiredness_prob * human_action_prob * robot_action_prob


    def sample(self, current_state, action):
        human_action = action.human_action
        human_rest_prob = current_state.human.tiredness / 10

        resultant_state = State(current_state.human, current_state.robot, current_state.belt, current_state.packed, current_state.missed, current_state.name)
        
        if human_action == AtomicAction.REST:
            resultant_state.human.position = Position.REST_POSITION
        elif human_action == AtomicAction.GOTO_DROPOFF:
            rand = np.random.rand()
            if rand <= human_rest_prob:
                resultant_state.human.position = Position.REST_POSITION
            else:
                resultant_state.human.position = Position.DROP_OFF

        elif human_action == AtomicAction.PUTDOWN:
            if not (current_state.human.holding_box and current_state.human.position == Position.DROP_OFF):
                resultant_state.human.position = Position.REST_POSITION
            else:
                rand = np.random.rand()
                if rand <= human_rest_prob:
                    resultant_state.human.position = Position.REST_POSITION
                else:
                    resultant_state.human.position = Position.DROP_OFF
                    resultant_state.human.holding_box = False
                    resultant_state.packed += 1

        elif human_action == AtomicAction.PICKUP:
            if not (current_state.human.position in belt_positions and not current_state.human.holding_box):
                resultant_state.human.position = Position.REST_POSITION
            else:
                index = belt_to_index(current_state.human.position)
                if current_state.belt[index] == 0:
                    resultant_state.human.position = Position.REST_POSITION
                else:
                    resultant_state.human.holding_box = True
                    resultant_state.belt[index] = 0

        elif human_action in go_to_belt_actions:
            rand = np.random.rand()
            if rand <= human_rest_prob:
                resultant_state.human.position = Position.REST_POSITION
            else:
                if human_action == AtomicAction.GOTO_P1:
                    intended = 0
                elif human_action == AtomicAction.GOTO_P2:
                    intended = 1
                elif human_action == AtomicAction.GOTO_P3:
                    intended = 2
                elif human_action == AtomicAction.GOTO_P4:
                    intended = 3
                else:
                    intended = 4

                bias_difference = current_state.human.lr_bias - intended

                if bias_difference == 0:
                    resultant_state.human.position = belt_positions[intended]
                elif bias_difference > 0:
                    probs = []
                    prob = 0.0
                    for i in range(intended, current_state.human.lr_bias + 1):
                        if i == intended:
                            prob += 0.5
                        prob += 0.5*(1 / (abs(bias_difference) + 1))
                        probs.append((i, prob))
                    rand = np.random.rand()

                    
                    # for i in range(len(probs) - 1):

                    #     if rand <= probs[i][1]:
                    #         resultant_state.human.position = belt_positions[probs[i][0]]


                elif bias_difference < 0:
                    probs = []
                    prob = 0
                    for i in range(current_state.human.lr_bias, intended + 1):
                        if i == intended:
                            prob += 0.5
                        prob += 0.5 * (1 / (abs(bias_difference) + 1))
                        probs.append((i, prob))
                    rand = np.random.rand()

                    for i in range(len(probs)):

                        if rand <= probs[i][1]:
                            resultant_state.human.position = belt_positions[probs[i][0]]

                    # resultant_state.human.position = belt_positions[np.random.choice(list(probs.keys()),size=1, p=list(probs.values()))]
                    # resultant_state.human.position = belt_positions[random.choice()]

        # ROBOT TIME:
        robot_action = action.robot_action
        if robot_action == AtomicAction.REST:
            resultant_state.robot.position = Position.REST_POSITION
        elif robot_action == AtomicAction.GOTO_DROPOFF:
            if resultant_state.human.position == Position.DROP_OFF:
                resultant_state.robot.position = Position.REST_POSITION
            else:
                resultant_state.robot.position = Position.DROP_OFF

        elif robot_action == AtomicAction.PICKUP:
            if not (current_state.robot.position in belt_positions and not current_state.robot.holding_box):
                resultant_state.robot.position = Position.REST_POSITION
            else:
                index = belt_to_index(current_state.robot.position)
                # No box at belt location:
                if current_state.belt[index] == 0 or resultant_state.human.position == belt_positions[index]:
                    resultant_state.robot.position = Position.REST_POSITION
                else:
                    resultant_state.robot.holding_box = True
                    resultant_state.belt[index] = 0

        elif robot_action == AtomicAction.PUTDOWN:
            if not (current_state.robot.holding_box and current_state.robot.position == Position.DROP_OFF) \
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
                    resultant_state.robot.position = belt_positions[index]
                else:
                    resultant_state.robot.position = belt_positions[index + 1]
            elif index == len(belt_positions) - 1:
                if rand <= 0.9:
                    resultant_state.robot.position = belt_positions[index]
                else:
                    resultant_state.robot.position = belt_positions[index - 1]
            else:
                if rand <= 0.8:
                    resultant_state.robot.position = belt_positions[index]
                elif rand <= 0.9:
                    resultant_state.robot.position = belt_positions[index - 1]
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
        resultant_state.belt[-1] = 0

        # print(resultant_state.belt)

        return resultant_state

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

class ObservationModel(pomdp_py.ObservationModel):

    def sample(self, next_state, action):

        next_tiredness = next_state.human.tiredness
        next_lr_bias = next_state.human.lr_bias

        next_tiredness += np.random.normal(0, 0.1)
        next_tiredness = int(round(next_tiredness))
        next_tiredness = max(min(10,next_tiredness),0)


        next_lr_bias += np.random.normal(0, 0.1)
        next_lr_bias = int(round(next_lr_bias))
        next_lr_bias = max(min(5,next_lr_bias),0)

        human = copy.copy(next_state.human)
        human.tiredness = next_tiredness
        human.lr_bias = next_lr_bias

        print("True human tiredness: ", next_state.human.tiredness)
        print("Observed human tiredness: ", human.tiredness)

        print("True human lr_bias: ", next_state.human.lr_bias)
        print("Observed human lr_bias: ", human.lr_bias)


        # human
        return Observation(human, next_state.robot, next_state.belt, next_state.packed, next_state.missed, next_state.name)

    # def observation_probability(self, observation, next_state,action):

    #     next_tiredness = next_state.human.tiredness
    #     next_lr_bias = next_state.human.lr_bias


    #     tiredness_candidates = [observation.tiredness -2,observation.tiredness -1, observation.tiredness, observation.tiredness +1, observation.tiredness + 2  ]
    #     lr_bias_candidates = [observation.lr_bias - 1 , observation.lr_bias  ,observation.lr_bias + 1 ]

    #     tiredness_candidates = [v for v in tiredness_candidates if v>=1 and v<=10]
    #     lr_bias_candidates = [v for v in lr_bias_candidates if v>=1 and v<=5]

    #     tiredness_p = [ norm.pdf(next_tiredness, candidate , 0.5) for candidate in tiredness_candidate]
    #     lr_bias_p = [ norm.pdf(next_tiredness, candidate , 0.5) for candidate in lr_bias_candidates]

class PolicyModel(pomdp_py.RolloutPolicy):

    ACTIONS = [Action(a1,a2) for a1 in range(0,9) for a2 in range(0,9)]

    def rollout(self, state, *args):
        return random.sample(self.get_all_actions(), 1)[0]

    def get_all_actions(self, state=None, history=None):
        return PolicyModel.ACTIONS

class RewardModel(pomdp_py.RewardModel):

    def reward(self, state,action):
        # random for no
        return 1

    def sample(self, state, action, next_state):
        return self.reward(state, action)


# Generate initial state
#
# Generate initial belief
#
#
# 5 by 5 space


class Problem(pomdp_py.POMDP):

    def __init__(self, init_true_state, init_belief):
        agent = pomdp_py.Agent(init_belief,
                                PolicyModel(),
                                TransitionModel(),
                                ObservationModel(),
                                RewardModel())
        env = pomdp_py.Environment(init_true_state,
                                    TransitionModel(),
                                    RewardModel())
        super().__init__(agent, env, name="Problem")


human1 = Human(Position.PICKUP_1,False, 1, 3)
human2 = Human(Position.PICKUP_1,False, 5, 7)
robot = Robot(Position.REST_POSITION, False)
init_true_state = State(human1, robot, np.array([0,1,0,0,0]), 0,0, "whatevs")
# init_belief = pomdp_py.Histogram({State(human, robot, np.array([0,1,0,0,0]), 0,0): 0.5,
#                                   State(human, robot, np.array([0,1,0,0,0]), 0,0): 0.5})

init_belief = pomdp_py.Particles([State(human1, robot, np.array([0,1,0,0,0]), 0,0, "whatevs1"), State(human2, robot, np.array([0,1,0,0,0]), 0,0, "whatevs2")])

prob = Problem(init_true_state, init_belief)

# Step 1; in main()
# creating planners
pomcp = pomdp_py.POMCP(max_depth=3, discount_factor=0.95,
                       planning_time=.5, exploration_const=110,
                       rollout_policy=prob.agent.policy_model)

# Steps 2-6; called in main()
def test_planner(tiger_problem, planner, nsteps=10):
    """Runs the action-feedback loop of Tiger problem POMDP"""
    for i in range(nsteps):  # Step 6
        # Step 2
        action = planner.plan(tiger_problem.agent)

        print("==== Step %d ====" % (i+1))
        print("True state:", tiger_problem.env.state)
        # print("Belief:", tiger_problem.agent.cur_belief)
        print("Action:", action)
        # Step 3; There is no state transition for the tiger domain.
        # In general, the ennvironment state can be transitioned
        # using
        #
        #   reward = tiger_problem.env.state_transition(action, execute=True)
        #
        # Or, it is possible that you don't have control
        # over the environment change (e.g. robot acting
        # in real world); In that case, you could skip
        # the state transition and re-estimate the state
        # (e.g. through the perception stack on the robot).
        reward = tiger_problem.env.reward_model.sample(tiger_problem.env.state, action, None)
        print("Reward:", reward)

        # Step 4
        # Let's create some simulated real observation;
        # Here, we use observation based on true state for sanity
        # checking solver behavior. In general, this observation
        # should be sampled from agent's observation model, as
        #
        real_observation = tiger_problem.agent.observation_model.sample(tiger_problem.env.state, action)
        #
        # or coming from an external source (e.g. robot sensor
        # reading). Note that tiger_problem.env.state should store
        # the environment state after transition.
        # real_observation = Observation(tiger_problem.env.state.human.tiredness, tiger_problem.env.state.human.lr_bias)
        print(">> Observation: %s" % real_observation)

        # Step 5
        # Update the belief. If the planner is POMCP, planner.update
        # also automatically updates agent belief.
        tiger_problem.agent.update_history(action, real_observation)
        planner.update(tiger_problem.agent, action, real_observation)
        if isinstance(planner, pomdp_py.POUCT):
            print("Num sims: %d" % planner.last_num_sims)
        if isinstance(tiger_problem.agent.cur_belief, pomdp_py.Histogram):
            new_belief = pomdp_py.update_histogram_belief(tiger_problem.agent.cur_belief,
                                                          action, real_observation,
                                                          tiger_problem.agent.observation_model,
                                                          tiger_problem.agent.transition_model)
            tiger_problem.agent.set_belief(new_belief)

test_planner(prob, pomcp, 10)



        



# P(o | s', a)
  
  # Only want PO in terms of robot's perception of human characteristics

  # Take true s' and apply noise to human chaaracteristics component

  # Return obs








