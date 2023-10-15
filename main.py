
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


# From repo
class Action():
    def __init__(self, name):
        self.name = name
    
    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Action):
            return self.name == other.name
        elif type(other) == str:
            return self.name == other

    def __str__(self):
        return self.name

    def __repr__(self):
        return "Action(%s)" % self.name

# From repo
class MoveAction(Action):
    NORTH = (0, -1) # x is horizontal; x+ is right. y is vertical; y- is up.
    EAST = (1, 0)  
    SOUTH = (0, 1)
    WEST = (-1, 0)
    

    def __init__(self, motion, name):
        if motion not in {MoveAction.NORTH, MoveAction.EAST,
                          MoveAction.SOUTH, MoveAction.WEST}:
            raise ValueError("Invalid move motion %s" % motion)
        self.motion = motion
        super().__init__("Move-%s" % str(name))

MoveNorth = MoveAction(MoveAction.NORTH, "NORTH")
MoveEast = MoveAction(MoveAction.EAST, "EAST")
MoveSouth = MoveAction(MoveAction.SOUTH, "SOUTH")
MoveWest = MoveAction(MoveAction.WEST, "WEST")

class BoxAction(Action):

    PICKUP = 0
    PUTDOWN = 1

    def __init__(self, name):
        super().__init__(name)



