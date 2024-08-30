import abc
import pdb
from enum import IntFlag, unique
from typing import TYPE_CHECKING, List, Tuple

import time
import numpy as np
from . import art

if TYPE_CHECKING:
    from .engine import GridEngine


@unique
class ItemKind(IntFlag):

    WALL = 0
    OBJECT = 2
    GOAL = 4
    AGENT = 8   
    OBJECT_ATTACHED = 14
    AGENT_ATTACHED = 18
    H_AGENT = 9
    H_AGENT_ATTACHED = 19


    def __contains__(self, item):
        return (self.value & item.value) == item.value


class ItemKind_encode():
    """
    one hot encoding of ItemKind, used for vector observations
    """

    encoding = {}
    WALL = 1
    OBJECT = 2
    GOAL = 4
    AGENT = 8   
    OBJECT_ATTACHED = 14
    AGENT_ATTACHED = 18
    H_AGENT = 9
    H_AGENT_ATTACHED = 19

    encoding[ItemKind.WALL] = WALL
    encoding[ItemKind.OBJECT] = OBJECT
    encoding[ItemKind.GOAL] = GOAL
    encoding[ItemKind.AGENT] = AGENT
    encoding[ItemKind.H_AGENT] = H_AGENT
    encoding[ItemKind.OBJECT_ATTACHED] = OBJECT_ATTACHED
    encoding[ItemKind.AGENT_ATTACHED] = AGENT_ATTACHED
    encoding[ItemKind.H_AGENT_ATTACHED] = H_AGENT_ATTACHED

class ItemKind_onehot():
    """
    one hot encoding of ItemKind, used for vector observations
    """
    encoding = {}
    encode_one_hot = {}
    num_itemkind = 8 # the number of distinct item kinds

    wall = np.zeros(shape=(num_itemkind), dtype='int32')
    object_ = np.zeros(shape=(num_itemkind), dtype='int32')
    goal = np.zeros(shape=(num_itemkind), dtype='int32')
    agent = np.zeros(shape=(num_itemkind), dtype='int32')
    object_attached = np.zeros(shape=(num_itemkind), dtype='int32')
    agent_attached = np.zeros(shape=(num_itemkind), dtype='int32')
    h_agent = np.zeros(shape=(num_itemkind), dtype='int32')
    h_agent_attached = np.zeros(shape=(num_itemkind), dtype='int32')

    wall[0] = 1
    object_[1] = 1
    goal[2] = 1
    agent[3] = 1
    h_agent[4] = 1
    object_attached[5] = 1
    agent_attached[6] = 1
    h_agent_attached[7] = 1

    encoding[ItemKind.WALL] = wall
    encoding[ItemKind.OBJECT] = object_
    encoding[ItemKind.GOAL] = goal
    encoding[ItemKind.AGENT] = agent
    encoding[ItemKind.H_AGENT] = h_agent
    encoding[ItemKind.OBJECT_ATTACHED] = object_attached
    encoding[ItemKind.AGENT_ATTACHED] = agent_attached
    encoding[ItemKind.H_AGENT_ATTACHED] = h_agent_attached

    e_WALL = 1
    e_OBJECT = 2
    e_GOAL = 4
    e_AGENT = 8   
    e_OBJECT_ATTACHED = 14
    e_AGENT_ATTACHED = 18
    e_H_AGENT = 9
    e_H_AGENT_ATTACHED = 19

    encode_one_hot[0] = np.zeros(shape=(num_itemkind), dtype='int32') #open grid cell
    encode_one_hot[e_WALL] = wall
    encode_one_hot[e_OBJECT] = object_
    encode_one_hot[e_GOAL] = goal
    encode_one_hot[e_AGENT] = agent
    encode_one_hot[e_H_AGENT] = h_agent
    encode_one_hot[e_OBJECT_ATTACHED] = object_attached
    encode_one_hot[e_AGENT_ATTACHED] = agent_attached
    encode_one_hot[e_H_AGENT_ATTACHED] = h_agent_attached




class ItemBase(abc.ABC):
    # param loc: (x,y) location in the grid; the origin of the grid (1,1) is the upper left corner;
    def __init__(self, loc: np.ndarray, kind: ItemKind, impassable: bool):
        self.loc = np.asarray(loc, dtype=int)
        self.kind = kind
        self.impassable = impassable
        self.engine: "GridEngine" = None

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return hash(id(self))

    @property
    def passable(self):
        return not self.impassable


class WallItem(ItemBase):
    def __init__(self, loc: np.ndarray):
        super().__init__(loc, ItemKind.WALL, impassable=True)

    @staticmethod
    def create_hwall(
        shape: Tuple[int, int], y: int, xstart: int = 0, xend: int = -1
    ) -> List["WallItem"]:
        if xend == -1:
            xend = shape[1]
        w = [WallItem(loc=(x, y)) for x in range(xstart, xend)]
        return w

    @staticmethod
    def create_vwall(
        shape: Tuple[int, int], x: int, ystart: int = 0, yend: int = -1
    ) -> List["WallItem"]:
        if yend == -1:
            yend = shape[0]
        w = [WallItem(loc=(x, y)) for y in range(ystart, yend)]
        return w

    @staticmethod
    def create_border(shape: Tuple[int, int]) -> List["WallItem"]:
        w0 = WallItem.create_hwall(shape, y=0, xstart=0, xend=-1)
        w1 = WallItem.create_hwall(shape, y=shape[0] - 1, xstart=1, xend=shape[0]-1)
        w2 = WallItem.create_vwall(shape, x=0, ystart=1, yend=-1)
        w3 = WallItem.create_vwall(shape, x=shape[1] - 1, ystart=1, yend=-1)
        return w0 + w1 + w2 + w3


class ObjectItem(ItemBase):
    """Base class for items to be manipulated by agents."""

    def __init__(self, loc: np.ndarray, mass: int = 1, kind: ItemKind = ItemKind.OBJECT, num_carriers: int = 2):
        #super().__init__(loc, ItemKind.OBJECT, impassable=True)
        super().__init__(loc, kind, impassable=True)
        self.carriers: List["AgentItem"] = [] # carriers attached to this object
        self.mass = mass
        self.num_carriers = num_carriers
        

    @property
    def attachable(self) -> bool:
       
        
        return (self.kind != ItemKind.OBJECT_ATTACHED)

    @property
    def dropped(self) -> bool:
        return len(self.carriers) == 0


class GoalItem(ItemBase):
    def __init__(self, loc: np.ndarray):
        super().__init__(loc, ItemKind.GOAL, impassable=False)


class MoveAgent(ItemBase):
    """Agent supporting cardinal moves."""

    def __init__(self, loc: np.ndarray, kind, impassable: bool = True):
        super().__init__(loc, kind, impassable=impassable)
        self.move_offsets = {
            "up": np.array((0, -1)),
            "down": np.array((0, 1)),
            "left": np.array((-1, 0)),
            "right": np.array((1, 0)),
        }
        self.loc.setflags(write=1) # allows to change the initial loc of an agent

    def move(self, direction: str) -> bool:
        """Move the agent and return success of move."""
        off = self.move_offsets.get(direction, None)
        if off is None:
            raise ValueError(f"Invalid move direction {direction}")
        return self.engine.move([self], self.engine.offset_locs([self], off), test=True)


class PickAgent(MoveAgent):
    """Agent supporting pick and move actions"""

    def __init__(self, loc: np.ndarray, kind: str = "robot_agent", impassable: bool = True, target_object = None):
        super().__init__(loc, kind, impassable=impassable)
        
        if "heuristic_agent" in kind:
            self.kind = ItemKind.H_AGENT
            
        else:
            self.kind = ItemKind.AGENT
        
        self.target_object =  target_object

        self.picked_object: ObjectItem = None
        




    @property
    def attached(self):
        """True if the agent is currently attached to an object"""
        return self.picked_object is not None

    def pick(self) -> bool:
        if self.attached:
            return False

        objs = self._reachable_objects()
        if len(objs) == 0:
            # No attachable objects
            return False


        

         #object is attached when it is attached by a heuristic and a policy agent
        #if an agent is already attached to the object and the agent is policy agent then only a heuristic agent can attach and vice versa
        #loop through possible objects and attached to the feasible one # this condition only in heuristic run

        #TODO when not using driver agent no need of two different agents to attach; any two agents work

        for obj in objs:

            if self.kind == ItemKind.AGENT:

                if len(obj.carriers) > 0 and obj.carriers[0].kind != ItemKind.H_AGENT_ATTACHED:
                    pass
                else:
                    self.picked_object = obj
                    self.kind = ItemKind.AGENT_ATTACHED
                    self.picked_object.carriers.append(self)
                    if len(self.picked_object.carriers) > 1:
                        self.picked_object.kind = ItemKind.OBJECT_ATTACHED
                    return True


            elif self.kind == ItemKind.H_AGENT:

                if len(obj.carriers) > 0 and obj.carriers[0].kind != ItemKind.AGENT_ATTACHED:
                    pass
                else:
                    self.picked_object = obj
                    self.kind = ItemKind.H_AGENT_ATTACHED
                    self.picked_object.carriers.append(self)
                    if len(self.picked_object.carriers) > 1:
                        self.picked_object.kind = ItemKind.OBJECT_ATTACHED
                    return True


    
            
            






    
        return False

    def drop(self) -> bool:

        if not self.attached:
            return False

        #as drop is done by heuristic agent no need to consider reachable goal plus need drop even when goal is not near for goal area
        # put attached object on a reachable goal
        # maybe better: introduce 4 drop actions: drop-up/down/right/left
        # goals = self._reachable_goals(self.picked_object)
        # if len(goals) == 0:
        #     # No reachable goals
        #     return False
        

        #it means the object is not attached to req number of agents; in this case just detach the agent when it drops

        #object drop by agent if it is still attachable
        if self.picked_object.attachable:

            
            self.picked_object.carriers.remove(self)
            if self.kind == ItemKind.H_AGENT_ATTACHED:
                    self.kind = ItemKind.H_AGENT
            else:
                    self.kind = ItemKind.AGENT
            self.picked_object = None
            
        else: 

           
            new_loc = self.picked_object.loc

            self.engine.move([self.picked_object], new_loc)
            carriers = self.picked_object.carriers.copy()


            
            for carrier in carriers:
 
          
                carrier.picked_object.carriers.remove(carrier)
                carrier.picked_object.kind = ItemKind.WALL
                carrier.picked_object = None
            
                
                if carrier.kind == ItemKind.H_AGENT_ATTACHED:
                    carrier.kind = ItemKind.H_AGENT
                else:
                    carrier.kind = ItemKind.AGENT
                
            
  

        return True

    def move(self, direction: str) -> bool:
        """Move the agent and return success of move."""

        #agent shouldn't be able to move if it is attached and the object it has picked is not attached by two agents
        if self.attached and self.picked_object.attachable:
            return False
        
        #
        
        off = self.move_offsets.get(direction, None)
        if off is None:
            raise ValueError(f"Invalid direction {direction}")
        group = []

        #attached the carrier and object both if it is attached so the carrier and object both moves along with the agent carrier
        if self.attached and len(self.picked_object.carriers)>1:
                group.append(self.picked_object)
                group.extend(self.picked_object.carriers)
        else:
            group.append(self)
          

        return self.engine.move(group, self.engine.offset_locs(group, off), test=True)

    def _reachable_objects(self) -> List[ObjectItem]:
        objs = []
        locs = np.array(
            [
                self.loc + (-1, 0),
                self.loc + (1, 0),
                self.loc + (0, -1),
                self.loc + (0, 1),
             ]
        )
        for loc in locs:
            items = self.engine.items(loc)
            if len(items) == 1 and (items[0].kind == ItemKind.OBJECT) and items[0].attachable:
                objs.append(items[0])
        return objs

    def _reachable_goals(self, picked_object) -> List[ObjectItem]:
        """@param picked_object: the object to be dropped"""

        # pdb.set_trace()

        goals = []
        #reachable goal locations: goal is in {left,right,up,down} adjacent cells;
        locs = np.array(
            [
                self.loc + (-1, 0),
                self.loc + (1, 0),
                self.loc + (0, -1),
                self.loc + (0, 1),
            ]
        )
        for loc in locs:
            items = self.engine.items(loc)
            if len(items) == 1 and items[0].kind == ItemKind.GOAL:
                goals.append(items[0])
            elif len(items) == 2:
                goal_temp = None
                success_counter = 0
                # if one item is goal, and the other item is picked_object: goal is reachable
                for item in items:
                    if item.kind == ItemKind.GOAL or item == picked_object:
                        success_counter += 1
                    if item.kind == ItemKind.GOAL:
                        goal_temp = item
                if success_counter == 2 and goal_temp is not None:
                    goals.append(goal_temp)
        return goals


