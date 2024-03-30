import random

import numpy as np
from typing import List
import itertools
from itertools import product
import time
from sm.envs.concert.gridworld.items import ItemKind, ItemBase
from sm.envs.concert.gridworld import items, games, engine, defaults
from sm.envs.concert.gyms.examples.utils import handle_not_element


class WarehouseGame(games.GameBase):
    def __init__(
        self,
        seed: int = None,
        shape=(9, 9),
        max_steps: int = 100,
        num_objects: int = 1,
        num_goals: int = 1,
    ) -> None:
        super().__init__(seed=seed)
        self.shape = shape
        self.max_steps = max_steps
        self.agent: items.PickAgent = None
        self.num_objects = num_objects
        self.num_goals = num_goals
        self.objects: List[items.ObjectItem] = None
        self.goals: List[items.GoalItem] = None

    def _create_game(self):
        self.engine = engine.GridEngine(self.shape)
        w = items.WallItem.create_border(self.shape)
        self.engine.add(w)
        self.agent = items.PickAgent((-1, -1))
        self.objects = [items.ObjectItem((-1, -1)) for _ in range(self.num_objects)]
        self.goals = [items.GoalItem((-1, -1)) for _ in range(self.num_goals)]
        self.engine.add([self.agent] + self.objects + self.goals)

    def _step(self, step_data: games.StepData):
        action = step_data.actions[self.agent]
        # Move the agent
        if action in ["up", "down", "left", "right"]:
            success = self.agent.move(action)
        elif action == "pick":
            success = self.agent.pick()
        elif action == "drop":
            success = self.agent.drop()

        if not success:
            step_data.add_reward(-0.1)

        # Check game end
        if self._check_goal():
            step_data.add_reward(1.0)
            step_data.terminate_game()
        elif step_data.steps_exceeded(self.max_steps):
            step_data.add_reward(-1.0)
            step_data.terminate_game()

    def _check_goal(self) -> bool:
        goal_locs = np.stack([g.loc for g in self.goals], 0)
        at_goals = self.engine.items(locs=goal_locs)
        dropped_objs = [
            i for i in at_goals if i.kind == items.ItemKind.OBJECT and i.dropped
        ]
        return len(dropped_objs) == len(self.goals)

    def render(self) -> np.ndarray:
        return defaults.render(self.engine)


class WarehouseGame_1(games.GameBase):
    """
    single-agent game; environment just uses agent IDs, agent objects are created and used in the game;
    """
    def __init__(
        self,
        agent_id: str = 'agent_1',
        seed: int = None,
        shape=(8, 8),
        max_steps: int = 100,
        num_objects: int = 1,
        num_goals: int = 1,
        num_obstacles: int = 0,
    ) -> None:
        super().__init__(seed=seed)
        self.shape = shape
        self.max_steps = max_steps
        self.agent_id = agent_id
        self.agent: items.PickAgent = None
        self.agent_dict = {} # mapping of agent ID to agent object
        self.num_objects = num_objects
        self.num_goals = num_goals
        self.objects: List[items.ObjectItem] = None
        self.goals: List[items.GoalItem] = None
        self.num_obstacles=num_obstacles,
        self.moves_since_pick = 1000 # the number of successful moves since the successful pick action

    def _create_game(self):
        """
        random locations of items
        """
        self.engine = engine.GridEngine(self.shape)
        w = items.WallItem.create_border(self.shape)
        self.engine.add(w)
        self.agent = items.PickAgent((-1, -1))
        self.agent_dict = {self.agent_id: self.agent}

        self.objects = [items.ObjectItem((-1, -1)) for _ in range(self.num_objects)]

        self.goals = [items.GoalItem((-1, -1)) for _ in range(self.num_goals)]
        #self.obstacles = [items.WallItem((-1, -1)) for _ in range(self.num_obstacles)] # FIXME does not work
        self.engine._add_randomly([self.agent] + self.objects + self.goals)

    # item locations are specified
    def _create_game_1(self):
        """
        deterministic item locations
        """
        self.engine = engine.GridEngine(self.shape)
        w = items.WallItem.create_border(self.shape)
        self.engine.add(w)
        self.agent = items.PickAgent((6, 2))
        self.agent_dict = {self.agent_id: self.agent}
        self.objects = [items.ObjectItem((6, 1))]
        self.goals = [items.GoalItem((5, 3))]
        self.obstacles = [items.WallItem((5,2))]
        self.engine._add([self.agent] + self.objects + self.goals + self.obstacles)

    def _step(self, step_data: games.StepData) -> games.StepData:
        step_data["reward"] = float(0.)
        action = step_data.actions[self.agent] # step_data.actions maps agent objects to actions
        step_data.add_reward(-0.01) # small penalty for each action; this should incentivise to reach the goal as quick as possible;
        if action in ["up", "down", "left", "right"]:
            success = self.agent.move(action)
            if success:
                self.moves_since_pick += 1
                step_data["infos"]["pick_success"] = float(0.0)
                step_data["infos"]["move_failure"] = float(0.0)
                step_data["infos"]["pick_failure"] = float(0.0)
                step_data["infos"]["drop_failure"] = float(0.0)
                step_data["infos"]["game_success"] = float(0.0)
                step_data["infos"]["steps_exceeded"] = float(0.0)
            else:
                step_data.add_reward(-0.05) # illegal move
                step_data["infos"]["pick_success"] = float(0.0)
                step_data["infos"]["move_failure"] = -float(1.0)
                step_data["infos"]["pick_failure"] = float(0.0)
                step_data["infos"]["drop_failure"] = float(0.0)
                step_data["infos"]["game_success"] = float(0.0)
                step_data["infos"]["steps_exceeded"] = float(0.0)
            #if success and self.agent.attached:
            #    step_data.add_reward(1.0) # reward for successful move with attached object
        elif action == "pick":
            success = self.agent.pick()
            if success:
                self.moves_since_pick = 0
                # step_data.add_reward(0.1)
                # for debugging
                # print("+++++++ PICK SUCCESS +++++++")
                step_data["infos"]["pick_success"] = float(1.0)
                step_data["infos"]["pick_failure"] = float(0.0)
                step_data["infos"]["move_failure"] = float(0.0)
                step_data["infos"]["drop_failure"] = float(0.0)
                step_data["infos"]["game_success"] = float(0.0)
                step_data["infos"]["steps_exceeded"] = float(0.0)
            else:
                # step_data.add_reward(-0.1) # illegal pick
                step_data["infos"]["pick_success"] = float(0.0)
                step_data["infos"]["pick_failure"] = -float(1.0)
                step_data["infos"]["move_failure"] = float(0.0)
                step_data["infos"]["drop_failure"] = float(0.0)
                step_data["infos"]["game_success"] = float(0.0)
                step_data["infos"]["steps_exceeded"] = float(0.0)
        elif action == "drop":
            success = self.agent.drop()
            if success:
                # step_data.add_reward(0.1)
                step_data["infos"]["drop_failure"] = float(0.0)
                step_data["infos"]["move_failure"] = float(0.0)
                step_data["infos"]["pick_failure"] = float(0.0)
                step_data["infos"]["game_success"] = float(0.0)
                step_data["infos"]["steps_exceeded"] = float(0.0)
                step_data["infos"]["pick_success"] = float(0.0)
            else:
                # step_data.add_reward(-0.1) # illegal drop
                step_data["infos"]["drop_failure"] = -float(1.0)
                step_data["infos"]["move_failure"] = float(0.0)
                step_data["infos"]["pick_failure"] = float(0.0)
                step_data["infos"]["game_success"] = float(0.0)
                step_data["infos"]["steps_exceeded"] = float(0.0)
                step_data["infos"]["pick_success"] = float(0.0)

        # Check game end
        if self._check_goal(): # various _check_goal* methods are implemented
            #print("+++++++++++ game success ++++++++++++")
            step_data.add_reward(1.0)
            step_data["infos"]["game_success"] = float(1.0)
            step_data["infos"]["steps_exceeded"] = float(0.0)
            step_data.terminate_game({"goal_reached": True})
        elif step_data.steps_exceeded(self.max_steps):
            # step_data.add_reward(-1.0)
            step_data["infos"]["steps_exceeded"] = -float(1.0)
            step_data["infos"]["game_success"] = float(0.0)
            step_data.terminate_game({"goal_reached": False})

        return step_data

    def _check_goal(self) -> bool:
        goal_locs = np.stack([g.loc for g in self.goals], 0)
        at_goals = self.engine.items(locs=goal_locs)
        dropped_objs = [
            i for i in at_goals if i.kind == items.ItemKind.OBJECT and i.dropped
        ]
        # for debugging
        if len(dropped_objs) == len(self.goals):
            pass
            #print("++++++++ ALL OBJECTS ON GOALS ++++++++")
        return len(dropped_objs) == len(self.goals)

    def _check_goal_pick(self) -> bool:
        """
        game is successful if one object has been picked
        """
        if self.agent.attached:
            # for debugging
            print("++++++++ PICK SUCCESSFUL ++++++++")
            return True
        else:
            return False

    def _check_goal_find_goal(self) -> bool:
        """
        copied from find goal game; the game is successful if a goal has been found, i.e. the agent moves on a goal
        """
        the_items = self.engine.items(locs=self.agent.loc)
        the_items = [i for i in the_items if i.kind == items.ItemKind.GOAL]
        return len(the_items) > 0

    def _check_goal_movewithobject(self) -> bool:
        """
        game is successful if the object has been picked and the agent has made a number of moves with the attached object;
        """
        if self.agent.attached and self.moves_since_pick == 2:
            print("++++++++ PICK SUCCESSFUL, MOVED TWICE ++++++++")
            return True
        else:
            return False

    def _check_goal_movewithobject_until_goaldistance(self, goal_distance) -> bool:
        """
        game success if the agent has moved with the object until a given distance to the goal;
        """
        distance_to_goal = self.move_distance(self.agent, self.goals[0])
        if self.agent.attached and distance_to_goal == goal_distance:
            print("++++++++++++++++ MOVED WITH OBJECT; DISTANCE TO GOAL: {} +++++++++++++".format(goal_distance))
            return True
        else:
            return False

    def render(self, image_observation:bool=True) -> np.ndarray:
        return defaults.render(self.engine, image_observation=image_observation)


class WarehouseGame_2(games.GameBase_multiagent):
    """
    the game: 2 agents have to transport 2 objects (each agent transports one object); the agents have to learn: 1) object selection, 2)
    efficient path, 3) collision avoidance;
    multi-agent game; environment just uses agent IDs, agent objects are created and used in the game;
    """
    def __init__(
        self,
        env_config:dict,
        agent_ids: List[str] = ['agent_1', 'agent_2'],
        #seed: int = None,
        shape=(8, 8),
        max_steps: int = 100,
        num_objects: int = 1,
        num_goals: int = 1,
        num_obstacles: int = 0,
        collaborative_transport = False,
        random_items = [True, False, False] # should items be initiated with random locations: [agents, objects, goals]
    ) -> None:
        super().__init__()
        self.shape = shape
        self.max_steps = max_steps
        self.agent_ids = agent_ids
        # self.agent_dict = {} # mapping of agent ID to agent object
        self.num_objects = num_objects
        self.num_goals = num_goals
        self.objects: List[items.ObjectItem] = None
        self.goals: List[items.GoalItem] = None
        self.num_obstacles=num_obstacles,
        self.moves_since_pick = 1000 # the number of successful moves since the successful pick action
        self.collaborative_transport=collaborative_transport
        self.random_items = random_items
        if 'seed' in env_config.keys():
            self.rng = np.random.default_rng(env_config['seed']) # seeding the random number generator for improved reproducibility of training results
        else:
            self.rng = np.random.default_rng(42)
        assert len(agent_ids) == 2
        assert num_objects == 2

        # configuration of reward scheme parameters
        self.reward_game_success = handle_not_element(env_config, "reward_game_success", 1.0)
        print("reward_game_success set to {}".format(self.reward_game_success))
        self.reward_each_action = handle_not_element(env_config, "reward_each_action", -0.01)
        print("reward_each_action set to {}".format(self.reward_each_action))
        self.reward_illegal_action = handle_not_element(env_config, "reward_illegal_action", -0.1)
        print("reward_illegal_action set to {}".format(self.reward_illegal_action))

    def _create_game(self, goal_area:bool=False):
        # create wall encircling the gridworld
        self.engine = engine.GridEngine(self.shape, rng=self.rng)
        w = items.WallItem.create_border(self.shape)
        self.engine.add(w)

        rand_items = [] # list of items with random initial locations
        specified_items = [] # list of items with specified initial locations

        assert self.num_objects == 2
        assert len(self.agent_ids) == 2

        if self.random_items[0] == True: # random agent locations
            for agent_id in self.agent_ids:
                agent = items.PickAgent((-1, -1), collaborative_transport=self.collaborative_transport)
                self.agent_dict[agent_id] = agent
                rand_items.append(agent)
        else: # specify agent locations
            agent_1 = items.PickAgent((3, 3), collaborative_transport=self.collaborative_transport)
            agent_2 = items.PickAgent((4, 2), collaborative_transport=self.collaborative_transport)
            self.agent_dict[self.agent_ids[0]] = agent_1
            self.agent_dict[self.agent_ids[1]] = agent_2
            specified_items.append(agent_1)
            specified_items.append(agent_2)

        if self.random_items[1] == True: # random object locations
            self.objects = [items.ObjectItem((-1,-1)) for _ in range(self.num_objects)]
            for o in self.objects:
                rand_items.append(o)
        else: # specify object locations
            self.objects = [items.ObjectItem((1, 1)), items.ObjectItem((1, 2))]
            for o in self.objects:
                specified_items.append(o)

        if self.random_items[2] == True: # random goal locations
            self.goals = [items.GoalItem((-1,-1) for _ in range(self.num_goals))]
            for g in self.goals:
                rand_items.append(g)
        else: # specify goal locations
            self.goals = [items.GoalItem((7, 8)), items.GoalItem((8, 8))]
            for g in self.goals:
                specified_items.append(g)

        self.engine._add(specified_items)
        self.engine._add_randomly(rand_items)

    def _create_game_deterministic(self, goal_area:bool = False):
        """
        deterministic locations for goals, objects and 2 agents
        """
        self.engine = engine.GridEngine(self.shape, rng=self.rng)
        w = items.WallItem.create_border(self.shape)
        self.engine.add(w)

        assert len(self.agent_ids) == 2
        agent_1 = items.PickAgent((3, 3), collaborative_transport=self.collaborative_transport)
        agent_2 = items.PickAgent((4, 2), collaborative_transport=self.collaborative_transport)
        self.agent_dict[self.agent_ids[0]] = agent_1
        self.agent_dict[self.agent_ids[1]] = agent_2

        self.objects = [items.ObjectItem((4, 3)), items.ObjectItem((5, 2))]
        self.goals = [items.GoalItem((1, 6)), items.GoalItem((8, 8))]
        # self.obstacles = [items.WallItem((5,2))]
        self.engine._add(list(self.agent_dict.values()) + self.objects + self.goals)


    def _step(self, step_data: games.StepData_multiagent) -> games.StepData_multiagent:
        self.rng.shuffle(self.active_agent_ids) # avoid bias due to the sequence of agents in the list
        for agent in self.active_agent_ids:
            agent_obj = self.agent_dict[agent]
            # for each agent in turn: apply action and reward agent
            step_data[agent]["reward"] = float(0.)
            action = step_data[agent]['action']
            step_data.add_reward(self.reward_each_action, agent) # small penalty for each action; this should incentivise to reach the goal as quick as possible;
            step_data[agent]["infos"]["game_success"] = float(0.0)
            step_data[agent]["infos"]["steps_exceeded"] = float(0.0)
            if action in ["up", "down", "left", "right"]:
                step_data[agent]["infos"]["pick_success"] = float(0.0)
                step_data[agent]["infos"]["pick_failure"] = float(0.0)
                step_data[agent]["infos"]["drop_failure"] = float(0.0)
                success = agent_obj.move(action)
                if success:
                    step_data[agent]["infos"]["move_failure"] = float(0.0)
                else:
                    step_data.add_reward(self.reward_illegal_action, agent) # illegal move
                    step_data[agent]["infos"]["move_failure"] = -float(1.0)

                    # debug output for move failures;  with masked move actions, success=False should actually not occur;
                    """
                    if action == "left":
                        loc = np.array([agent_obj.loc + (-1, 0)])
                        if not agent_obj.picked_object == None:
                            loc_att_obj = np.array([agent_obj.picked_object.loc + (-1, 0)])
                    elif action == "right":
                        loc = np.array([agent_obj.loc + (1, 0)])
                        if not agent_obj.picked_object == None:
                            loc_att_obj = np.array([agent_obj.picked_object.loc + (1, 0)])
                    elif action == "up":
                        loc = np.array([agent_obj.loc + (0, -1)])
                        if not agent_obj.picked_object == None:
                            loc_att_obj = np.array([agent_obj.picked_object.loc + (0, -1)])
                    elif action == "down":
                        loc = np.array([agent_obj.loc + (0, 1)])
                        if not agent_obj.picked_object == None:
                            loc_att_obj = np.array([agent_obj.picked_object.loc + (0, 1)])
                    items_loc = self.engine.items(loc)
                    if agent_obj.picked_object == None:
                        # no object attached
                        print("+++++ MOVE FAILURE: agent: {}, action: {}, items on target location: {}".format(agent, action, items_loc))
                    else:
                        # object attached
                        items_loc_att_obj = self.engine.items(loc_att_obj)
                        if agent_obj in items_loc_att_obj:
                            items_loc_att_obj.remove(agent_obj)
                        if agent_obj.picked_object in items_loc:
                            items_loc.remove(agent_obj.picked_object)
                        print("+++++ MOVE FAILURE with object attached: agent: {}, action: {}, items on target location: {}, "
                              "items on target loc w.r.t. attached object: {}".format(agent, action, items_loc, items_loc_att_obj))
                    """

                #if success and self.agent.attached:
                #    step_data.add_reward(1.0) # reward for successful move with attached object
            elif action == "pick":
                step_data[agent]["infos"]["move_failure"] = float(0.0)
                step_data[agent]["infos"]["drop_failure"] = float(0.0)
                success = agent_obj.pick()
                if success:
                    # step_data.add_reward(0.1)
                    # for debugging
                    # print("+++++++ PICK SUCCESS +++++++")
                    step_data[agent]["infos"]["pick_success"] = float(1.0)
                    step_data[agent]["infos"]["pick_failure"] = float(0.0)
                else:
                    step_data.add_reward(self.reward_illegal_action, agent) # illegal pick
                    step_data[agent]["infos"]["pick_success"] = float(0.0)
                    step_data[agent]["infos"]["pick_failure"] = -float(1.0)
            elif action == "drop":
                step_data[agent]["infos"]["move_failure"] = float(0.0)
                step_data[agent]["infos"]["pick_failure"] = float(0.0)
                step_data[agent]["infos"]["pick_success"] = float(0.0)
                success = agent_obj.drop()
                if success:
                    step_data.add_reward(self.reward_game_success, agent) # the agent has achieved its goal (i.e., transport 1 object) and terminates
                    step_data[agent]["infos"]["drop_failure"] = float(0.0)
                    step_data[agent]["terminated"] = True # after successful drop, the agent terminates
                else:
                    step_data.add_reward(self.reward_illegal_action, agent) # illegal drop
                    step_data[agent]["infos"]["drop_failure"] = -float(1.0)
            elif action =="do_nothing":
                step_data[agent]["infos"]["move_failure"] = float(0.0)
                step_data[agent]["infos"]["pick_failure"] = float(0.0)
                step_data[agent]["infos"]["pick_success"] = float(0.0)
                step_data[agent]["infos"]["drop_failure"] = float(0.0)

        # Check game end
        if self._check_goal(): # various _check_goal* methods are implemented
            #print("+++++++++++ game success ++++++++++++")
            for agent in self.active_agent_ids:
                #step_data.add_reward(self.reward_game_success, agent)
                step_data[agent]["infos"]["game_success"] = float(1.0)
                step_data[agent]['terminated'] = True
            step_data.terminate_game({"goal_reached": True})
        elif step_data.steps_exceeded(self.max_steps):
            for agent in self.active_agent_ids:
                # step_data.add_reward(-1.0, agent)
                step_data[agent]["infos"]["steps_exceeded"] = -float(1.0)
                step_data[agent]['terminated'] = True
            step_data.terminate_game({"goal_reached": False})

        return step_data

    def _check_goal(self) -> bool:
        goal_locs = np.stack([g.loc for g in self.goals], 0)
        at_goals = self.engine.items(locs=goal_locs)
        dropped_objs = [
            i for i in at_goals if i.kind == items.ItemKind.OBJECT and i.dropped
        ]
        # for debugging
        if len(dropped_objs) == len(self.objects):
            print("++++++++ ALL OBJECTS ON GOALS ++++++++")
        return len(dropped_objs) == len(self.objects)

    def _check_goal_pick(self) -> bool:
        """
        game is successful if one object has been picked
        """
        if self.agent.attached:
            # for debugging
            print("++++++++ PICK SUCCESSFUL ++++++++")
            return True
        else:
            return False

    def _check_goal_find_goal(self) -> bool:
        """
        copied from find goal game; the game is successful if a goal has been found, i.e. the agent moves on a goal
        """
        the_items = self.engine.items(locs=self.agent.loc)
        the_items = [i for i in the_items if i.kind == items.ItemKind.GOAL]
        return len(the_items) > 0

    def _check_goal_movewithobject(self) -> bool:
        """
        game is successful if the object has been picked and the agent has made a number of moves with the attached object;
        """
        if self.agent.attached and self.moves_since_pick == 2:
            print("++++++++ PICK SUCCESSFUL, MOVED TWICE ++++++++")
            return True
        else:
            return False

    def _check_goal_movewithobject_until_goaldistance(self, goal_distance) -> bool:
        """
        game success if the agent has moved with the object until a given distance to the goal;
        """
        distance_to_goal = self.move_distance(self.agent, self.goals[0])
        if self.agent.attached and distance_to_goal == goal_distance:
            print("++++++++++++++++ MOVED WITH OBJECT; DISTANCE TO GOAL: {} +++++++++++++".format(goal_distance))
            return True
        else:
            return False

    def render(self, image_observation:bool=True) -> np.ndarray:
        return defaults.render(self.engine, image_observation=image_observation)

    def is_adjacent_to_impassable(self, direction: str, item: ItemBase) -> (bool, ItemBase):
        """
        Returns True if an impassable item is adjacent to param "item" in the direction of param "direction"
        """
        if direction == "left":
            loc = np.array([item.loc + (-1, 0)])
        elif direction == "right":
            loc = np.array([item.loc + (1, 0)])
        elif direction == "up":
            loc = np.array([item.loc + (0, -1)])
        elif direction == "down":
            loc = np.array([item.loc + (0, 1)])

        items = self.engine.items(loc)
        contains_impassable = False
        impassable_item = None
        for item in items:
            if item.impassable:
                contains_impassable = True
                impassable_item = item
                break
        return contains_impassable, impassable_item

    def is_adjacent_to(self, direction: str, item: ItemBase, other_item: ItemBase):
        """
        Returns True if other_item is adjacent to item in the direction of param direction
        """
        if direction == "left":
            loc = np.array([item.loc + (-1, 0)])
        elif direction == "right":
            loc = np.array([item.loc + (1, 0)])
        elif direction == "up":
            loc = np.array([item.loc + (0, -1)])
        elif direction == "down":
            loc = np.array([item.loc + (0, 1)])

        items = self.engine.items(loc)
        contains_item = False
        for it in items:
            if it == item:
                contains_item = True
                break
        return contains_item

class WarehouseGame_3_notused(games.GameBase):
    """
    single-agent game; the human player is part of the game as a dynamic obstacle, and he is controlled by a heuristic
    algorithm;
    """
    def __init__(
        self,
        env_config,
        #agent_ids: List[str] = ['agent_1', 'agent_2'],
        seed: int = None,
        shape=(8, 8),
        max_steps: int = 100,
        num_objects: int = 1,
        num_goals: int = 1,
        num_obstacles: int = 0,
    ) -> None:
        super().__init__(seed=seed)
        self.shape = shape
        self.max_steps = max_steps
        #self.agent_ids = agent_ids
        # self.agent_dict = {} # mapping of agent ID to agent object
        self.num_objects = num_objects
        self.num_goals = num_goals
        self.objects: List[items.ObjectItem] = None
        self.goals: List[items.GoalItem] = None
        self.num_obstacles=num_obstacles,
        self.moves_since_pick = 1000 # the number of successful moves since the successful pick action

        # configuration of reward scheme parameters
        self.reward_game_success = handle_not_element(env_config, "reward_game_success", 1.0)
        print("reward_game_success set to {}".format(self.reward_game_success))
        self.reward_each_action = handle_not_element(env_config, "reward_each_action", -0.01)
        print("reward_each_action set to {}".format(self.reward_each_action))
        self.reward_illegal_action = handle_not_element(env_config, "reward_illegal_action", -0.1)
        print("reward_illegal_action set to {}".format(self.reward_illegal_action))

    def _create_game(self):
        """
        random locations of items
        """
        self.engine = engine.GridEngine(self.shape)
        w = items.WallItem.create_border(self.shape)
        self.engine.add(w)

        #for agent_id in self.agent_ids:
        #    agent = items.PickAgent((-1, -1))
        #    self.agent_dict[agent_id] = agent
        self.agent = items.PickAgent((-1, -1))
        self.objects = [items.ObjectItem((-1, -1)) for _ in range(self.num_objects)]

        self.goals = [items.GoalItem((-1, -1)) for _ in range(self.num_goals)]
        #self.obstacles = [items.WallItem((-1, -1)) for _ in range(self.num_obstacles)] # FIXME does not work
        self.engine._add_randomly([self.agent] + self.objects + self.goals) # TODO add human agent randomly

    def _create_game_1(self):
        """
        deterministic locations for goals, objects and 2 agents
        """
        self.engine = engine.GridEngine(self.shape)
        w = items.WallItem.create_border(self.shape)
        self.engine.add(w)

        self.agent = items.PickAgent((3, 3))
        #agent_2 = items.PickAgent((4, 2))
        #self.agent_dict[self.agent_ids[0]] = agent_1
        #self.agent_dict[self.agent_ids[1]] = agent_2

        self.objects = [items.ObjectItem((4, 3)), items.ObjectItem((5, 2))]
        self.goals = [items.GoalItem((1, 6)), items.GoalItem((8, 8))]
        # self.obstacles = [items.WallItem((5,2))]
        self.engine._add([self.agent] + self.objects + self.goals) # TODO add deterministic location for human agent


    def _step(self, step_data: games.StepData) -> games.StepData:
        #random.shuffle(self.agent_ids) # avoid bias due to the sequence of agents in list agent_ids
        #for agent in self.agent_ids:
        #    agent_obj = self.agent_dict[agent]
            # for each agent in turn: apply action and reward agent
        step_data["reward"] = float(0.)
        actions = list(step_data['actions'].values())
        assert len(actions) == 1 # exactly one action, as we only consider single-agent RL
        action = actions[0]
        step_data.add_reward(self.reward_each_action) # small penalty for each action; this should incentivise to reach the goal as quick as possible;
        step_data["infos"]["game_success"] = float(0.0)
        step_data["infos"]["steps_exceeded"] = float(0.0)
        if action in ["up", "down", "left", "right"]:
            step_data["infos"]["pick_success"] = float(0.0)
            step_data["infos"]["pick_failure"] = float(0.0)
            step_data["infos"]["drop_failure"] = float(0.0)
            success = self.agent.move(action)
            if success:
                step_data["infos"]["move_failure"] = float(0.0)
            else:
                step_data.add_reward(self.reward_illegal_action) # illegal move
                step_data["infos"]["move_failure"] = -float(1.0)

                #if success and self.agent.attached:
                #step_data.add_reward(1.0) # reward for successful move with attached object
        elif action == "pick":
            step_data["infos"]["move_failure"] = float(0.0)
            step_data["infos"]["drop_failure"] = float(0.0)
            success = self.agent.pick()
            if success:
                # step_data.add_reward(0.1)
                # for debugging
                # print("+++++++ PICK SUCCESS +++++++")
                step_data["infos"]["pick_success"] = float(1.0)
                step_data["infos"]["pick_failure"] = float(0.0)
            else:
                step_data.add_reward(self.reward_illegal_action) # illegal pick
                step_data["infos"]["pick_success"] = float(0.0)
                step_data["infos"]["pick_failure"] = -float(1.0)
        elif action == "drop":
            step_data["infos"]["move_failure"] = float(0.0)
            step_data["infos"]["pick_failure"] = float(0.0)
            step_data["infos"]["pick_success"] = float(0.0)
            success = self.agent.drop()
            if success:
                # step_data.add_reward(0.1, agent)
                step_data["infos"]["drop_failure"] = float(0.0)
            else:
                step_data.add_reward(self.reward_illegal_action) # illegal drop
                step_data["infos"]["drop_failure"] = -float(1.0)
        elif action =="do_nothing":
            step_data["infos"]["move_failure"] = float(0.0)
            step_data["infos"]["pick_failure"] = float(0.0)
            step_data["infos"]["pick_success"] = float(0.0)
            step_data["infos"]["drop_failure"] = float(0.0)

        # Check game end
        if self._check_goal(): # various _check_goal* methods are implemented
            #print("+++++++++++ game success ++++++++++++")
            #for agent in self.agent_ids:
            step_data.add_reward(self.reward_game_success)
            step_data["infos"]["game_success"] = float(1.0)
            step_data['terminated'] = True
            step_data.terminate_game({"goal_reached": True})
        elif step_data.steps_exceeded(self.max_steps):
            #for agent in self.agent_ids:
            # step_data.add_reward(-1.0)
            step_data["infos"]["steps_exceeded"] = -float(1.0)
            step_data['terminated'] = True
            step_data.terminate_game({"goal_reached": False})

        return step_data

    def _check_goal(self) -> bool:
        goal_locs = np.stack([g.loc for g in self.goals], 0)
        at_goals = self.engine.items(locs=goal_locs)
        dropped_objs = [
            i for i in at_goals if i.kind == items.ItemKind.OBJECT and i.dropped
        ]
        # for debugging
        #if len(dropped_objs) == len(self.goals):
         #   print("++++++++ ALL OBJECTS ON GOALS ++++++++")
        return len(dropped_objs) == len(self.goals)

    def _check_goal_pick(self) -> bool:
        """
        game is successful if one object has been picked
        """
        if self.agent.attached:
            # for debugging
            print("++++++++ PICK SUCCESSFUL ++++++++")
            return True
        else:
            return False

    def _check_goal_find_goal(self) -> bool:
        """
        copied from find goal game; the game is successful if a goal has been found, i.e. the agent moves on a goal
        """
        the_items = self.engine.items(locs=self.agent.loc)
        the_items = [i for i in the_items if i.kind == items.ItemKind.GOAL]
        return len(the_items) > 0

    def _check_goal_movewithobject(self) -> bool:
        """
        game is successful if the object has been picked and the agent has made a number of moves with the attached object;
        """
        if self.agent.attached and self.moves_since_pick == 2:
            print("++++++++ PICK SUCCESSFUL, MOVED TWICE ++++++++")
            return True
        else:
            return False

    def _check_goal_movewithobject_until_goaldistance(self, goal_distance) -> bool:
        """
        game success if the agent has moved with the object until a given distance to the goal;
        """
        distance_to_goal = self.move_distance(self.agent, self.goals[0])
        if self.agent.attached and distance_to_goal == goal_distance:
            print("++++++++++++++++ MOVED WITH OBJECT; DISTANCE TO GOAL: {} +++++++++++++".format(goal_distance))
            return True
        else:
            return False

    def render(self, image_observation:bool=True) -> np.ndarray:
        return defaults.render(self.engine, image_observation=image_observation)

    def is_adjacent_to_impassable(self, direction: str, item: ItemBase) -> (bool, ItemBase):
        """
        Returns True if an impassable item is adjacent to param "item" in the direction of param "direction"
        """
        if direction == "left":
            loc = np.array([item.loc + (-1, 0)])
        elif direction == "right":
            loc = np.array([item.loc + (1, 0)])
        elif direction == "up":
            loc = np.array([item.loc + (0, -1)])
        elif direction == "down":
            loc = np.array([item.loc + (0, 1)])

        items = self.engine.items(loc)
        contains_impassable = False
        impassable_item = None
        for item in items:
            if item.impassable:
                contains_impassable = True
                impassable_item = item
                break
        return contains_impassable, impassable_item

    def is_adjacent_to(self, direction: str, item: ItemBase, other_item: ItemBase):
        """
        Returns True if other_item is adjacent to item in the direction of param direction
        """
        if direction == "left":
            loc = np.array([item.loc + (-1, 0)])
        elif direction == "right":
            loc = np.array([item.loc + (1, 0)])
        elif direction == "up":
            loc = np.array([item.loc + (0, -1)])
        elif direction == "down":
            loc = np.array([item.loc + (0, 1)])

        items = self.engine.items(loc)
        contains_item = False
        for it in items:
            if it == item:
                contains_item = True
                break
        return contains_item


class WarehouseGame_3(games.GameBase_multiagent):
    """
    multi-agent game; environment just uses agent IDs, agent objects are created and used in the game;
    """
    def __init__(
        self,
        env_config,
        agent_ids: List[str] = ['agent_1'],
        shape=(10, 10),
        max_steps: int = 100,
        num_objects: int = 1,
        num_goals: int = 1,
        obstacle: bool = True,
        dynamic: bool= True,
        initial_dynamic_pos = [2,2],
        mode = "training",
        goal_area = False,
        three_grid_object = False,
        seed = 42
    ) -> None:
        super().__init__()
        self.shape = shape
        self.max_steps = max_steps
        self.agent_ids = agent_ids
        self.agent_dict = {} # mapping of agent ID to agent object
        self.num_objects = num_objects
        #self.num_goals = num_goals # not used
        self.objects: List[items.ObjectItem] = None
        self.goals: List[items.GoalItem] = None
        self.obstacle= obstacle
        self.dynamic = dynamic
        self.goal_area = goal_area
        self.initial_dynamic_pos = initial_dynamic_pos
        self.mode = mode
        self.three_grid_object = three_grid_object
        self.seed = seed
        print("Ware house games seed 8ius", self.seed)

        self.moves_since_pick = 1000 # the number of successful moves since the successful pick action

        print(env_config)
       
        self.rng = np.random.default_rng(self.seed) # seeding the random number generator for improved reproducibility of training results
        self.seed = self.seed
     
        # configuration of reward scheme parameters
        self.reward_game_success = handle_not_element(env_config, "reward_game_success", 1.0)
        print("reward_game_success set to {}".format(self.reward_game_success))
        self.reward_each_action = handle_not_element(env_config, "reward_each_action", -0.01)
        print("reward_each_action set to {}".format(self.reward_each_action))
        self.reward_illegal_action = handle_not_element(env_config, "reward_illegal_action", -0.1)
        print("reward_illegal_action set to {}".format(self.reward_illegal_action))
        self.reward_wrong_pick = handle_not_element(env_config, "reward_wrong_pick", -0.3)
        print("reward_wrong_pick set to {}".format(self.reward_wrong_pick))
        self.reward_succesful_pick = handle_not_element(env_config, "reward_succesful_pick", -0.15)
        print("reward_succesful_pick set to {}".format(self.reward_succesful_pick))
        self.reward_wrong_attachment = handle_not_element(env_config, "reward_wrong_attachment", -0.1)
        print("reward_wrong_attachment set to {}".format(self.reward_wrong_attachment))


    def _create_game(self, goal_area:bool = False):
        """
        random locations of items
        """
        
        self.engine = engine.GridEngine(self.shape, rng=self.rng)
        w = items.WallItem.create_border(self.shape)
        self.engine.add(w)
        

        for agent_id in self.agent_ids:

            if agent_id != 'random_agent':
                #creates heuristic agent if kind = heuristic_agent
                agent = items.PickAgent((-1, -1), kind = agent_id, three_grid_object= self.three_grid_object)
                self.agent_dict[agent_id] = agent


        if goal_area:
            
            # create goal area
            #if three grid object then create goal area of size 3
            if self.three_grid_object:
                self.goals, goals_coordinates = self.create_goal_area(size=3)
        #select objects location randomly not overlapping with goal area coordinates
                object_cells = list(itertools.product([x for x in range(2,self.shape[0]-2)], [x for x in range(2,self.shape[1]-2)]))
                object_cells = [x for x in object_cells if (x[0]-1,x[1]) 
                            not in goals_coordinates and  (x[0]+1,x[1])
                            not in goals_coordinates and (x[0],x[1]-1)
                            not in goals_coordinates and (x[0],x[1]+1)
                            not in goals_coordinates and x
                            not in goals_coordinates]
                self.objects = []
                
                chosen_coord = tuple(self.rng.choice(object_cells))
                horizontal_coordinates = [chosen_coord,(chosen_coord[0]+1,chosen_coord[1]),(chosen_coord[0]-1,chosen_coord[1])]
                vertical_coordinates = [chosen_coord,(chosen_coord[0],chosen_coord[1]+1),(chosen_coord[0],chosen_coord[1]-1)]
                three_grid_coordinates = [horizontal_coordinates, vertical_coordinates][self.rng.choice([0, 1])]


                self.objects.append(items.ObjectItem(three_grid_coordinates, num_carriers = len(self.agent_dict)))
                for tup in three_grid_coordinates:
                    if tup in object_cells:

                        object_cells.remove(tup)
            else:
                self.goals, goals_coordinates = self.create_goal_area(size=2)
           
                #select objects location randomly not overlapping with goal area coordinates
                object_cells = list(itertools.product([x for x in range(1,self.shape[0]-1)], [x for x in range(1,self.shape[1]-1)]))
                print(goals_coordinates,'goals_coordinates')
                object_cells = [object_cell for object_cell in object_cells if (object_cell[0], object_cell[1]) not in goals_coordinates]
                print(object_Cells, 'objects cells without goal coordinates')

                                
                self.objects = []
                for i in range(self.num_objects): # TODO if goal_area == False, self.num_objects is not used
                    chosen_coord = tuple(self.rng.choice(object_cells))
                    self.objects.append(items.ObjectItem(chosen_coord, num_carriers = len(self.agent_dict)))
                    object_cells.remove(chosen_coord)


            self.engine._add(list(self.goals))
            self.engine._add(list(self.objects))
            
            self.engine._add_randomly(list(self.agent_dict.values())) 
         
            
        else:

            
            # create a single goal location
            #select random x and y for goal but it shouldn't be at the edges
            goal_x = np.random.randint(2,self.shape[0]-3) #if shape is 10 then 10-3 select any from 2 to 7 grid dont select 0,1 and 8,9
            goal_y = np.random.randint(2,self.shape[1]-3)
            self.goals = [items.GoalItem((goal_x, goal_y))]


            #create a single object
            #select object location randomly not overlapping with goal location
            object_cells = list(itertools.product([x for x in range(2,self.shape[0]-2)], [x for x in range(2,self.shape[1]-2)]))
            object_cells = [x for x in object_cells if x not in [(goal_x, goal_y)]]
            #in case of two policy agents don't have object at corners as can't be attached by three agents
            policy_agents = [agent for agent in self.agent_ids if agent!='heuristic_agent' and agent!='random_agent']
            if len(policy_agents)>1:
            
                corner_cells = [(1,self.shape[1]-2), (1,1), (self.shape[0]-2, 1), (self.shape[0]-2,self.shape[1]-2)]
     
                object_cells = [cell for cell in object_cells if cell not in corner_cells]


            object_x, object_y= random.choice(object_cells)

            #pop random agent if its there
            self.agent_dict.pop("random_agent", None)
            self.objects = [items.ObjectItem((object_x, object_y), num_carriers = len(self.agent_dict))]
          

            if self.obstacle:           

                #if obstacle is dynamic use random agent as obstacle item 
                if self.dynamic:
  
                 
                    #grid cells are all cells excluding walls for dynamic obstacle; 
                    #for eg range(1,9) : 1,2,3,4,5,6,7,8 ; 0and 9 will be walls
                    grid_cells = list(itertools.product([x for x in range(1,self.shape[0]-1)], [x for x in range(1,self.shape[1]-1)]))
                    occupied_cells = [(object_x, object_y), (goal_x, goal_y)]

                

                    #for dynamic object all are free cells except cells occupied by object and goal
                    free_obs_cells = [x for x in grid_cells if x not in occupied_cells]


                    #if initial position is free
                    if self.initial_dynamic_pos in free_obs_cells:
                        #add dynamic obstacle in a fixed initial position while generating env
                        self.obstacles = [items.PickAgent(self.initial_dynamic_pos, kind="random_agent")]
                    else:

                        #adjacent grid cells of  initial position i.e left,right,up, down and all diagonal cells to the postion cell
                        points_near = [(self.initial_dynamic_pos[0]+1, self.initial_dynamic_pos[1]),(self.initial_dynamic_pos[0]-1, self.initial_dynamic_pos[1]),
                                        (self.initial_dynamic_pos[0]+1, self.initial_dynamic_pos[1]+1),(self.initial_dynamic_pos[0]-1, self.initial_dynamic_pos[1]-1),
                                        (self.initial_dynamic_pos[0]+1, self.initial_dynamic_pos[1]-1),(self.initial_dynamic_pos[0]-1, self.initial_dynamic_pos[1]+1),
                                        (self.initial_dynamic_pos[0], self.initial_dynamic_pos[1]+1),(self.initial_dynamic_pos[0], self.initial_dynamic_pos[1]-1)]

                        #if initial position is not free check adjacent free position and use that for initial position
                        free_adjacent_points = [x for x in points_near if x in free_obs_cells]
                        self.obstacles = [items.PickAgent(free_adjacent_points[0], kind="random_agent")]
        

                    #add object, goal, obstacle and agents in the environment
                    self.engine._add(list(self.objects))
                    self.engine._add(list(self.goals))
                    # add dynamic obstacle not randomly
                    self.engine._add(list(self.obstacles))
                    #remove random agent from agent dict if it is there
                    self.agent_dict.pop("random_agent", None)
                    #add agents to env randomly
                    self.engine._add_randomly(list(self.agent_dict.values())) 
                    #now add dynamic obstacle as random agent to agent dict
                    self.agent_dict["random_agent"] = self.obstacles[0]
                else:
                    # get all locations within walls
                    grid_cells = list(itertools.product([x for x in range(1,self.shape[0]-1)], [x for x in range(1,self.shape[1]-1)]))
                    

                    #lambda func which gives list of neighboring coordinates
                    neighbors = lambda x, y : [(x2, y2) for x2 in range(x-1, x+2)
                                                for y2 in range(y-1, y+2)
                                                if (-1 < x <= self.shape[0] and
                                                    -1 < y <= self.shape[1] and
                                                    (x != x2 or y != y2) and
                                                    (0 <= x2 <= self.shape[0]) and
                                                    (0 <= y2 <= self.shape[1]))]
                    
                    # obstacle shoudn't be around goal location and object location and around their neighboring cells
                    occupied_cells = [(object_x, object_y),(goal_x, goal_y)]
                    occupied_cells.extend(neighbors(object_x,object_y))
                    occupied_cells.extend(neighbors(goal_x,goal_y))

                    free_obs_cells= [x for x in grid_cells if x not in occupied_cells]
                    
                    #if obstacle is static just use wall item in free cells as obstacle
                    self.obstacles = [items.WallItem((random.choice(free_obs_cells)))]
                    self.engine._add(list(self.obstacles)) 
                    self.engine._add(list(self.objects))
                    self.engine._add(list(self.goals))
                    
                    self.engine._add_randomly(list(self.agent_dict.values())) 
            else:
                # no obstacle
                self.engine._add(list(self.objects))
                self.engine._add(list(self.goals))
                self.engine._add_randomly(list(self.agent_dict.values()))

       


    def _create_game_deterministic(self, goal_area:bool = False):
        """
        initial locations for goals, objects and agents are specified
        """
        self.engine = engine.GridEngine(self.shape, rng=self.rng)
        w = items.WallItem.create_border(self.shape)
        self.engine.add(w)

        #assert len(self.agent_ids) == 2
        for agent_id in self.agent_ids:
            if agent_id != 'random_agent':
                if agent_id == 'heuristic_agent':
                    agent_obj = items.PickAgent((7, 1), kind = agent_id) # location of heuristic agent
       

                else:
                    agent_obj = items.PickAgent((8, 1), kind = agent_id) # location of policy agent
                self.agent_dict[agent_id] = agent_obj

        if self.num_objects == 2:
            self.objects = [items.ObjectItem((1, 8), num_carriers = len(self.agent_ids)), items.ObjectItem((8, 8), num_carriers = len(self.agent_ids))]
        else:
            self.objects = [items.ObjectItem((3, 6), num_carriers = len(self.agent_ids))]
        if goal_area:
            self.goals, goals_coordinates = self.create_goal_area(size=2, upper_left_corner=(3,3))
        else:
            if self.obstacle:
                #if dynamic obstacle
                if self.dynamic:

                    #all available grid cells in the grid
                    grid_cells = list(itertools.product([x for x in range(1,self.shape[0]-1)], [x for x in range(1,self.shape[1]-1)]))

                    #already occupied grid cells by agents, goal and object
                    occupied_cells = [(7, 1), (1, 1), (8, 1), (1,8), (4,5)]

                    #free cells
                    free_obs_cells = [x for x in grid_cells if x not in occupied_cells]

                    #if initial position is free
                    if self.initial_dynamic_pos in free_obs_cells:
                        #add dynamic obstacle in a fixed initial position while generating env
                        self.obstacles = [items.PickAgent(self.initial_dynamic_pos, kind="random_agent")]
                    else:

                        #adjacent grid cells of  initial position i.e left,right,up, down and all diagonal cells to the postion cell
                        points_near = [(self.initial_dynamic_pos[0]+1, self.initial_dynamic_pos[1]),(self.initial_dynamic_pos[0]-1, self.initial_dynamic_pos[1]),
                                        (self.initial_dynamic_pos[0]+1, self.initial_dynamic_pos[1]+1),(self.initial_dynamic_pos[0]-1, self.initial_dynamic_pos[1]-1),
                                        (self.initial_dynamic_pos[0]+1, self.initial_dynamic_pos[1]-1),(self.initial_dynamic_pos[0]-1, self.initial_dynamic_pos[1]+1),
                                        (self.initial_dynamic_pos[0], self.initial_dynamic_pos[1]+1),(self.initial_dynamic_pos[0], self.initial_dynamic_pos[1]-1)]

                        #if initial position is not free check adjacent free position and use that for initial position
                        free_adjacent_points = [x for x in points_near if x in free_obs_cells]
                        self.obstacles = [items.PickAgent(free_adjacent_points[0], kind="random_agent")]


                    
                   
                else:
                    self.obstacles = [items.WallItem((2,6))]


                
                self.engine._add(list(self.obstacles ))
            self.goals = [items.GoalItem((4, 5))]
        
        
        self.engine._add(list(self.objects ))
        #remove random agent from agent dict if it is there
        self.agent_dict.pop("random_agent", None)
        self.engine._add(list(self.agent_dict.values()))
        self.engine._add(list(self.goals ))
        
        
        if self.obstacle and self.dynamic:
            # add dynamic obstacle as random agent to agent dict
            self.agent_dict["random_agent"] = self.obstacles[0]


   


    def create_goal_area(self, size:int = 2, upper_left_corner = None) -> List[items.GoalItem]:
        """
        Creates a randomly positioned square goal area; the goal area does not cover locations on the edges of the gridworld;
        @param size: the size of the goal area is size x size
        @return: a list of GoalItem objects making up the goal area
        """
        area = [] # the return value
        # create upper left corner coordinates for goal area;
        # coordinates may not be on gridworld edges
        if not upper_left_corner == None:
            # deterministic goal area
            ulc_x = upper_left_corner[0]
            ulc_y = upper_left_corner[1]
        else:
            # random goal area
            ulc_x = np.random.randint(2, self.shape[0] - 3)  # if shape is 10 then 10-3 select any from 2 to 7 grid do not select 0,1 and 8,9
            ulc_y = np.random.randint(2, self.shape[1] - 3)
        # determine lower right coordinates for goal area
        lrc_x = ulc_x + size - 1
        lrc_y = ulc_y + size - 1
        # determine required (x,y) offsets for goal area, so no goal items are on gridworld edges
        offset_x = max(0, lrc_x - (self.shape[0] - 3))
        offset_y = max(0, lrc_y - (self.shape[1] - 3))
        ulc_x = ulc_x - offset_x
        ulc_y = ulc_y - offset_y

        area_coordinates = []
        # construct goal area
        for x in range(ulc_x, ulc_x + size):
            for y in range(ulc_y, ulc_y + size):
                area.append(items.GoalItem((x, y)))
                area_coordinates.append((x,y))

        """
        the following code is not working
        
        # create "seed" coordinates of the area
        seed_x = random.randint(2, self.shape[0] - 3)  # if shape is 10 then 10-3 select any from 2 to 7 grid dont select 0,1 and 8,9
        seed_y = random.randint(2, self.shape[1] - 3)
        area.append(items.GoalItem((seed_x, seed_y)))
        required_goal_items = size * size
        #edge_x_len = 1
        #edge_y_len = 1
        #previous_x = seed_x
        #previous_y = seed_y

        # construct the other coordinates around the seed area, making up the square goal area, not covering locations on the edges of the gridworld;
        for x in range(seed_x - (size - 1), seed_x + size):
            if len(area) == required_goal_items:
                break # already enough goal items to cover the goal area
            if x < 2 or x > self.shape[0]-3:
                continue # avoid coordinates on the edges
            #if not x == previous_x:
            #    previous_x = x
            #    edge_x_len += 1
            for y in range(seed_y - (size - 1), seed_y + size):
                if y < 2 or y > self.shape[1]-3:
                    continue
                #if not y == previous_y:
                #    previous_y = y
                #    edge_y_len += 1
                if not x == seed_x or not y == seed_y:
                    area.append(items.GoalItem((x, y)))
                #if edge_y_len == area_size:
                #    break
                if len(area) == required_goal_items:
                    break
            #if edge_x_len == area_size:
            #    break
            
            """
        return (area,area_coordinates)

    def _step(self, step_data: games.StepData_multiagent) -> games.StepData_multiagent:
        #self.rng.shuffle(self.agent_ids) # avoid bias due to the sequence of agents in list agent_ids
        if 'agent_' in self.agent_ids:
            # 'agent_1' is the agent to be trained; in order to train that agent towards avoiding collisions with the other agents
            # (heuristic agent and dynamic obstacle), agent_1 must be the last one to act, only then he gets reward_illegal_action
            # in case of colliding agents;
            self.agent_ids.remove('agent_')
            self.agent_ids.append('agent_')
        for agent in self.agent_ids:

           

            agent_obj = self.agent_dict[agent]
            # for each agent in turn: apply action and reward agent

            step_data[agent]["reward"] = float(0.)
            action = step_data[agent]['action']

            if agent != "heuristic_agent" and agent != "random_agent":

                if self._check_object_attached and  self.goal_area != True:
                    pass
                else:
                    # small penalty for each action; this should incentivise to reach the goal as quick as possible;
                    # no penalty if object is attached and being driven by heuristic agent
                    step_data.add_reward(self.reward_each_action, agent) 
           
                    h_agent = self.agent_dict['heuristic_agent']
                    if self.three_grid_object:
                       
                        #penalty in case of wrong attachment of agents in three grid objects
                        if  h_agent.attached and agent_obj.attached:
                            # print(agent_obj.picked_object.loc,"location of three grid object")
                          
                            # print("locations of both attached agents", h_agent.loc, agent_obj.loc)
                            
                            middle_obj_coord = np.median(agent_obj.picked_object.loc, axis=0)
                            # print(middle_obj_coord,"mid coordinates")

                            # print("object grid location", agent_obj.picked_object.loc)
                            # print("agent obj location",agent, agent_obj.loc, "proper location for agent obj shold not be", "x",[middle_obj_coord[0], h_agent.loc[0]],"y",[middle_obj_coord[1], h_agent.loc[1]])
                    
                          

                #in case of three grid object object is only attached if
                #heuristic agent attaches to middle object and policy agent attaches to end objects of theree grid object and opposite sides of heuristic attached agent 
                            
                            
                            if agent_obj.loc[0] not in [middle_obj_coord[0], h_agent.loc[0]] and agent_obj.loc[1] not in [middle_obj_coord[1], h_agent.loc[1]]:
                                step_data.add_reward(0.08, agent)

                                # print("correctly attached reward")
                            
                            else:
                                step_data.add_reward(-0.08, agent)
                                # print("penlaty for wrong attachment")
                    
                                
        

                        
                    
                    #this penalty is use only in case of multliple objects
                    if self.num_objects>1:
                        print("hope this is not printing")
                        #first check whether heursitic agent and policy agents are attached
                        if h_agent.attached and agent_obj.attached:
                            #now if there is any objects attached by two agents
                            object_attached = [x for x in self.objects if x.attachable != True ]
                            #if agent and heuristic agent has already picked an object
                            #but no any object is attached by two agents; i.e both agents attached to two separate objects
                            if len(object_attached) < 1:
                                #add penalty in that case; so it will drop and pick correct object(picked by heuristic)
                                step_data.add_reward(self.reward_wrong_attachment, agent)
                                # print("+++++++Penalty for still sticking to wrong object.++++++++")

            step_data[agent]["infos"]["game_success"] = float(0.0)
            step_data[agent]["infos"]["steps_exceeded"] = float(0.0)
            step_data[agent]["infos"]["first_object_attached"] = float(0.0) #use in goal area to see when first object is attached by both agents

            if action in ["up", "down", "left", "right"]:
                step_data[agent]["infos"]["pick_success"] = float(0.0)
                step_data[agent]["infos"]["pick_failure"] = float(0.0)
                step_data[agent]["infos"]["drop_failure"] = float(0.0)

                success = agent_obj.move(action)
                if success:
                    step_data[agent]["infos"]["move_failure"] = float(0.0)
                else:
                    if agent != "heuristic_agent" and agent != "random_agent":
                        step_data.add_reward(self.reward_illegal_action, agent) # illegal move
                        step_data[agent]["infos"]["move_failure"] = -float(1.0)
                    else:
                        step_data[agent]["infos"]["move_failure"] = float(0.0)
                    # debug output for move failures;  with masked move actions, success=False should actually not occur;
                    """
                    if action == "left":
                        loc = np.array([agent_obj.loc + (-1, 0)])
                        if not agent_obj.picked_object == None:
                            loc_att_obj = np.array([agent_obj.picked_object.loc + (-1, 0)])
                    elif action == "right":
                        loc = np.array([agent_obj.loc + (1, 0)])
                        if not agent_obj.picked_object == None:
                            loc_att_obj = np.array([agent_obj.picked_object.loc + (1, 0)])
                    elif action == "up":
                        loc = np.array([agent_obj.loc + (0, -1)])
                        if not agent_obj.picked_object == None:
                            loc_att_obj = np.array([agent_obj.picked_object.loc + (0, -1)])
                    elif action == "down":
                        loc = np.array([agent_obj.loc + (0, 1)])
                        if not agent_obj.picked_object == None:
                            loc_att_obj = np.array([agent_obj.picked_object.loc + (0, 1)])
                    items_loc = self.engine.items(loc)
                    if agent_obj.picked_object == None:
                        # no object attached
                        print("+++++ MOVE FAILURE: agent: {}, action: {}, items on target location: {}".format(agent, action, items_loc))
                    else:
                        # object attached
                        items_loc_att_obj = self.engine.items(loc_att_obj)
                        if agent_obj in items_loc_att_obj:
                            items_loc_att_obj.remove(agent_obj)
                        if agent_obj.picked_object in items_loc:
                            items_loc.remove(agent_obj.picked_object)
                        print("+++++ MOVE FAILURE with object attached: agent: {}, action: {}, items on target location: {}, "
                            "items on target loc w.r.t. attached object: {}".format(agent, action, items_loc, items_loc_att_obj))
                    """

                #if success and self.agent.attached:
                #    step_data.add_reward(1.0) # reward for successful move with attached object
            elif action == "pick":
                step_data[agent]["infos"]["move_failure"] = float(0.0)
                step_data[agent]["infos"]["drop_failure"] = float(0.0)
                success = agent_obj.pick()
                if success: #no reward for picking object; game success reward is attached later for picking object in the training phase
                    # for debugging
                    # print("+++++++ PICK SUCCESS +++++++")
                    
                    if agent != "heuristic_agent" and agent != "random_agent":
                        #in case of multiple objects
                        if self.goal_area and self.num_objects>1:
                         #in case of goal area if policy agent picks object add some reward
                            step_data.add_reward(self.reward_succesful_pick, agent)
                            print("policy agent has successfuly picked an object",)
                            #penalty if policy agent pick wrong object(i.e object not already picked by heuristic agent)
                            h_agent = self.agent_dict['heuristic_agent']
                            #first check whether heursitic agent is attached
                            if h_agent.attached:
                                #now if there is any objects attached by two agents
                                object_attached = [x for x in self.objects if x.attachable != True ]
                                #if agent picks and object and heuristic agent has already picked an object
                                #but no any object is attached by two agents; i.e both agents attached to two separate objects
                                if len(object_attached) < 1:
                                    # print("----- Adding penalty as different agent attached to different object------")
                                    #add penalty in that case; i.e wrong pick
                                 
                                    step_data.add_reward(self.reward_wrong_pick, agent)



                        step_data[agent]["infos"]["pick_success"] = float(1.0)
                        step_data[agent]["infos"]["pick_failure"] = float(0.0)
                else:
                    if agent != "heuristic_agent" and agent != "random_agent":
                        step_data.add_reward(self.reward_illegal_action, agent) # illegal pick
                        step_data[agent]["infos"]["pick_success"] = float(0.0)
                        step_data[agent]["infos"]["pick_failure"] = -float(1.0)
                    else:
                        step_data[agent]["infos"]["pick_success"] = float(0.0)
                        step_data[agent]["infos"]["pick_failure"] = float(0.0)
            elif action == "drop":
                step_data[agent]["infos"]["move_failure"] = float(0.0)
                step_data[agent]["infos"]["pick_failure"] = float(0.0)
                step_data[agent]["infos"]["pick_success"] = float(0.0)
                success = agent_obj.drop()
                if success:
                    # step_data.add_reward(0.1, agent)
                    step_data[agent]["infos"]["drop_failure"] = float(0.0)
                    #if goal area and object is dropped add some reward to the agent
                    # if self.goal_area:
                    #     step_data.add_reward(0.3, agent)
                else:
                    if agent != "heuristic_agent" and agent != "random_agent":
                        step_data.add_reward(self.reward_illegal_action, agent) # illegal drop
                        step_data[agent]["infos"]["drop_failure"] = -float(1.0)
                    else:
                        step_data[agent]["infos"]["drop_failure"] = float(0.0)


            elif action =="do_nothing":
                step_data[agent]["infos"]["move_failure"] = float(0.0)
                step_data[agent]["infos"]["pick_failure"] = float(0.0)
                step_data[agent]["infos"]["pick_success"] = float(0.0)
                step_data[agent]["infos"]["drop_failure"] = float(0.0)

        # Check game end

        #if goal area True use goal area goal success condition

        if self.goal_area == True:
           
            if self.mode != 'evaluation':
                if self.three_grid_object:
                    
                    if self._check_objects_attached():
                        print("+++++++++++ game success; object attached correctly by all agents ++++++++++++")
                        for agent in self.agent_ids:
                            if agent != "heuristic_agent" and agent != "random_agent":
                                step_data.add_reward(self.reward_game_success, agent) # game success reward for agent getting attached
                                step_data[agent]["infos"]["game_success"] = float(1.0)

                            step_data[agent]['terminated'] = True
                        step_data.terminate_game({"goal_reached": True})
                      
                    elif step_data.steps_exceeded(self.max_steps):
                        for agent in self.agent_ids:
                            if agent != "heuristic_agent" and agent != "random_agent":
                                # step_data.add_reward(-1.0, agent)
                                step_data[agent]["infos"]["steps_exceeded"] = -float(1.0)
                            step_data[agent]['terminated'] = True
                        step_data.terminate_game({"goal_reached": False})

                else:

                    


                    if self. _check_goal_goal_area():
                        print("+++++++++++ game success ++++++++++++")
                        for agent in self.agent_ids:
                            if agent != "heuristic_agent" and agent != "random_agent":
                                step_data.add_reward(self.reward_game_success, agent) # game success reward for agent getting attached
                                step_data[agent]["infos"]["game_success"] = float(1.0)
                                
                            
                            step_data[agent]['terminated'] = True
                        step_data.terminate_game({"goal_reached": True})
                    elif step_data.steps_exceeded(self.max_steps):
                        for agent in self.agent_ids:
                            if agent != "heuristic_agent" and agent != "random_agent":
                                # step_data.add_reward(-1.0, agent)
                                step_data[agent]["infos"]["steps_exceeded"] = -float(1.0)
                            step_data[agent]['terminated'] = True
                        step_data.terminate_game({"goal_reached": False})
            else: #mode is evaluation so game ends with heuristic agent driving composite object to goal and dropping it
                
                

                
                if self._check_goal_area_eval(): # various _check_goal* methods are implemented
                    #print("+++++++++++ game success ++++++++++++")
                    for agent in self.agent_ids:
                        #beacuse game success right now is driven by heuristic_policy
                        #if agent != "heuristic_agent":

                            #step_data.add_reward(self.reward_game_success, agent)
                        step_data[agent]["infos"]["game_success"] = float(1.0)
                        step_data[agent]['terminated'] = True
                    step_data.terminate_game({"goal_reached": True})


                else:

                    if self._check_object_attached() : #check whether an object is attached by two agents
                        print("+++++ one object attached by both agents++++++")
                        for agent in self.agent_ids:
                            step_data[agent]["infos"]["first_object_attached"] = float(1.0)

                

                        

                    if step_data.steps_exceeded(self.max_steps):
                        for agent in self.agent_ids:
                            # step_data.add_reward(-1.0, agent)
                            step_data[agent]["infos"]["steps_exceeded"] = float(1.0)
                            step_data[agent]['terminated'] = True
                        step_data.terminate_game({"goal_reached": False})


        else:
           


            #for training mode game ends when policy agent picks object(attached to the object)
            if self.mode != 'evaluation':
                if self._check_goal_collab_transport(): # various _check_goal* methods are implemented
                    #print("+++++++++++ game success ++++++++++++")
                    for agent in self.agent_ids:
                        if agent != "heuristic_agent" and agent != "random_agent":
                            step_data.add_reward(self.reward_game_success, agent) # game success reward for agent getting attached
                            step_data[agent]["infos"]["game_success"] = float(1.0)
                        step_data[agent]['terminated'] = True
                    step_data.terminate_game({"goal_reached": True})
                elif step_data.steps_exceeded(self.max_steps):
                    for agent in self.agent_ids:
                        if agent != "heuristic_agent" and agent != "random_agent":
                            # step_data.add_reward(-1.0, agent)
                            step_data[agent]["infos"]["steps_exceeded"] = -float(1.0)
                        step_data[agent]['terminated'] = True
                    step_data.terminate_game({"goal_reached": False})

            else: #mode is evaluation so game ends with heuristic agent driving composite object to goal and dropping it

                if self._check_goal(): # various _check_goal* methods are implemented
                    print("+++++++++++ game success ++++++++++++")
                    for agent in self.agent_ids:
                        #beacuse game success right now is driven by heuristic_policy
                        #if agent != "heuristic_agent":

                            #step_data.add_reward(self.reward_game_success, agent)
                        step_data[agent]["infos"]["game_success"] = float(1.0)
                        step_data[agent]['terminated'] = True
                    step_data.terminate_game({"goal_reached": True})
                elif step_data.steps_exceeded(self.max_steps):
                    for agent in self.agent_ids:
                        # step_data.add_reward(-1.0, agent)
                        step_data[agent]["infos"]["steps_exceeded"] = -float(1.0)
                        step_data[agent]['terminated'] = True
                    step_data.terminate_game({"goal_reached": False})

        return step_data

    def _check_goal(self) -> bool:
        goal_locs = np.stack([g.loc for g in self.goals], 0)
        at_goals = self.engine.items(locs=goal_locs)
        dropped_objs = [i for i in at_goals if i.kind == items.ItemKind.OBJECT and i.dropped]

        if len(dropped_objs) == self.num_objects:
           print("++++++++ ALL OBJECTS ON GOALS ++++++++")


        #end game if object is attached to both agents
        object_attached = [i for i in self.objects if len(i.carriers) == 2]

        # if len(object_attached) == len(self.objects):
        #     print("both agent attached to the object", [i.carriers for i in object_attached])
        #     print([self.agent_dict[agent].attached for agent in self.agent_ids])
        #     print('-----------------------')

        return len(dropped_objs) == self.num_objects

    def _check_goal_collab_transport(self) -> bool:
        """
        game is successful if agent gets attached to the object

        """
        attached_agents = []
        policy_agents = []
        for agent_id in self.agent_ids:

            if agent_id != "random_agent" and agent_id != "heuristic_agent":
                
                agent_obj = self.agent_dict[agent_id]

                policy_agents.append(agent_obj)
                
                if agent_obj.attached:
                    attached_agents.append(agent_obj)

        if len(attached_agents) == len(policy_agents):
                #print("++++++++ GAME SUCCESSFUL; POLICY AGENT ATTACHED ++++++++")
                return True
        else:
            return False


    def _check_goal_goal_area(self) -> bool:

        """
        game is successful if one object is on goal area and another one is attached by policy agent or if both objects are in goal areas

        """

        goal_locs = np.stack([g.loc for g in self.goals], 0)
        
        at_goals = [self.engine.items(locs=goal_loc) for goal_loc in goal_locs]
 

        dropped_objs = []

        for at_goal in at_goals:

            for i in at_goal:
                if i.kind == items.ItemKind.OBJECT and i.dropped:
                    dropped_objs.append(i)
        
        
          




     

        # if (len(dropped_objs) == (self.num_objects)):

            
        #     print(dropped_objs, "dropped_objectsd", len(dropped_objs))

        #     print(" Both objects are already on goal, objects generated at goal issue ?")

        #     return True

        if len( dropped_objs) > 0: #at least one not attached object in goal
            # agents_attached = []
            for agent_id in self.agent_ids:

                if agent_id != "random_agent" and agent_id != "heuristic_agent":
                    agent_obj = self.agent_dict[agent_id]

                    print("object coorectly dropped", dropped_objs, agent_obj.attached)
                    
                    if agent_obj.attached: #if any policy agent is attached
                        # agents_attached.append(agent_obj)
                        # if len(agents_attached)>1:
                        return True
            return False
        else:
            return False

    def _check_goal_area_eval(self) -> bool:


        """
        game is successful if one object is on goal area and another one is attached by policy agent

        """


        goal_locs = np.stack([g.loc for g in self.goals], 0)
        at_goals = self.engine.items(locs=goal_locs)
        dropped_objs = [i for i in at_goals if i.kind == items.ItemKind.OBJECT and i.dropped] #not attached objects in goal

        if len( dropped_objs) > 0: #at least one not attached object in goal
            agents_attached = []
            for agent_id in self.agent_ids:

                # if agent_id != "random_agent" and agent_id != "heuristic_agent":
                agent_obj = self.agent_dict[agent_id]
                    
                if agent_obj.attached: #if any policy agent is attached
                        agents_attached.append(agent_obj)
                        if len(agents_attached)>1:
                            return True
            return False
        else:
            return False


    
    
    def _check_object_attached(self) -> bool:
        
        '''
        to check whether any of the object is attached by both agents

        '''

        
        #object not being attachable means it is attached by both agents
        attached_objs = [i for i in self.objects if i.attachable != True ]

        if len(attached_objs) > 0:
            return True
        else:
            return False

    def _check_objects_attached(self) -> bool:
        
        '''
        to check whether any of the object is attached by both agents

        '''

        
        #object not being attachable means it is attached by both agents
        attached_objs = [i for i in self.objects if i.attachable != True ]

        if len(attached_objs) == len(self.objects):
            return True
        else:
            return False




    def _check_goal_pick(self) -> bool:
        """
        game is successful if one object has been picked
        """
        if self.agent.attached:
            # for debugging
            print("++++++++ PICK SUCCESSFUL ++++++++")
            return True
        else:
            return False

    def _check_goal_find_goal(self) -> bool:
        """
        copied from find goal game; the game is successful if a goal has been found, i.e. the agent moves on a goal
        """
        the_items = self.engine.items(locs=self.agent.loc)
        the_items = [i for i in the_items if i.kind == items.ItemKind.GOAL]
        return len(the_items) > 0

    def _check_goal_movewithobject(self) -> bool:
        """
        game is successful if the object has been picked and the agent has made a number of moves with the attached object;
        """
        if self.agent.attached and self.moves_since_pick == 2:
            print("++++++++ PICK SUCCESSFUL, MOVED TWICE ++++++++")
            return True
        else:
            return False

    def _check_goal_movewithobject_until_goaldistance(self, goal_distance) -> bool:
        """
        game success if the agent has moved with the object until a given distance to the goal;
        """
        distance_to_goal = self.move_distance(self.agent, self.goals[0])
        if self.agent.attached and distance_to_goal == goal_distance:
            print("++++++++++++++++ MOVED WITH OBJECT; DISTANCE TO GOAL: {} +++++++++++++".format(goal_distance))
            return True
        else:
            return False

    def render(self, image_observation:bool=True) -> np.ndarray:
        return defaults.render(self.engine, image_observation=image_observation)

    def is_adjacent_to_impassable(self, direction: str, item: ItemBase) -> (bool, ItemBase):
        """
        Returns True if an impassable item is adjacent to param "item" in the direction of param "direction"
        """
        if direction == "left":
            loc = np.array([item.loc + (-1, 0)])
        elif direction == "right":
            loc = np.array([item.loc + (1, 0)])
        elif direction == "up":
            loc = np.array([item.loc + (0, -1)])
        elif direction == "down":
            loc = np.array([item.loc + (0, 1)])

        items = self.engine.items(loc)
        contains_impassable = False
        impassable_item = None
        for item in items:
            if item.impassable:
                contains_impassable = True
                impassable_item = item
                break
        return contains_impassable, impassable_item

    def is_adjacent_to(self, direction: str, item: ItemBase, other_item: ItemBase):
        """
        Returns True if other_item is adjacent to item in the direction of param direction
        """
        if direction == "left":
            loc = np.array([item.loc + (-1, 0)])
        elif direction == "right":
            loc = np.array([item.loc + (1, 0)])
        elif direction == "up":
            loc = np.array([item.loc + (0, -1)])
        elif direction == "down":
            loc = np.array([item.loc + (0, 1)])

        items = self.engine.items(loc)
        contains_item = False
        for it in items:
            if it == item:
                contains_item = True
                break
        return contains_item

def play_warehousegame():
    import matplotlib.pyplot as plt

    print("Use w,a,s,d to move and q to toggle pick/drop")

    game = WarehouseGame_1(max_steps=100, num_objects=1, num_goals=1, num_obstacles=1)
    game.reset()

    fig, ax = plt.subplots()
    mpl_img = ax.imshow(game.render())

    def on_press(event):
        if event.key == "w":
            game.step({game.agent: "up"})
        elif event.key == "s":
            game.step({game.agent: "down"})
        elif event.key == "a":
            game.step({game.agent: "left"})
        elif event.key == "d":
            game.step({game.agent: "right"})
        elif event.key == "q":
            game.step({game.agent: "drop"})
        elif event.key == "e":
            game.step({game.agent: "pick"})
        if game.done:
            print(game.reward_history)
            game.reset()
        mpl_img.set_data(game.render())
        fig.canvas.draw()

    fig.canvas.mpl_connect("key_press_event", on_press)
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    plt.show()


if __name__ == "__main__":
    play_warehousegame()