import abc
import copy
from typing import Any, Dict

import numpy as np
from gym.utils import seeding

from .items import ItemBase, ItemKind

ActionDict = Dict[ItemBase, Any]


class StepData(dict):
    def __init__(self, actions: ActionDict, num_steps: int) -> None: # actions is a dict with agent objects as keys and action strings (e.g. 'right') as values
        super().__init__()
        self["reward"] = 0.0
        self["terminate"] = False
        self["actions"] = actions
        self["num_steps"] = num_steps
        self["infos"] = {}

    def add_reward(self, r: float):
        """Add reward for the current step."""
        self["reward"] += r

    def terminate_game(self, terminal_info: Any = None):
        """Terminate the game after the current step."""
        self["terminate"] = True
        self["terminal_info"] = terminal_info

    @property
    def actions(self) -> ActionDict:
        return self["actions"]

    @property
    def reward(self) -> float:
        return self["reward"]

    @property
    def infos(self) -> dict:
        return self["infos"]

    @property
    def terminated(self) -> bool:
        return self["terminate"]

    def steps_exceeded(self, max_steps: int) -> bool:
        return self["num_steps"] >= max_steps

    @property
    def terminal_info(self) -> Any:
        return self.get("terminal_info", None)


class StepData_multiagent(dict):
    def __init__(self, actions: ActionDict, num_steps: int, agent_dict, active_agent_ids = ['agent_1', 'agent_2']) -> None:
        """
        actions is a dict with agent objects as keys and action strings (e.g. 'right') as values;
        agent_dict maps agent ids (keys) to agent objects
        """
        super().__init__()
        self.agent_dict = agent_dict
        self.active_agent_ids = active_agent_ids
        self["num_steps"] = num_steps
        for agent in active_agent_ids:
            self[agent] = {}
            self[agent]["reward"] = 0.0
            self[agent]["terminated"] = False
            self[agent]["action"] = actions[self.agent_dict[agent]]
            self[agent]["infos"] = {}

    def add_reward(self, r: float, agent):
        """Add reward for the current step."""
        self[agent]["reward"] += r

    def terminate_game(self, terminal_info: Any = None):
        """Terminate the game after the current step."""
        for agent in self.active_agent_ids:
            self[agent]["terminated"] = True

        self["terminal_info"] = terminal_info

    @property
    def actions(self, agent) -> ActionDict:
        return self[agent]["actions"]

    @property
    def reward(self, agent) -> float:
        return self[agent]["reward"]

    @property
    def infos(self, agent) -> dict:
        return self[agent]["infos"]

    @property
    def terminated(self,agent) -> bool:
        return self[agent]["terminate"]

    def steps_exceeded(self, max_steps: int) -> bool:
        return self["num_steps"] >= max_steps

    @property
    def terminal_info(self) -> Any:
        return self.get("terminal_info", None)


class GameBase(abc.ABC):
    """Base class for games."""

    def __init__(self, seed: int = None) -> None:
        self.rng = self.seed(seed)

    def reset(self, **kwargs):
        """Reset the game. Ends current episode"""
        self.num_steps = 0
        self.reward_history = []
        self.action_history = []
        self.done = False
        self.terminal_info = None
        if "deterministic_game" in kwargs and kwargs["deterministic_game"] == True:
            self._create_game_1() # item locations are specified (not randomised as in method _create_game())
        else:
            self._create_game()


    def step(self, actions: Dict[ItemBase, Any]) -> bool:
        """Step game from actions and return completion status."""
        step_data = self._pre_step(actions)
        self.step_data = self._step(step_data)
        self._post_step(step_data)
        return self.done

    def seed(self, seed: int = None) -> np.random.Generator:
        """Seed the random number generator"""
        rng, _ = seeding.np_random(seed)
        return rng

    def _pre_step(self, actions: ActionDict) -> StepData:
        # if self.done:
        #     raise ValueError("Game ended, call game.reset first")
        self.num_steps += 1
        sd = StepData(actions, self.num_steps)
        return sd

    def _post_step(self, step_data: StepData):
        self.action_history.append(step_data.actions)
        self.reward_history.append(step_data.reward)
        self.done = step_data.terminated
        self.terminal_info = step_data.get("terminal_info", None)

    def move_distance(self, item_1: ItemBase, item_2: ItemBase):
        """
        returns the minimal number of moves (left, right, up, down) between two items in the gridworld;
        """
        dist = abs(item_1.loc[0] - item_2.loc[0]) + abs(item_1.loc[1] - item_2.loc[1])
        # for debugging
        #print("++++++++++++++++ DEBUG OUTPUT: move distance between {} and {} is {} ++++++++++++++++++".format(item_1.loc, item_2.loc, dist))
        return dist

    @property
    def reward(self):
        """Returns the last-step reward"""
        return self.reward_history[-1]

    @abc.abstractmethod
    def _create_game(self):
        ...

    @abc.abstractmethod
    def _step(self, step_data: StepData):
        ...

    @abc.abstractmethod
    def render(self) -> np.ndarray:
        ...


class GameBase_multiagent(GameBase):
    """exends GameBase with multi-agent compatibility"""

    def __init__(self, seed: int = None, agent_ids: list = ['agent_1', 'agent_2'], collaborative_transport: bool = True) -> None:
        self.rng = self.seed(seed)
        self.agent_ids = agent_ids
        self.active_agent_ids = agent_ids # agents may terminate individually
        self.agent_dict = {} # maps agent ids (keys) to agent objects; agent_dict is created when calling _create_game* in child class
        self.reward_history = {} # keys: agent id, values: reward history
        self.action_history = {} # keys: agent id, values: action history
        self.collaborative_transport = collaborative_transport # if true, an object requires two attached agents for transport, otherwise one agent can do the transport
        #self.terminated = {}
        #for agent in agent_ids:
        #    self.terminated[agent] = False

    #@property
    #def get_terminated(self):
    #    return self.terminated

    def reset(self, **kwargs):
        """Reset the game. Ends current episode"""
        self.num_steps = 0
        if "goal_area" in kwargs:
            goal_area = kwargs["goal_area"]
        else:
            goal_area = False

        for agent in self.agent_ids:
            self.reward_history[agent] = []
            self.action_history[agent] = []
        self.done = False
        self.active_agent_ids = self.agent_ids
        self.terminal_info = None
        if "deterministic_game" in kwargs and kwargs["deterministic_game"] == True:
            self._create_game_deterministic(goal_area=goal_area)  # item locations are specified (not randomised as in method _create_game())
        else:
            self._create_game(goal_area=goal_area)

    def step(self, actions: Dict[ItemBase, Any]) -> bool: # actions is a dict with agent objects as keys and action strings as values
        """Step game from actions and return completion status."""
        step_data = self._pre_step(actions)
        self.step_data = self._step(step_data) # to be implemented by child class
        self._post_step(step_data)
        return self.done

    def seed(self, seed: int = None) -> np.random.Generator:
        """Seed the random number generator"""
        rng, _ = seeding.np_random(seed)
        return rng

    def _pre_step(self, actions: ActionDict) -> StepData:
        # if self.done:
        #     raise ValueError("Game ended, call game.reset first")
        self.num_steps += 1
        sd = StepData_multiagent(actions, self.num_steps, self.agent_dict, active_agent_ids=self.active_agent_ids)
        return sd

    def _post_step(self, step_data: StepData):
        for agent in self.active_agent_ids:
            self.action_history[agent].append(step_data[agent]['action'])
            self.reward_history[agent].append(step_data[agent]['reward'])
        done = True
        active_agent_ids_copy = copy.copy(self.active_agent_ids)
        for agent in self.active_agent_ids:
            if step_data[agent]['terminated'] == True: # the agent terminated in the current time step
                active_agent_ids_copy.remove(agent)
            else:
                done = False
        self.active_agent_ids = active_agent_ids_copy
        self.done = done
        self.terminal_info = step_data.get("terminal_info", None)

    def move_distance(self, item_1: ItemBase, item_2: ItemBase):
        """
        returns the minimal number of moves (left, right, up, down) between two items in the gridworld;
        """
        dist = abs(item_1.loc[0] - item_2.loc[0]) + abs(item_1.loc[1] - item_2.loc[1])
        # for debugging
        # print("++++++++++++++++ DEBUG OUTPUT: move distance between {} and {} is {} ++++++++++++++++++".format(item_1.loc, item_2.loc, dist))
        return dist

    @property
    def reward(self):
        """Returns the last-step reward"""
        return self.reward_history[-1]

    @abc.abstractmethod
    def _create_game(self):
        ...

    @abc.abstractmethod
    def _step(self, step_data: StepData):
        ...

    @abc.abstractmethod
    def render(self) -> np.ndarray:
        ...


