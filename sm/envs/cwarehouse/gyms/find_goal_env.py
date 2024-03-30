from typing import Tuple
import gym
import numpy as np
from gym import spaces
from gym.envs.registration import register

from ..gridworld import games
from ..gridworld.examples.find_goal_game import FindGoalGame, FindGoalGame_1
# from ray.rllib.env.multi_agent_env import MultiAgentEnv

from PIL import Image


Actions = ["left", "right", "up", "down"]
Observation = np.ndarray


class FindGoalEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        grid_shape=(9, 9),
        obs_shape=(96, 96),
        seed: int = None,
        max_steps: int = 50,
        random_initial_locs: bool = True,
    ):
        agent_loc = (-1, -1) if random_initial_locs else (2, 2)
        goal_loc = (-1, -1)  # if random_initial_locs else (5, 6)
        self.game = FindGoalGame(
            shape=grid_shape,
            max_steps=max_steps,
            seed=seed,
            initial_agent_loc=agent_loc,
            initial_goal_loc=goal_loc,
        )
        self.viewer = None
        # # Action space is discrete
        self.actions = Actions
        self.action_space = spaces.Discrete(len(self.actions))
        # Observation space is nd-box
        self.obs_size = obs_shape[::-1]
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=obs_shape + (3,),
            dtype="uint8",
        )
        # Reward range
        self.reward_range = (-1, 1)
        # Seed
        self.reset()

    def step(self, action: int) -> Tuple[Observation, float, bool, dict]:
        self.game.step({self.game.agent: Actions[action]})
        return (
            self._gen_obs(),
            self.game.reward,
            self.game.done,
            {"terminal_info": self.game.terminal_info},
        )

    def reset(self) -> Observation:
        self.game.reset()
        return self._gen_obs()

    def render(self, mode="human"):
        img = self.game.render()
        img = Image.fromarray(img, "RGB").resize(self.obs_size, resample=0)
        img = np.array(img)
        if mode == "human":
            if self.viewer is None:
                from gym.envs.classic_control import rendering

                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
        else:
            return img

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def seed(self, seed):
        self.game.seed(seed)

    def _gen_obs(self) -> dict:
        obs = {
            "image": self.render(mode="rgb_array"),
        }
        return obs


# class FindGoalEnv_1(MultiAgentEnv):
#     """
#     modification of FindGoalEnv: aligned with RLlib requirements for MARL environments
#     """

#     metadata = {"render.modes": ["human", "rgb_array"]}

#     def __init__(self, agent_ids, shape=(9, 9), seed=None, max_steps=100):
#         assert len(agent_ids) == 1
#         self.time_step = 0
#         self.agents = agent_ids
#         # just take one agent for the find goal game
#         self.game = FindGoalGame_1(
#             self.agents[0], shape=shape, max_steps=max_steps, seed=seed
#         )
#         self.viewer = None
#         # # Action space is discrete
#         self.actions = Actions
#         self.action_space = spaces.Discrete(len(self.actions))
#         # Observation space is nd-box
#         oimg = spaces.Box(
#             low=0,
#             high=255,
#             shape=(
#                 shape[0] * 5,
#                 shape[1] * 5,
#                 3,
#             ),
#             dtype="uint8",
#         )
#         # self.observation_space = spaces.Dict({"image": oimg}) # original code cheind
#         self.observation_space = oimg
#         # Reward range
#         self.reward_range = (-1, 1)
#         self.reset()

#     # param action_dict maps agent IDs (keys) to actions
#     def step(self, action_dict):
#         rewards = {}
#         dones = {}
#         infos = {}
#         self.time_step += 1
#         action_dict_game = {}  # maps agent objects to actions
#         assert len(self.agents) == 1  # the following code works only for one agent
#         for agent in self.agents:
#             agent_obj = self.game.agent_dict[agent]
#             action_dict_game[agent_obj] = Actions[action_dict[agent]]
#         # step_data = games.StepData(action_dict_game, self.time_step)
#         self.game.step(action_dict_game)
#         dones["__all__"] = self.game.done
#         for agent in self.agents:
#             rewards[agent] = self.game.reward_history[-1]
#             dones[agent] = self.game.done
#             infos[agent] = {}
#         return self._gen_obs(), rewards, dones, infos

#     def reset(self) -> dict:
#         self.time_step = 0
#         self.game.reset()
#         return self._gen_obs()

#     def render(self, mode="human"):
#         img = self.game.render()
#         if mode == "human":
#             if self.viewer is None:
#                 from gym.envs.classic_control import rendering

#                 self.viewer = rendering.SimpleImageViewer()
#             self.viewer.imshow(img)
#         else:
#             return img

#     def close(self):
#         if self.viewer is not None:
#             self.viewer.close()

#     def seed(self, seed):
#         self.game.seed(seed)

#     def _gen_obs(self) -> dict:
#         """
#         *** original code ***
#         obs = {
#             "image": self.render(mode="rgb_array"),
#         }
#         """
#         obs = self.render(mode="rgb_array")
#         observations = {}
#         for agent in self.agents:
#             observations[agent] = obs
#         return observations


