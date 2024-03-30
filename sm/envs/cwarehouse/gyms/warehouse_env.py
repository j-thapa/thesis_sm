
# import gym
# import numpy as np
# from gym import spaces
# from PIL import Image

# from gym import ObservationWrapper
# import tensorflow as tf


# import concert
# from concert.gridworld.items import ItemKind, ItemBase
# from ..gridworld.items import ItemKind_onehot
# from concert.gyms.examples.utils import handle_not_element

# import matplotlib.pyplot as plt


# from concert.gyms.heuristic_agent import HeuristicAgent, RandomAgent

# from ..gridworld.examples.warehouse_game import WarehouseGame, WarehouseGame_1, WarehouseGame_2, WarehouseGame_3

# Actions = ["left", "right", "up", "down", "pick", "drop"]
# Actions_extended = ["left", "right", "up", "down", "pick", "drop", "do_nothing"]
# Actions_reduced = ["left", "right", "up", "down", "pick"]



# class WarehouseEnv(gym.Env):
#     metadata = {"render.modes": ["human", "rgb_array"]}

#     def __init__(self, shape=(9, 9), num_objects: int = 1, seed=None, max_steps=100):
#         self.game = WarehouseGame(
#             shape=shape,
#             max_steps=max_steps,
#             seed=seed,
#             num_objects=num_objects,
#             num_goals=num_objects,
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
#         self.observation_space = spaces.Dict({"image": oimg})
#         # Reward range
#         self.reward_range = (-1, 1)
#         # Seed
#         self.reset()

#     def step(self, action: int):
#         self.game.step({self.game.agent: Actions[action]})
#         return self._gen_obs(), self.game.reward, self.game.done, {}

#     def reset(self) -> dict:
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
#         obs = {
#             "image": self.render(mode="rgb_array"),
#         }
#         return obs

# def rgb_to_grayscale(image):
#     """
#     not used
#     """
#     with Image.open(image) as im:
#         grayscale_image = im.convert('L')
#     return grayscale_image

# class GrayscaleEnv(ObservationWrapper):
#     """
#     not used;

#     Wrapper for gridworld environments, converting original RGB images to grayscale images.
#     """
#     # FIXME make multi-agent compatible; in the multi-agent case the environment returns a dictionary (keys: agent_ids, values: images), not an image
#     def __init__(self, env, f):
#         super().__init__(env)
#         assert callable(f)
#         self.f = f

#     def observation(self, observation):
#         print("++++++++ ENV WRAPPER: OBSERVATION CONVERTED TO GRAYSCALE +++++++++ ")
#         return self.f(observation)


#     # Override `observation` to custom process the original observation
#     # coming from the env.
#     #def observation(self, observation):
#     #    with Image.open(observation) as obs:
#     #        grayscale_obs = obs.convert('L')
#     #        print("++++++++ ENV WRAPPER: OBSERVATION CONVERTED TO GRAYSCALE +++++++++ ")
#     #    return grayscale_obs

# def flattenVectorObs(observations: dict):
#     """
#     not used;

#     Takes a gridworld's dict of vector observations per agent_id (3-dimensional: 2-dim grid with one-hot encoded observations per grid cell)
#     and flattens the vector observations (3-dim => 1-dim)
#     """
#     for agent_id in observations.keys():
#         assert isinstance(observations[agent_id], np.ndarray)
#         print("in flattenVectorObs(): observation dtype is {}".format(observations[agent_id].dtype))
#         print("in flattenVectorObs(): observation for agent {} is: ".format(agent_id))
#         print(observations[agent_id])

#         observations[agent_id] = observations[agent_id].flatten()

#         print("after flattening, dtype is {}".format(observations[agent_id].dtype))
#         print("observation is: ")
#         print(observations[agent_id])


#     return observations

# class FlattenedVectorObsEnv(ObservationWrapper):
#     """
#     not used
#     """
#     def __init__(self, env, f):
#         super().__init__(env)
#         assert callable(f)
#         self.f = f

#     def observation(self, observation):
#         print("calling FlattenedVectorObsEnv.observation() on observation:")
#         print(observation)
#         result = self.f(observation)
#         return result

# """
# example code for transforming observations via a wrapper, from https://github.com/openai/gym/blob/master/gym/wrappers/transform_observation.py


# from gym import ObservationWrapper


# class TransformObservation(ObservationWrapper):
#     #Transform the observation via an arbitrary function.
#     #Example::
#     #    >>> importgym
#     #    >>> env =gym.make('CartPole-v1')
#     #    >>> env = TransformObservation(env, lambda obs: obs + 0.1*np.random.randn(*obs.shape))
#     #    >>> env.reset()
#     #    array([-0.08319338,  0.04635121, -0.07394746,  0.20877492])
#     #Args:
#     #    env (Env): environment
#     #    f (callable): a function that transforms the observation
    

#     def __init__(self, env, f):
#         super().__init__(env)
#         assert callable(f)
#         self.f = f

#     def observation(self, observation):
#         return self.f(observation)
# """



# class WarehouseEnv_1(MultiAgentEnv):
#     """
#         modification of WarehouseEnv: single-agent environment, but aligned with RLlib requirements for MARL environments
#     """
#     metadata = {"render.modes": ["human", "rgb_array"]}

#     def __init__(self, agent_ids: [str], shape=(10, 10), num_objects=1, seed=None, max_steps=200, deterministic_game=False,
#                  image_observation=False, action_masking=True, num_obstacles=0):
#         """
#         @param
#             action_masking: if True, pick & drop actions are completely masked out; if False, no action masking is applied;
#         """
#         super().__init__()
#         assert len(agent_ids) == 1
#         self.timestep = 0
#         self.agents = agent_ids
#         self.game = WarehouseGame_1(
#             agent_id=self.agents[0],
#             shape=shape,
#             max_steps=max_steps,
#             seed=seed,
#             num_objects=num_objects,
#             num_goals=num_objects,
#             num_obstacles=num_obstacles,
#         )
#         self.deterministic_game:bool = deterministic_game
#         self.image_observation:bool = image_observation
#         self.action_masking = action_masking
#         self.viewer = None
#         # # Action space is discrete
#         self.actions = Actions
#         self.action_space = spaces.Discrete(len(self.actions))
#         if self.image_observation:
#             # image observation; observation space is nd-box
#             observation_space = spaces.Box(
#                 low=0,
#                 high=255,
#                 shape=(
#                     shape[0]*4,
#                     shape[1]*4,
#                     3,
#                 ),
#                 dtype="uint8",
#             )
#             if self.action_masking:
#                 observation_space = spaces.Dict(
#                     {
#                         "action_mask": spaces.Box(0.0, 1.0, shape=(self.action_space.n,)),
#                         "observations": observation_space,
#                     }
#                 )
#         else:
#             # vector observation: a flattened 3-dim state with one-hot encoded items (=> vector) on grid locations (=> matrix);
#             observation_space = spaces.MultiBinary(shape[0] * shape[1] * ItemKind_onehot.num_itemkind)
#             #observation_space = spaces.Box(low=0.,high=1.,shape=(shape[0] ,shape[1] ,ItemKind_onehot.num_itemkind),dtype="float")
#             if self.action_masking:
#                 observation_space = spaces.Dict(
#                     {
#                         "action_mask": spaces.Box(0.0, 1.0, shape=(self.action_space.n,)),
#                         "observations": observation_space,
#                     }
#                 )

#         # self.observation_space = spaces.Dict({"image": oimg}) # orig code cheind
#         self.observation_space = observation_space
#         if self.action_masking:
#             print("+++++ Environment created, observation space is {}".format(self.observation_space["observations"]))
#         else:
#             print("+++++ Environment created, observation space is {}".format(self.observation_space))
#         # Reward range
#         self.reward_range = (-1, 1) # TODO this is not used
#         # Seed
#         self.reset()

#     def step(self, action_dict):
#         #observations = {}
#         rewards = {}
#         dones = {}
#         infos = {}
#         self.timestep += 1
#         action_dict_game = {} # maps agent objects to actions
#         assert len(self.agents) == 1 # the following code only works for one agent
#         for agent in self.agents:
#             agent_obj = self.game.agent_dict[agent]
#             action_dict_game[agent_obj] = Actions[action_dict[agent]]
#         self.game.step(action_dict_game)
#         dones["__all__"] = self.game.done
#         for agent in self.agents:
#             rewards[agent] = self.game.reward_history[-1]
#             dones[agent] = self.game.done
#             infos[agent] = self.game.step_data.infos
#         #print("++++++ STEP returning observations ")
#         #print(self._gen_obs())
#         return self._gen_obs(image_observation=self.image_observation), rewards, dones, infos

#     def reset(self) -> dict:
#         self.timestep = 0
#         if self.deterministic_game == True:
#             self.game.reset(deterministic_game=True)
#         else:
#             self.game.reset()
#         #print("+++++ RESET returning:")
#         #print(self._gen_obs())
#         observation = self._gen_obs(image_observation=self.image_observation)
#         return observation

#     def render(self, mode="human", image_observation:bool=True):
#         state = self.game.render(image_observation=image_observation) # state can be an image or a matrix
#         if mode == "human" and image_observation:
#             if self.viewer is None:
#                 from gym.envs.classic_control import rendering

#                 self.viewer = rendering.SimpleImageViewer()
#             self.viewer.imshow(state)
#         else:
#             return state

#     def close(self):
#         if self.viewer is not None:
#             self.viewer.close()

#     def seed(self, seed):
#         self.game.seed(seed)

#     def _gen_obs(self, image_observation: bool=True) -> dict:
#         """
#         original code
#         obs = {
#             "image": self.render(mode="rgb_array"),
#         }
#         """
#         obs = self.render(mode="rgb_array", image_observation=image_observation)

#         observations = {}
#         for agent in self.agents:
#             agent_obj: concert.gridworld.items.PickAgent = self.game.agent_dict[agent]
#             if self.action_masking:
#                 action_mask = self.gen_action_mask(agent_obj)
#                 obs = {
#                     "observations": obs,
#                     "action_mask": action_mask
#                 }
#             observations[agent] = obs
#         return observations

#     def gen_action_mask(self, agent_obj):
#         """
#         generates the action mask, corresponding to the current status of param agent_obj;
#         """
#         if agent_obj.attached:
#             # object already attached => no pick action available
#             if len(agent_obj._reachable_goals(agent_obj.picked_object)) > 0:
#                 # a goal is reachable, allow drop action
#                 action_mask = [1., 1., 1., 1., 0., 1.]  # [left, right, up, down, pick, drop]
#             else:
#                 # no goal is reachable, mask out drop action
#                 action_mask = [1., 1., 1., 1., 0., 0.]
#         else:
#             # no object attached => no drop action available
#             if len(agent_obj._reachable_objects()) > 0:
#                 # at least one object is reachable, allow pick action
#                 action_mask = [1., 1., 1., 1., 1., 0.]
#             else:
#                 # no object reachable, mask out pick action
#                 action_mask = [1., 1., 1., 1., 0., 0.]
#         return action_mask


# class WarehouseEnv_2(MultiAgentEnv):
#     """
#         extends WarehouseEnv_1 with multi-agent compatibility
#     """
#     metadata = {"render.modes": ["human", "rgb_array"]}

#     def __init__(self, env_config:dict, agent_ids=['agent_1', 'agent_2'], shape=(10, 10), num_objects=2, seed=None, max_steps=400, deterministic_game=False,
#                  image_observation=False, action_masking=True, num_obstacles=0, random_items=[1,0,0]):
#         super().__init__()
#         self.timestep = 0
#         self._agent_ids = agent_ids
#         self.active_agent_ids = agent_ids
#         self.game = WarehouseGame_2(env_config,
#             agent_ids=self._agent_ids,
#             shape=shape,
#             max_steps=max_steps,
#             #seed=self.seed(),
#             num_objects=num_objects,
#             num_goals=num_objects,
#             num_obstacles=num_obstacles,
#             random_items=random_items
#         )
#         self.deterministic_game:bool = deterministic_game
#         self.image_observation:bool = image_observation
#         self.action_masking = action_masking
#         self.viewer = None
#         # # Action space is discrete
#         self.actions = Actions_extended
#         self.action_space = spaces.Discrete(len(self.actions))
#         if self.image_observation:
#             # image observation; observation space is nd-box
#             observation_space = spaces.Box(
#                 low=0,
#                 high=255,
#                 shape=(
#                     shape[0]*4,
#                     shape[1]*4,
#                     3,
#                 ),
#                 dtype="uint8",
#             )

#             if self.action_masking:
#                 observation_space = spaces.Dict(
#                     {
#                         "action_mask": spaces.Box(0.0, 1.0, shape=(self.action_space.n,)),
#                         "observations": observation_space,
#                     }
#                 )
#         else:
#             # vector observation: a flattened 3-dim state with one-hot encoded items (=> vector) on grid locations (=> matrix);
#             observation_space = spaces.MultiBinary(shape[0] * shape[1] * ItemKind_onehot.num_itemkind)
#             #observation_space = spaces.Box(low=0.,high=1.,shape=(shape[0] ,shape[1] ,ItemKind_onehot.num_itemkind),dtype="float")
#             if self.action_masking:
#                 observation_space = spaces.Dict(
#                     {
#                         "action_mask": spaces.Box(0.0, 1.0, shape=(self.action_space.n,)),
#                         "observations": observation_space,
#                     }
#                 )

#         # self.observation_space = spaces.Dict({"image": oimg}) # orig code cheind
#         self.observation_space = observation_space
#         if self.action_masking:
#             print("+++++ Environment created, observation space is {}".format(self.observation_space["observations"]))
#         else:
#             print("+++++ Environment created, observation space is {}".format(self.observation_space))

#         # seeding RNGs (for reproducability of training results)
#         #if "seed" in env_config.keys():
#         #    self.seed(env_config["seed"])
#         #    self.action_space.seed(env_config["seed"])
#         #else:
#         #    self.seed(1)
#         #    self.action_space.seed(1)

#         self.reset()

#     def step(self, action_dict):
#         observations = {}
#         rewards = {}
#         dones = {}
#         infos = {}
#         self.timestep += 1
#         action_dict_game = {} # maps agent objects to actions
#         for agent in action_dict.keys():
#             agent_obj = self.game.agent_dict[agent]
#             action_dict_game[agent_obj] = Actions_extended[action_dict[agent]]
#         self.game.step(action_dict_game)
#         dones["__all__"] = self.game.done
#         obs = self._gen_obs(image_observation=self.image_observation)
#         for agent in action_dict.keys():
#             # only an active agent (i.e., the agent id is a key in action_dict) should return observations, rewards etc.
#             observations[agent] = obs[agent]
#             rewards[agent] = self.game.reward_history[agent][-1]
#             dones[agent] = self.game.step_data[agent]['terminated']
#             infos[agent] = self.game.step_data[agent]['infos']
#             # for debugging
#             # print("agent {} action mask: {}".format(agent, observations[agent]["action_mask"]))
#         return observations, rewards, dones, infos

#     def reset(self) -> dict:
#         self.timestep = 0
#         if self.deterministic_game == True:
#             self.game.reset(deterministic_game=True)
#         else:
#             self.game.reset(deterministic_game=False)
#         #print("+++++ RESET returning:")
#         #print(self._gen_obs())
#         obs = self._gen_obs(image_observation=self.image_observation)
#         return obs

#     def render(self, mode="human", image_observation:bool=True):
#         state = self.game.render(image_observation=image_observation) # state can be an image or a matrix
#         if mode == "human" and image_observation:
#             if self.viewer is None:
#                 from gym.envs.classic_control import rendering

#                 self.viewer = rendering.SimpleImageViewer()
#             self.viewer.imshow(state)
#         else:
#             return state

#     def close(self):
#         if self.viewer is not None:
#             self.viewer.close()

#     def seed(self, seed): # TODO call with game.seed
#         self.game.seed(seed)

#     def _gen_obs(self, image_observation: bool=True) -> dict:
#         """
#         returns a dict with agent ids as keys and observations as values;
#         """

#         observations = {}
#         for agent in self._agent_ids:
#             obs = self.render(mode="rgb_array", image_observation=image_observation)
#             agent_obj: concert.gridworld.items.PickAgent = self.game.agent_dict[agent]
#             if self.action_masking:
#                 action_mask = self.gen_action_mask(agent_obj)
#                 action_mask = np.asarray(action_mask, dtype='float32')
#                 obs = {
#                     "observations": obs,
#                     "action_mask": action_mask
#                 }
#             observations[agent] = obs
#         return observations

#     def gen_action_mask(self, agent_obj: concert.gridworld.items.PickAgent):
#         """
#         generates the action mask, corresponding to the current status of param agent_obj;
#         pick & drop actions are masked such that they are only available if pick/drop is
#         possible in the current status of the environment; move actions are masked out to avoid collisions
#         with impassable items;
#         """
#         # masking pick/drop actions
#         if agent_obj.attached:
#             # object already attached => no pick action available
#             if len(agent_obj._reachable_goals(agent_obj.picked_object)) > 0:
#                 # a goal is reachable, allow drop action
#                 action_mask = [1., 1., 1., 1., 0., 1., 1.]  # [left, right, up, down, pick, drop, do nothing]
#             else:
#                 # no goal is reachable, mask out drop action
#                 action_mask = [1., 1., 1., 1., 0., 0., 1.]
#         else:
#             # no object attached => no drop action available
#             if len(agent_obj._reachable_objects()) > 0:
#                 # at least one object is reachable, allow pick action
#                 action_mask = [1., 1., 1., 1., 1., 0., 1.]
#             else:
#                 # no object reachable, mask out pick action
#                 action_mask = [1., 1., 1., 1., 0., 0., 1.]

#         # masking move actions
#         # agent is adjacent to an impassable item
#         adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("left", agent_obj)
#         if adjacent_to_impassable and not impassable_item == agent_obj.picked_object:
#             action_mask[0] = 0.
#         adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("right", agent_obj)
#         if adjacent_to_impassable and not impassable_item == agent_obj.picked_object:
#             action_mask[1] = 0.
#         adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("up", agent_obj)
#         if adjacent_to_impassable and not impassable_item == agent_obj.picked_object:
#             action_mask[2] = 0.
#         adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("down", agent_obj)
#         if adjacent_to_impassable and not impassable_item == agent_obj.picked_object:
#             action_mask[3] = 0.
#         # attached object is adjacent to an impassable item
#         if not agent_obj.picked_object == None:
#             adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("left", agent_obj.picked_object)
#             if adjacent_to_impassable and not impassable_item == agent_obj:
#                 action_mask[0] = 0.
#             adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("right", agent_obj.picked_object)
#             if adjacent_to_impassable and not impassable_item == agent_obj:
#                 action_mask[1] = 0.
#             adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("up", agent_obj.picked_object)
#             if adjacent_to_impassable and not impassable_item == agent_obj:
#                 action_mask[2] = 0.
#             adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("down", agent_obj.picked_object)
#             if adjacent_to_impassable and not impassable_item == agent_obj:
#                 action_mask[3] = 0.

#         return action_mask


# class WarehouseEnv_3(MultiAgentEnv):
#     """
#         extends WarehouseEnv_1 with multi-agent compatibility
#     """
#     metadata = {"render.modes": ["human", "rgb_array"]}

#     def __init__(self, env_config, agent_ids=['agent_1'], shape=(10, 10), num_objects=1, seed=None, max_steps=40, deterministic_game=False, mode = "training",
#                  image_observation=False, action_masking=True, obstacle= False, obs_stacking = True, stacked_layers = 4,  heuristic_agent = True, goal_area=False, central_critic = True,
#                  dynamic=False, three_grid_object=True, goals_coord = [(2,2),(7,7),(4,2),(2,7)]):
#         super().__init__()
#         ##heuristic agent id will append in multiagentenv class agent ids attribute



#         #environment agents : heuristic and random agents
#         self.env_agents = []
               
#         if heuristic_agent == True and "heuristic_agent" not in agent_ids:
#             self.env_agents.append("heuristic_agent")
        
#         #if dynamic is true add a random_agent
#         if dynamic == True and "random_agent" not in agent_ids:
#             self.env_agents.append("random_agent")

#         #create random agent heuristic algorithm
#         if dynamic == False:
#             self.goals_coord = [(2,2)] # dummy coords
#         if dynamic == True:
        
#             #4 goals cooradinates/waypoints for dynamic obstacle/random agent 
#             self.goals_coord = goals_coord

#             #config for randomness in dynamic obstacle
#             self.dynamic_randomness = handle_not_element(env_config, "dynamic_obstacle_randomness", 0.25)

#             print("Dynamic agent randomness set to {}".format(self.dynamic_randomness))
            
#             self.r_agent =  RandomAgent(shape =[shape[0], shape[1]], items_number = 9,
#             wall_encoding = 0, agent_encoding = 3, h_agent_encoding = 4, r_agent_encoding = 7, object_encoding = 1, random_factor = self.dynamic_randomness,
#             object_attached_encoding = 5, goal_encoding = 2, agent_attached_encoding = 6,  goals_coord = self.goals_coord )

#         #create new h_agent to get action for heuristic for each new game
#         if heuristic_agent == True:

#             self.heuristic_randomness = handle_not_element(env_config, "heuristic_agent_randomness", 0.25)

#             print("Heuristic agent randomness set to {}".format(self.heuristic_randomness))

#             self.h_agent = HeuristicAgent(shape =[shape[0], shape[1]], items_number = 9,
#             wall_encoding = 0, agent_encoding = 3, h_agent_encoding = 4, r_agent_encoding = 7, object_encoding = 1, random_factor = self.heuristic_randomness ,
#             object_attached_encoding = 5, goal_encoding = 2, 
#             agent_attached_encoding = 6, dynamic=dynamic, obstacle=obstacle, three_grid_object=three_grid_object)
        
#         self.timestep = 0
#         self.mode = mode
#         self._agent_ids = agent_ids

#         self.central_critic = central_critic


#         #both environment and policy agents
#         self.all_agents = self.env_agents + self._agent_ids

#         self.goal_area = goal_area
#         self.three_grid_object = three_grid_object
        

        
#         self.game = WarehouseGame_3(env_config,
#             agent_ids=self.all_agents,
#             shape=shape,
#             max_steps=max_steps,
#             #seed=self.seed(),
#             num_objects=num_objects,
#             num_goals=num_objects,
#             obstacle = obstacle,
#             dynamic = dynamic,
#             mode = mode,
#             goal_area = goal_area,
#             #pass the first goals_coordinate as the initial position of dynamic obstacle ie dynamic obstacle will generate at this position when env is created
#             initial_dynamic_pos= self.goals_coord[0],
#             three_grid_object= self.three_grid_object
#         )
#         self.obstacle = obstacle
#         self.dynamic = dynamic
#         self.deterministic_game:bool = deterministic_game
#         self.goal_area:bool = goal_area
#         self.image_observation:bool = image_observation
#         self.action_masking = action_masking
#         self.viewer = None

#         self.terminateds = set()
#         self.truncateds = set()

#         self.obs_stacking = obs_stacking
#         self.stacked_layers = stacked_layers

#         if self.obs_stacking:
#             self.obs_stacked = []
#         # # Action space is discrete
#         ##allow drop action too in goal_area settings
#         if goal_area:
#             self.actions = Actions_extended
#         else:
#             self.actions = Actions_reduced



#         self.action_space = spaces.Discrete(len(self.actions))

#         if self.image_observation:
#             # image observation; observation space is nd-box
#             if self.obs_stacking:
#                 observation_space = spaces.Box(
#                     low=0,
#                     high=255,
#                     shape=(self.stacked_layers,
#                         shape[0]*4,
#                         shape[1]*4,
#                         3,
#                     ),
#                     dtype="uint8",
#                 )

#             else:

#                 observation_space = spaces.Box(
#                     low=0,
#                     high=255,
#                     shape=(
#                         shape[0]*4,
#                         shape[1]*4,
#                         3,
#                     ),
#                     dtype="uint8",
#                 )

#             if self.action_masking and self.central_critic:
#                 observation_space = spaces.Dict(
#                     {
#                         "action_mask": spaces.Box(0.0, 1.0, shape=(spaces.Discrete(len(self.actions)).n,)),
#                         "num_agents": spaces.Box(0.0, 1.0, shape=(spaces.Discrete(len(self._agent_ids)).n,)),
#                         "observations": observation_space,     
#                     }
#                 )
#             elif self.action_masking:
#                 observation_space = spaces.Dict(
#                     {
#                         "action_mask": spaces.Box(0.0, 1.0, shape=(spaces.Discrete(len(self.actions)).n,)),
#                         "observations": observation_space,
#                     }
#                 )



#         else:
#             # vector observation: a flattened 3-dim state with one-hot encoded items (=> vector) on grid locations (=> matrix);
#             if self.obs_stacking:
#                 observation_space = spaces.MultiBinary([self.stacked_layers, shape[0] * shape[1] * ItemKind_onehot.num_itemkind])

#             else:
#                 observation_space = spaces.MultiBinary(shape[0] * shape[1] * ItemKind_onehot.num_itemkind)
#             #observation_space = spaces.Box(low=0.,high=1.,shape=(shape[0] ,shape[1] ,ItemKind_onehot.num_itemkind),dtype="float")
#             if self.action_masking:
#                 observation_space = spaces.Dict(
#                     {
#                         "action_mask": spaces.Box(0.0, 1.0, shape=(spaces.Discrete(len(self.actions)).n,)),
#                         "observations": observation_space,
#                     }
#                 )

#         # self.observation_space = spaces.Dict({"image": oimg}) # orig code cheind

#         # self._obs_space_in_preferred_format = True
#         # self._obs_space_in_preferred_format = True
#         # self.action_space = gym.spaces.Dict()
#         # self.observation_space = gym.spaces.Dict()


        
#         # for p_agent in list(self._agent_ids):
#         #     self.action_space[p_agent] =  spaces.Discrete(len(self.actions))
#         #     self.observation_space[p_agent] = observation_space


#         self.observation_space= observation_space
    


        
 
#         if self.action_masking:
#             print("+++++ Environment created, observation space is {}".format(self.observation_space["observations"].shape))
#             print("+++++ Environment created, action space is {}".format(self.observation_space["action_mask"].shape))
#         else:
#             print("+++++ Environment created, observation space is {}".format(self.observation_space))

#         self.reset()

#     def step(self, action_dict, turn_basis = False):
#         observations = {}
#         rewards = {}
#         terminateds = {}
#         infos = {}
#         self.timestep += 1
#         action_dict_game = {} # maps agent objects to actions


#         #see if object is attached to more than one agent
#         objects_attached = [i for i in self.game.objects if i.attachable != True]

#         #if object is attahced by one agent or more
#         obj_agent = [i for i in self.game.objects if len(i.carriers)>0]

#         carriers = [] # the list of object carriers, if the object is attached to more than one carrier

       
#         #random and heuristic agent uses vector observation for path finding
#         default_obs = list(self._gen_obs(image_observation = False, heuristic = True))


 
      
       

#          #if object is attached to more than one agent
#         for obj in objects_attached:

            

            

#             #select move for herustic algorithm for composite object(object attached to two agents)

#             move = self.h_agent.next_move(observation = default_obs , composite = True, goals_area = [(x.loc[0],x.loc[1]) for x in self.game.goals])


#             action_dict_game[self.game.agent_dict['heuristic_agent']] = Actions_extended[move] #only heuristic agent will determine the moves

#             #to get policy_agent item; remove heuristic agent from carrier copy list and the remaining one is policy agent
#             obj_carr = obj.carriers.copy()
#             obj_carr.remove(self.game.agent_dict['heuristic_agent'])
#             for obj_carrier in obj_carr:
#                 policy_agent = obj_carrier

            
#                 action_dict_game[policy_agent] = Actions_extended[6] #policy agent will do nothing
#                 carriers.extend(obj.carriers)


#         for agent in self.all_agents:

#             agent_obj = self.game.agent_dict[agent]
         

    
#             if agent != 'random_agent':
#                 if agent_obj not in carriers and agent_obj.attached:
#                     # agent_obj is attached to the object, and agent_obj is the only carrier
                    
#                     if self.goal_area and agent != "heuristic_agent":
#                         action_dict_game[agent_obj] = Actions_extended[action_dict[agent]] #get move from policy 
#                     else:
#                         action_dict_game[agent_obj] = Actions_extended[6] #do nothing if agent_obj is attached but not in the carriers list





#                 elif agent_obj not in carriers:
#                     #if agent is not attached and not in carriers list

                    
                    
#                     if agent == 'heuristic_agent':
#                         #if turn basis on then heuristic agent will do nothing if the other agent is having a move; heuristic agent will only move when other agent is not moving
#                         #turn basis True means only one step at a time, if heuristic agent is passed as doing nothing and none of the agent is attached to the object
#                         if turn_basis and action_dict['heuristic_agent'] == 6 and len(obj_agent)<1:
#                             move = 6 #heuristic agent does nothing
#                         else:
  
#                             move = self.h_agent.next_move(observation = default_obs , composite = False, goals_area = [(x.loc[0],x.loc[1]) for x in self.game.goals]) #get move from heruistic algorithm
                                
#                         action_dict_game[agent_obj] = Actions_extended[move]

#                     else:
#                         #turn basis True means only one step at a time, if heuristic agent is making move and none of the agent is attached to the object
#                         if turn_basis and action_dict['heuristic_agent'] != 6 and len(obj_agent)<1:
#                             action_dict_game[agent_obj] = Actions_extended[6] #policy agents does nothing
#                         else:

#                             action_dict_game[agent_obj] = Actions_extended[action_dict[agent]] #get move from policy for other agents
                
              

                            
#             else:

#                     #if agent is random agent
                    

#                     #get random agent move from heuristic pass goal loc and observation 
 
#                     random_agent_move= self.r_agent.next_move(observation = default_obs)
                     
#                     action_dict_game[agent_obj] =Actions_extended[random_agent_move] 

                        
    
#         self.game.step(action_dict_game)


    
                    
#         terminateds["__all__"] = self.game.done

#         #after a step is taken swap the observations if obs stacking is used 

#         obs = self._gen_obs(image_observation=self.image_observation, swap=True)

#         truncateds = {}


#         for agent in self._agent_ids:

    
#             observations[agent] = obs[agent]
#             rewards[agent] = self.game.reward_history[agent][-1]
#             terminateds[agent] = self.game.step_data[agent]['terminated'] 

#             truncateds[agent] =  self.game.step_data[agent]["infos"]["steps_exceeded"] == -1 #equals to -1 means step exceeded so episode is truncated
#             infos[agent] = self.game.step_data[agent]['infos']
#             truncateds["__all__"] = truncateds[agent] 
#             # for debugging
#             # print("agent {} action mask: {}".format(agent, observations[agent]["action_mask"]))
        
      

#         return observations, rewards, terminateds, truncateds, infos
        
#     def reset(self, *, seed=None, options=None) -> dict:
#         self.terminateds = set()
#         self.truncateds = set()
#         self.timestep = 0
#         infos = {}
#         obs_env = {}
#         if self.deterministic_game == True:
#             self.game.reset(deterministic_game=True, goal_area=self.goal_area)
#         else:
#             self.game.reset(deterministic_game=False, goal_area=self.goal_area)
#         #print("+++++ RESET returning:")
#         #print(self._gen_obs())

#         if self.obs_stacking: #reset stacked obs when game is reset
#             self.obs_stacked = []

#          #reset goal weights for dynamic obstacle/random agent whenever environment is reset   
#         if self.dynamic and self.obstacle: 
#             self.r_agent.goals_weights =  [(self.r_agent.goals_coord.index(i)+1)/10 for i in self.r_agent.goals_coord]

#         obs = self._gen_obs(image_observation=self.image_observation)


#         # for agent in self._agent_ids:
#         #     obs_env[agent], infos[agent] = obs[agent],  ''
        

#         return obs, {}

#     def render(self, mode="human", image_observation:bool=True):
#         state = self.game.render(image_observation=image_observation) # state can be an image or a matrix
#         if mode == "human" and image_observation:
#             if self.viewer is None:
#                 from gym.envs.classic_control import rendering

#                 self.viewer = rendering.SimpleImageViewer()
#             self.viewer.imshow(state)
#         else:
#             return state

#     def close(self):
#         if self.viewer is not None:
#             self.viewer.close()

#     def seed(self, seed):
#         self.game.seed(seed)

#     def _gen_obs(self, image_observation: bool=True, swap: bool=False, heuristic: bool=False) -> dict:
#         """
#         returns a dict with agent ids as keys and observations as values; swap if obs stacking is used
#         """

#         observations = {}

#         obs = self.render(mode="rgb_array", image_observation=image_observation)

#         if heuristic == True:
#             return obs #in this case obs is used by heuristic algorithm

#         if self.obs_stacking:
#             if len(self.obs_stacked) == 0: #i.e game has been reset
#                 self.obs_stacked = np.asarray([obs] * self.stacked_layers, dtype='float32') #all stacked obs will be the set to reset env initial obs
#             else:
#                 if swap:
#                     #swaps the stacked observations 
#                     #new obs in last obs; previous obs are swapped with the next obs
#                     for i in range((self.stacked_layers)-1):
#                         self.obs_stacked[i] = self.obs_stacked[i+1]
    
#                     self.obs_stacked[-1] = obs

#         for agent in self._agent_ids:
#             obs = self.render(mode="rgb_array", image_observation=image_observation)


            
#             agent_obj: concert.gridworld.items.PickAgent = self.game.agent_dict[agent]
#             if self.action_masking:
#                 action_mask = self.gen_action_mask(agent_obj)
#                 action_mask = np.asarray(action_mask, dtype='float32')

#                 if self.obs_stacking:
 
#                     obs = {
#                         "observations": self.obs_stacked, #use stacked obs 
#                         "action_mask": action_mask
#                     }
#                     observations[agent] = obs
#                 else:

#                     if self.central_critic:

#                         obs = {
#                             "observations": obs, # TODO qmix requires "obs" as key, PPO with ActionMaskModel requires "observations"
#                             "action_mask": action_mask,
#                             "num_agents" : np.random.rand(len(self._agent_ids))
#                         }

#                     else:
#                         obs = {
#                             "observations": obs, # TODO qmix requires "obs" as key, PPO with ActionMaskModel requires "observations"
#                             "action_mask": action_mask
#                         }
#                     observations[agent] = obs
      

        
        
      
#         return observations

#     def gen_action_mask(self, agent_obj: concert.gridworld.items.PickAgent):
#         """
#         generates the action mask, corresponding to the current status of param agent_obj;
#         pick & drop actions are completely masked out, i.e., they are only available if pick/drop is
#         possible in the current status of the environment;
#         """

#         #["left", "right", "up", "down", "pick", "drop", "do_nothing"]
#         # masking pick/drop actions
#         if len(agent_obj._reachable_objects()) > 0:
#             # at least one object is reachable, allow pick action
#             action_mask = [1., 1., 1., 1., 1.]

#             #if goal area we need drop action and do nothing too for the policy agent
#             if self.goal_area:
#                 action_mask = [1., 1., 1., 1., 1., 0., 1.]
#         else:
#             # no object reachable, mask out pick action
#             action_mask = [1., 1., 1., 1., 0.]
#             if self.goal_area:
#                 action_mask = [1., 1., 1., 1., 0., 0., 1.]

#         #all action except do nothing ruled out when any of the object is attached by two agents
#         #if self.game._check_object_attached:


#         # masking move actions
#         # agent is adjacent to an impassable item
#         adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("left", agent_obj)
#         if adjacent_to_impassable and not impassable_item == agent_obj.picked_object:
#             action_mask[0] = 0.
#         adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("right", agent_obj)
#         if adjacent_to_impassable and not impassable_item == agent_obj.picked_object:
#             action_mask[1] = 0.
#         adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("up", agent_obj)
#         if adjacent_to_impassable and not impassable_item == agent_obj.picked_object:
#             action_mask[2] = 0.
#         adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("down", agent_obj)
#         if adjacent_to_impassable and not impassable_item == agent_obj.picked_object:
#             action_mask[3] = 0.
#         # attached object is adjacent to an impassable item


#         #if goal area and object is attached then allow only drop and do nothing option
#         if self.goal_area and agent_obj.picked_object != None:
#             action_mask = [0, 0, 0, 0, 0, 1, 1]


#         if not agent_obj.picked_object == None:
#             adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("left", agent_obj.picked_object)
#             if adjacent_to_impassable and not impassable_item == agent_obj:
#                 action_mask[0] = 0.
#             adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("right", agent_obj.picked_object)
#             if adjacent_to_impassable and not impassable_item == agent_obj:
#                 action_mask[1] = 0.
#             adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("up", agent_obj.picked_object)
#             if adjacent_to_impassable and not impassable_item == agent_obj:
#                 action_mask[2] = 0.
#             adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("down", agent_obj.picked_object)
#             if adjacent_to_impassable and not impassable_item == agent_obj:
#                 action_mask[3] = 0.

#         return action_mask

# class WarehouseEnv_3_notused(gym.Env):
#     """
#         Environment for single-agent RL (the robot); based on WarehouseEnv_2; the human agent is kind of a dynamic objstacle, being
#         part of the environment; the human agent behaves according to a hardcoded, heuristic policy;
#     """
#     metadata = {"render.modes": ["human", "rgb_array"]}

#     def __init__(self, env_config, shape=(10, 10), num_objects=2, seed=None, max_steps=400, deterministic_game=False,
#                  image_observation=False, action_masking=True, num_obstacles=0):
#         super().__init__()
#         self.timestep = 0
#         #self._agent_ids = agent_ids
#         self.game = WarehouseGame_3(env_config,
#             #agent_ids=self._agent_ids,
#             shape=shape,
#             max_steps=max_steps,
#             seed=seed, # TODO: use "seed" from env_config; better: create self.seed, and used this when calling the seed method
#             num_objects=num_objects,
#             num_goals=num_objects,
#             num_obstacles=num_obstacles,
#         )
#         self.deterministic_game:bool = deterministic_game
#         self.image_observation:bool = image_observation
#         self.action_masking = action_masking
#         self.viewer = None
#         # # Action space is discrete
#         self.actions = Actions_extended
#         self.action_space = spaces.Discrete(len(self.actions))
#         if self.image_observation:
#             # image observation; observation space is nd-box
#             observation_space = spaces.Box(
#                 low=0,
#                 high=255,
#                 shape=(
#                     shape[0]*4,
#                     shape[1]*4,
#                     3,
#                 ),
#                 dtype="uint8",
#             )
#             if self.action_masking:
#                 observation_space = spaces.Dict(
#                     {
#                         "action_mask": spaces.Box(0.0, 1.0, shape=(self.action_space.n,)),
#                         "observations": observation_space,
#                     }
#                 )
#         else:
#             # vector observation: a flattened 3-dim state with one-hot encoded items (=> vector) on grid locations (=> matrix);
#             observation_space = spaces.MultiBinary(shape[0] * shape[1] * ItemKind_onehot.num_itemkind)
#             #observation_space = spaces.Box(low=0.,high=1.,shape=(shape[0] ,shape[1] ,ItemKind_onehot.num_itemkind),dtype="float")
#             if self.action_masking:
#                 observation_space = spaces.Dict(
#                     {
#                         "action_mask": spaces.Box(0.0, 1.0, shape=(self.action_space.n,)),
#                         "observations": observation_space,
#                     }
#                 )

#         # self.observation_space = spaces.Dict({"image": oimg}) # orig code cheind
#         self.observation_space = observation_space
#         if self.action_masking:
#             print("+++++ Environment created, observation space is {}".format(self.observation_space["observations"]))
#         else:
#             print("+++++ Environment created, observation space is {}".format(self.observation_space))

#         # seeding RNGs (for reproducability of training results)
#         #if "seed" in env_config.keys():
#         #    self.seed(env_config["seed"])
#         #    self.action_space.seed(env_config["seed"])
#         #else:
#         #    self.seed(1)
#         #    self.action_space.seed(1)

#         self.reset()

#     def step(self, action: int):
#         #observations = {}
#         #rewards = {}
#         #dones = {}
#         #infos = {}
#         self.timestep += 1
#         #action_dict_game = {} # maps agent objects to actions
#         #for agent in self._agent_ids:
#         #    agent_obj = self.game.agent_dict[agent]
#         #    action_dict_game[agent_obj] = Actions_extended[action_dict[agent]]
#         self.game.step({self.game.agent: Actions_extended[action]})
#         #dones["__all__"] = self.game.done
#         observation = self._gen_obs(image_observation=self.image_observation)
#         #for agent in self._agent_ids:
#         #    observations[agent] = obs[agent]
#         #    rewards[agent] = self.game.reward_history[agent][-1]
#         #    dones[agent] = self.game.step_data[agent]['terminated']
#         #    infos[agent] = self.game.step_data[agent]['infos']
#             # for debugging
#             # print("agent {} action mask: {}".format(agent, observations[agent]["action_mask"]))
#         return observation, self.game.reward, self.game.done, self.game.step_data['infos']

#     def reset(self) -> dict:
#         self.timestep = 0
#         if self.deterministic_game == True:
#             self.game.reset(deterministic_game=True)
#         else:
#             self.game.reset(deterministic_game=False)
#         #print("+++++ RESET returning:")
#         #print(self._gen_obs())
#         obs = self._gen_obs(image_observation=self.image_observation)
#         return obs

#     def render(self, mode="human", image_observation:bool=True):
#         state = self.game.render(image_observation=image_observation) # state can be an image or a matrix
#         if mode == "human" and image_observation:
#             if self.viewer is None:
#                 from gym.envs.classic_control import rendering

#                 self.viewer = rendering.SimpleImageViewer()
#             self.viewer.imshow(state)
#         else:
#             return state

#     def close(self):
#         if self.viewer is not None:
#             self.viewer.close()

#     def seed(self, seed): # TODO call with game.seed
#         self.game.seed(seed)

#     def _gen_obs(self, image_observation: bool=True) -> dict:
#         obs = self.render(mode="rgb_array", image_observation=image_observation)
#         if self.action_masking:
#             action_mask = self.gen_action_mask(self.game.agent)
#             action_mask = np.asarray(action_mask, dtype='float32')
#             obs = {
#                     "observations": obs,
#                     "action_mask": action_mask
#             }
#         return obs

#     def gen_action_mask(self, agent_obj: concert.gridworld.items.PickAgent):
#         """
#         generates the action mask, corresponding to the current status of param agent_obj;
#         pick & drop actions are completely masked out, i.e., they are only available if pick/drop is
#         possible in the current status of the environment;
#         """
#         # masking pick/drop actions
#         if agent_obj.attached:
#             # object already attached => no pick action available
#             if len(agent_obj._reachable_goals(agent_obj.picked_object)) > 0:
#                 # a goal is reachable, allow drop action
#                 action_mask = [1., 1., 1., 1., 0., 1., 1.]  # [left, right, up, down, pick, drop, do nothing]
#             else:
#                 # no goal is reachable, mask out drop action
#                 action_mask = [1., 1., 1., 1., 0., 0., 1.]
#         else:
#             # no object attached => no drop action available
#             if len(agent_obj._reachable_objects()) > 0:
#                 # at least one object is reachable, allow pick action
#                 action_mask = [1., 1., 1., 1., 1., 0., 1.]
#             else:
#                 # no object reachable, mask out pick action
#                 action_mask = [1., 1., 1., 1., 0., 0., 1.]

#         # masking move actions
#         # agent is adjacent to an impassable item
#         adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("left", agent_obj)
#         if adjacent_to_impassable and not impassable_item == agent_obj.picked_object:
#             action_mask[0] = 0.
#         adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("right", agent_obj)
#         if adjacent_to_impassable and not impassable_item == agent_obj.picked_object:
#             action_mask[1] = 0.
#         adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("up", agent_obj)
#         if adjacent_to_impassable and not impassable_item == agent_obj.picked_object:
#             action_mask[2] = 0.
#         adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("down", agent_obj)
#         if adjacent_to_impassable and not impassable_item == agent_obj.picked_object:
#             action_mask[3] = 0.
#         # attached object is adjacent to an impassable item
#         if not agent_obj.picked_object == None:
#             adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("left", agent_obj.picked_object)
#             if adjacent_to_impassable and not impassable_item == agent_obj:
#                 action_mask[0] = 0.
#             adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("right", agent_obj.picked_object)
#             if adjacent_to_impassable and not impassable_item == agent_obj:
#                 action_mask[1] = 0.
#             adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("up", agent_obj.picked_object)
#             if adjacent_to_impassable and not impassable_item == agent_obj:
#                 action_mask[2] = 0.
#             adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("down", agent_obj.picked_object)
#             if adjacent_to_impassable and not impassable_item == agent_obj:
#                 action_mask[3] = 0.

#         return action_mask