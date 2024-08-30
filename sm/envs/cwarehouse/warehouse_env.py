from functools import partial
import gym
from gym.spaces import Box
from gym.wrappers import TimeLimit
import numpy as np

from .multiagentenv import MultiAgentEnv

import matplotlib
import matplotlib.pyplot as plt

import gym
import numpy as np
from gym import spaces
from PIL import Image

from gym import ObservationWrapper

import sys
from .gridworld.items import ItemKind, ItemBase, ItemKind_onehot, ItemKind_encode


import matplotlib.pyplot as plt


from .gridworld.examples.warehouse_game import WarehouseGame




Actions_list = ["left", "right", "up", "down", "pick/drop", "do_nothing"]





class WarehouseMultiEnv(MultiAgentEnv):

    def __init__(self, batch_size=None, **kwargs):
        super().__init__(batch_size, **kwargs)

        self.env_agents = []

        self.seed =kwargs['all_args'].seed


        self.n_agents = kwargs['all_args'].num_agents
        self.obstacle =kwargs['all_args'].obstacle

        self.num_objects =kwargs['all_args'].num_objects

        self.shape =kwargs['all_args'].grid_shape

        self._agent_ids = []
        self.heuristic_agent  = kwargs['all_args']

        self.partial_observation = kwargs['all_args'].partial_observation


        # add for randomizing
        self.agent_permutation = None
        self.agent_recovery = None

        for num in range(self.n_agents):
            if num % 2 == 0 :
                self._agent_ids.append(f'agent_{num}') #add robot agent
            else:
                self._agent_ids.append(f"heuristic_agent_{num}") #add robot agent pair heuristic agent



        self.timestep = 0
        self.mode =kwargs['all_args'].mode

        self.max_steps =kwargs['all_args'].max_steps


        self.game = WarehouseGame(
            agent_ids=self._agent_ids,
            shape=self.shape,
            max_steps=self.max_steps,
            seed=self.seed,
            num_objects=self.num_objects,
            obstacle = self.obstacle
        )



        self.deterministic_game:bool =kwargs['all_args'].deterministic_game
   
        self.image_observation:bool =kwargs['all_args'].image_observation
        self.action_masking =kwargs['all_args'].action_masking
        self.viewer = None

        self.terminateds = set()
        self.truncateds = set()

        # # Action space is discrete
        ##allow drop action too in goal_area settings
 
        self.actions = Actions_list

        self.n_actions = len(self.actions)



        if self.image_observation:
            # image observation; observation space is nd-box
   

            observation_space = spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        self.shape[0]*4,
                        self.shape[1]*4,
                        3,
                    ),
                    dtype="uint8",
                )


        else:
            # vector observation: a flattened 3-dim state with one-hot encoded items (=> vector) on grid locations (=> matrix);
    
            #observation_space = spaces.Box(shape=(self.shape[0] * self.shape[1] * ItemKind_onehot.num_itemkind + 2 + ItemKind_onehot.num_itemkind,),low=-np.inf, high=np.inf, dtype=np.float32)
            #for not one hot encoding
            if self.image_observation:
                observation_space = spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        shape[0]*4,
                        shape[1]*4,
                        3,
                    ),
                    dtype="uint8",
                )

            else:

                if self.partial_observation:
                    # one hot encoded observation
                    observation_space = spaces.Box(shape=(3 * 3 * ItemKind_onehot.num_itemkind + 2 + self.num_objects * 6 
                    + ItemKind_onehot.num_itemkind,),low=-np.inf, high=np.inf, dtype=np.float32)

                else:
                    

                    # one hot encoded observation
                    observation_space = spaces.Box(shape=(self.shape[0] * self.shape[1] * ItemKind_onehot.num_itemkind + 2 + self.num_objects * 3 
                    + ItemKind_onehot.num_itemkind,),low=-np.inf, high=np.inf, dtype=np.float32)


        self.observation_space= [observation_space for _ in range(self.n_agents)]

        self.share_observation_space= [observation_space for _ in range(self.n_agents)]

        self.action_space = [spaces.Discrete(len(self.actions)) for _ in range(self.n_agents)]

        print("+++++ Environment created, observation space is {}".format(self.observation_space))
        print("+++++ Environment created, action space is {}".format(self.action_space))


    def step(self, actions):


        observations = {}
        rewards = {}
        terminateds = {}
        infos = {}
        self.timestep += 1
        action_dict_game = {} # maps agent objects to action

 

        # add for randomizing
        actions = np.array(actions)[self.agent_recovery].tolist()


        for agent in self._agent_ids :

            agent_obj = self.game.agent_dict[agent]

            agent_index = int(agent.split('_')[-1])

                
            action_dict_game[agent_obj] = Actions_list[actions[agent_index][0]] #get move from policy 

            # print("input the action for", agent)
            # act = input()
            # print(act)
            # action_dict_game[agent_obj] = act
            

  
                    
    
        self.game.step(action_dict_game)


        # print("generate image")
        # g = input()
        # self.save_image()
        
        terminateds["__all__"] = self.game.done


        truncateds = {}

        rewards = []
        dones = []
        infos = []


        for agent in self._agent_ids:

            

            rewards.append([self.game.reward_history[agent][-1]])
            dones.append( self.game.step_data[agent]['terminated'])


            
            self.game.step_data[agent]['infos']['reward_till'] = (self.game.reward_history[agent])
            infos.append( self.game.step_data[agent]['infos'])
      
       

        rewards = np.array(rewards)[self.agent_permutation]
        dones = np.array(dones)[self.agent_permutation]
        infos = np.array(infos)[self.agent_permutation]

        local_obs = np.array(self.get_obs())[self.agent_permutation]


        global_state = local_obs
        available_actions = np.array(self.get_avail_actions())[self.agent_permutation]

       

        return local_obs, global_state, rewards, dones, infos, available_actions

    # add for randomizing
    def permutate_idx(self):
        self.agent_permutation = np.random.permutation(self.n_agents)
        self.agent_recovery = [np.where(self.agent_permutation == i)[0][0] for i in range(self.n_agents)]
        self.agent_recovery = np.array(self.agent_recovery)




    def reset(self, *, seed= 42, options=None) -> dict:
        self.terminateds = set()
        self.truncateds = set()
        self.timestep = 0
        infos = {}
        obs_env = {}

        # add for randomizing
        self.permutate_idx()


        if self.deterministic_game == True:
            self.game.reset(deterministic_game=True, seed = self.seed , goal_area=True)
        else:
            self.game.reset(deterministic_game=False, seed= self.seed)


       # add for randomizing

       
        local_obs = np.array(self.get_obs())[self.agent_permutation]


        global_state = local_obs
        available_actions = np.array(self.get_avail_actions())[self.agent_permutation]

      


        return local_obs, global_state, available_actions

    def save_image(self):
        img =self.render(mode="rgb_array", image_observation= True)

        fig, ax = plt.subplots()
        mpl_img = ax.imshow(img)
        mpl_img.set_data(img)
        fig.canvas.draw()
        plt.savefig("test_img.jpg")
        plt.close()
   












    def get_obs(self):
        """ Returns all agent observat3ions in a list """
        # state = self.env._get_obs()
        obs_n = []



        for a in range(self.n_agents):
            # agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)
            # agent_id_feats[a] = 1.0
            # # obs_i = np.concatenate([state, agent_id_feats])
            # obs_i = np.concatenate([self.get_obs_agent(a), agent_id_feats])
            # obs_i = (obs_i - np.mean(obs_i)) / np.std(obs_i)
            if self.image_observation:
                obs_i = self.get_obs_agent(a)
                obs_i = (obs_i - np.mean(obs_i)) / np.std(obs_i)
            else:
                obs_i = self.get_obs_agent(a)

            obs_n.append(obs_i)

   
        return obs_n

    def get_obs_agent(self, agent_id):

        if self.image_observation:
            return grid_world


        if agent_id % 2 == 0:
            agent_id = f"agent_{agent_id}"
        else:
            agent_id = f"heuristic_agent_{agent_id}"

        

        agent_loc = self.game.agent_dict[agent_id].loc


        grid_world = self.render(mode="rgb_array", image_observation=self.image_observation, 
        partial_observation= self.partial_observation, loc = agent_loc)


        # agent_encode = [ItemKind_encode.encoding[self.game.agent_dict[agent_id].kind]]

        agent_encode = ItemKind_onehot.encoding[self.game.agent_dict[agent_id].kind]

        objects_loc = np.array([x.loc for x in self.game.objects])

        objects_attach = np.array([1 if x.kind == ItemKind.OBJECT else 0 for x in self.game.objects])

        goal_loc = np.array([x.loc for x in self.game.goals])

        object_g_loc = [tuple(i.loc) for i in self.game.objects if i.kind == ItemKind.WALL]


 

        goal_free = np.array([1 if tuple(x.loc) not in object_g_loc else 0 for x in self.game.goals])
    


        return np.concatenate([agent_loc, goal_loc.flatten(), goal_free, objects_loc.flatten(), objects_attach,
                               np.array(agent_encode), grid_world], axis = 0)



    def get_obs_objects(self):

        obj_obs = []

        for object in self.game.objects:

            object_loc = object.loc


            grid_world = self.render(mode="rgb_array", image_observation=self.image_observation, 
            partial_observation= self.partial_observation, loc = object_loc)


            # agent_encode = [ItemKind_encode.encoding[self.game.agent_dict[agent_id].kind]]

            object_encode = ItemKind_onehot.encoding[object.kind]

            objects_loc = np.array([x.loc for x in self.game.objects])

            objects_attach = np.array([1 if x.kind == ItemKind.OBJECT else 0 for x in self.game.objects])

            goal_loc = np.array([x.loc for x in self.game.goals])

            object_g_loc = [tuple(i.loc) for i in self.game.objects if i.kind == ItemKind.WALL]

    

            goal_free = np.array([1 if tuple(x.loc) not in object_g_loc else 0 for x in self.game.goals])

        


            obj_obs.append(np.concatenate([object_loc, goal_loc.flatten(), goal_free, objects_loc.flatten(), objects_attach,
                                np.array(object_encode), grid_world], axis = 0))

        return np.array(obj_obs)

    def get_obs_size(self):
        """ Returns the shape of the observation """
  
        return self.get_obs_agent(0).size


    def get_state(self, team=None):
        # TODO: needed in decemtralised action, not needed or similar to obs here
        # state = self.env._get_obs()
        # share_obs = []
        # for a in range(self.n_agents):
        #     agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)
        #     agent_id_feats[a] = 1.0
        #     # share_obs.append(np.concatenate([state, self.get_obs_agent(a), agent_id_feats]))
        #     state_i = np.concatenate([state, agent_id_feats])
        #     state_i = (state_i - np.mean(state_i)) / np.std(state_i)
        #     share_obs.append(state_i)
        return self.get_obs()

    def get_state_size(self):
        """ Returns the shape of the state/ same as obs in this use case"""
        return len(self.get_state()[0])

    def get_avail_actions(self):  
        """Use action masking to mask unavailable actions """
        avail_actions = np.zeros(shape=(self.n_agents, self.n_actions,))
        for a in range(self.n_agents):
            avail_actions[a] = self.get_avail_agent_actions(a)

        return avail_actions

      

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        if agent_id % 2 == 0 : #robot agent if even
            agent_obj = self.game.agent_dict[f'agent_{agent_id}']
        else:
            agent_obj = self.game.agent_dict[f'heuristic_agent_{agent_id}']
        action_mask = self.gen_action_mask(agent_obj)
        action_mask = np.asarray(action_mask, dtype='float32')
        

        return action_mask


    def gen_action_mask(self, agent_obj):
        """
        generates the action mask, corresponding to the current status of param agent_obj;

        """
        action_mask = [1., 1., 1., 1., 1., 1.]

        if agent_obj.attached:
            # # object already attached => no pick action available
            # if len(agent_obj._reachable_goals(agent_obj.picked_object)) > 0:
            #     # a goal is reachable, allow drop action
            #     action_mask = [1., 1., 1., 1., 1., 1.]  # [left, right, up, down, pick/drop, do_nothing]
            # else:
            #     # no goal is reachable, mask out drop action
            #     action_mask = [1., 1., 1., 1., 0., 1.]

            

            
    
            if agent_obj.picked_object.attachable:
                #mask out everything except drop and do nothing if the agent is attached but object still need more agents 
                return ([0., 0., 0., 0., 1., 1.])
            
            elif agent_obj.kind == ItemKind.AGENT_ATTACHED:
                #mask out everything except do_nothing if the robot agent is attached and object is attached by both robot and heuristic agents
                return ([0., 0., 0., 0., 0., 1.])

            #now if object is composite; have to take care of both driven agent and objects
    
            elif agent_obj.kind == ItemKind.H_AGENT_ATTACHED: #composite object
            
                picked_obj = agent_obj.picked_object
                driven_agent = [x for x in picked_obj.carriers if x != agent_obj][0]

                goals_loc = [x.loc for x in self.game.goals]

        

                if any(np.array_equal(np.array(picked_obj.loc), arr) for arr in goals_loc):
                    action_mask[4] = 1.0 
                else:
                    action_mask[4] = 0.0 #mask out drop option if the object is not in goal grid cel


                #no impassable moves for picked_object

                adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("left", picked_obj)
                if adjacent_to_impassable and (impassable_item != agent_obj and impassable_item != driven_agent):
                    action_mask[0] = 0.
                adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("right", picked_obj)
                if adjacent_to_impassable and (impassable_item != agent_obj and impassable_item != driven_agent):
                    action_mask[1] = 0.
                adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("up", picked_obj)
                if adjacent_to_impassable and (impassable_item != agent_obj and impassable_item != driven_agent):
                    action_mask[2] = 0.
                adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("down", picked_obj)
                if adjacent_to_impassable and (impassable_item != agent_obj and impassable_item != driven_agent):
                    action_mask[3] = 0.


                #no impassable moves for driven agent

                adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("left", driven_agent)
                if adjacent_to_impassable and (impassable_item != picked_obj):
                    action_mask[0] = 0.
                adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("right", driven_agent)
                if adjacent_to_impassable and (impassable_item != picked_obj):
                    action_mask[1] = 0.
                adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("up", driven_agent)
                if adjacent_to_impassable and (impassable_item != picked_obj):
                    action_mask[2] = 0.
                adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("down", driven_agent)
                if adjacent_to_impassable and (impassable_item != picked_obj):
                    action_mask[3] = 0.

                
                #no impassable moves for driving agent

                adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("left", agent_obj)
                if adjacent_to_impassable and (impassable_item != picked_obj):
                    action_mask[0] = 0.
                adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("right", agent_obj)
                if adjacent_to_impassable and (impassable_item != picked_obj):
                    action_mask[1] = 0.
                adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("up", agent_obj)
                if adjacent_to_impassable and (impassable_item != picked_obj):
                    action_mask[2] = 0.
                adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("down", agent_obj)
                if adjacent_to_impassable and (impassable_item != picked_obj):
                    action_mask[3] = 0.

            # print(agent_obj.kind, picked_obj.attachable,picked_obj.carriers )
            # print("composite objecr heuristic agent selecting move", action_mask)
            # print("-------------------------")

            return action_mask




                
        else:
            # no object attached => no drop action available
            if len(agent_obj._reachable_objects()) > 0:
                # at least one object is reachable, allow pick action
                action_mask = [1., 1., 1., 1., 1., 1.]
            else:
                # no object reachable, mask out pick action
                action_mask = [1., 1., 1., 1., 0, 1.]

        

        #no impassable moves for driven agent

        adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("left",agent_obj)
        if adjacent_to_impassable:
            action_mask[0] = 0.
        adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("right",agent_obj)
        if adjacent_to_impassable:
            action_mask[1] = 0.
        adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("up",agent_obj)
        if adjacent_to_impassable:
            action_mask[2] = 0.
        adjacent_to_impassable, impassable_item = self.game.is_adjacent_to_impassable("down",agent_obj)
        if adjacent_to_impassable:
            action_mask[3] = 0.




    
        return action_mask

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.n_actions  # CAREFUL! - for continuous dims, this is action space dim rather
        # return self.env.action_space.shape[0]

    def get_stats(self):
        return {}

    # # TODO: Temp hack
    # def get_agg_stats(self, stats):
    #     return {}

    # def reset(self, **kwargs):
    #     """ Returns initial observations and states"""
    #     self.steps = 0
    #     self.timelimit_env.reset()
     

    def render(self, mode = "human", image_observation: bool = True, partial_observation: bool = False, loc=None) -> np.ndarray:

        state = self.game.render(image_observation=image_observation, partial_observation = partial_observation,
        loc = loc) # state can be an image or a matrix
        if mode == "human" and image_observation:
            if self.viewer is None:
                from gym.envs.classic_control import rendering

                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(state)
        else:
            return state

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def seed(self, seed):
        return seed

    def get_env_info(self):

        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit,
                    "action_spaces": self.action_space,
                    "actions_dtype": np.float32,
                    "normalise_actions": False
                    }
        return env_info
