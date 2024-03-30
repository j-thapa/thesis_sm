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
from sm.envs.concert.gridworld.items import ItemKind, ItemBase, ItemKind_onehot


import matplotlib.pyplot as plt


from sm.envs.concert.gyms.heuristic_agent import HeuristicAgent, RandomAgent

from sm.envs.concert.gridworld.examples.warehouse_game import WarehouseGame, WarehouseGame_1, WarehouseGame_2, WarehouseGame_3







#different actions are used cause in diferent settings no need to drop; also have to factor action mask

Actions = ["left", "right", "up", "down", "pick", "drop", "do_nothing"]
Actions_extended = ["left", "right", "up", "down", "pick", "drop", "do_nothing"]
Actions_reduced = ["left", "right", "up", "down", "pick", "drop", "do_nothing"]









class WarehouseMultiEnv(MultiAgentEnv):

    def __init__(self, batch_size=None, **kwargs):
        super().__init__(batch_size, **kwargs)

        self.env_agents = []

        self.seed =kwargs['all_args'].seed






        self.three_grid_object =kwargs['all_args'].three_grid_object

        self.n_agents =kwargs['all_args'].num_agents
        self.obstacle =kwargs['all_args'].obstacle

        self.num_objects =kwargs['all_args'].num_objects

        self.shape =kwargs['all_args'].grid_shape

        self._agent_ids = []
        for num in range(self.n_agents):
            self._agent_ids.append(f'agent_{num+1}')

        self.heuristic_agent  =kwargs['all_args'].heuristic_agent
        self.dynamic =kwargs['all_args'].random_agent

        if self.heuristic_agent == True:
            self.env_agents.append("heuristic_agent")
        
        #if dynamic is true add a random_agent
        if self.dynamic == True:
            self.env_agents.append("random_agent")


        #create random agent heuristic algorithm
        if self.dynamic == False:
            self.goals_coord = [(2,2)] # dummy coords
        if self.dynamic == True:
        
            #4 goals cooradinates/waypoints for dynamic obstacle/random agent 
            self.goals_coord =kwargs['all_args'].goals_coord

            #config for randomness in dynamic obstacle
            self.dynamic_randomness =kwargs['all_args'].dynamic_obstacle_randomness

            print("Dynamic agent randomness set to {}".format(self.dynamic_randomness))
            
            self.r_agent =  RandomAgent(shape =[self.shape[0], self.shape[1]], items_number = 9,
            wall_encoding = 0, agent_encoding = 3, h_agent_encoding = 4, r_agent_encoding = 7, object_encoding = 1, random_factor = self.dynamic_randomness,
            object_attached_encoding = 5, goal_encoding = 2, agent_attached_encoding = 6,  goals_coord = self.goals_coord )

        #create new h_agent to get action for heuristic for each new game
        if self.heuristic_agent == True:

            self.heuristic_randomness =kwargs['all_args'].heuristic_agent_randomness

            print("Heuristic agent randomness set to {}".format(self.heuristic_randomness))

            self.h_agent = HeuristicAgent(shape =[self.shape[0], self.shape[1]], items_number = 9,
            wall_encoding = 0, agent_encoding = 3, h_agent_encoding = 4, r_agent_encoding = 7, object_encoding = 1, random_factor = self.heuristic_randomness ,
            object_attached_encoding = 5, goal_encoding = 2, 
            agent_attached_encoding = 6, dynamic=self.dynamic, obstacle=self.obstacle, three_grid_object=self.three_grid_object)
        

        self.timestep = 0
        self.mode =kwargs['all_args'].mode

        self.max_steps =kwargs['all_args'].max_steps



        #both environment and policy agents
        self.all_agents = self.env_agents + self._agent_ids

        self.goal_area =kwargs['all_args'].goal_area

        env_config = {
            "reward_game_success": kwargs["all_args"].reward_game_success, # default: 1.0
            "reward_each_action": kwargs["all_args"].reward_each_action, # default: -0.01
            "reward_illegal_action": kwargs["all_args"].reward_illegal_action, # default: -0.05
            "seed": self.seed
            }





        self.game = WarehouseGame_3(env_config,
            agent_ids=self.all_agents,
            shape=self.shape,
            max_steps=self.max_steps,
            seed=self.seed,
            num_objects=self.num_objects,
            num_goals=self.num_objects,
            obstacle = self.obstacle,
            dynamic = self.dynamic,
            mode = self.mode,
            goal_area = self.goal_area,
            #pass the first goals_coordinate as the initial position of dynamic obstacle ie dynamic obstacle will generate at this position when env is created
            initial_dynamic_pos= self.goals_coord[0],
            three_grid_object= self.three_grid_object
        )



        self.deterministic_game:bool =kwargs['all_args'].deterministic_game
   
        self.image_observation:bool =kwargs['all_args'].image_observation
        self.action_masking =kwargs['all_args'].action_masking
        self.viewer = None

        self.terminateds = set()
        self.truncateds = set()

        self.obs_stacking =kwargs['all_args'].obs_stacking
        self.stacked_layers =kwargs['all_args'].stacked_layers
        if self.obs_stacking:
            self.obs_stacked = []
        # # Action space is discrete
        ##allow drop action too in goal_area settings
        if self.goal_area:
            self.actions = Actions_extended
        else:
            self.actions = Actions_reduced

        self.n_actions = len(self.actions)



        if self.image_observation:
            # image observation; observation space is nd-box
            if self.obs_stacking:
                observation_space = spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.stacked_layers,
                        self.shape[0]*4,
                        self.shape[1]*4,
                        3,
                    ),
                    dtype="uint8",
                )

            else:

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

            # if self.action_masking and self.central_critic:
            #     observation_space = spaces.Dict(
            #         {
            #             "action_mask": spaces.Box(0.0, 1.0, shape=(spaces.Discrete(len(self.actions)).n,)),
            #             "num_agents": spaces.Box(0.0, 1.0, shape=(spaces.Discrete(len(self._agent_ids)).n,)),
            #             "observations": observation_space,     
            #         }
            #     )
            # elif self.action_masking:
            #     observation_space = spaces.Dict(
            #         {
            #             "action_mask": spaces.Box(0.0, 1.0, shape=(spaces.Discrete(len(self.actions)).n,)),
            #             "observations": observation_space,
            #         }
            #     )



        else:
            # vector observation: a flattened 3-dim state with one-hot encoded items (=> vector) on grid locations (=> matrix);
            if self.obs_stacking:
                observation_space = spaces.MultiBinary([self.stacked_layers, self.shape[0] * self.shape[1] * ItemKind_onehot.num_itemkind])

            else:
                observation_space = spaces.MultiBinary(self.shape[0] * self.shape[1] * ItemKind_onehot.num_itemkind)
            #observation_space = spaces.Box(low=0.,high=1.,shape=(shape[0] ,shape[1] ,ItemKind_onehot.num_itemkind),dtype="float")
            # if self.action_masking:
            #     observation_space = spaces.Dict(
            #         {
            #             "action_mask": spaces.Box(0.0, 1.0, shape=(spaces.Discrete(len(self.actions)).n,)),
            #             "observations": observation_space,
            #         }
            #     )

        # self.observation_space = spaces.Dict({"image": oimg}) # orig code cheind

        # self._obs_space_in_preferred_format = True
        # self._obs_space_in_preferred_format = True
        # self.action_space = gym.spaces.Dict()
        # self.observation_space = gym.spaces.Dict()


        
        # for p_agent in list(self._agent_ids):
        #     self.action_space[p_agent] =  spaces.Discrete(len(self.actions))
        #     self.observation_space[p_agent] = observation_space


        self.observation_space= [observation_space for _ in range(self.n_agents)]

        self.share_observation_space= [observation_space for _ in range(self.n_agents)]

        self.action_space = [spaces.Discrete(len(self.actions)) for _ in range(self.n_agents)]

    


        
 
        # if self.action_masking:
        #     print("+++++ Environment created, observation space is {}".format(self.observation_space["observations"].shape))
        #     print("+++++ Environment created, action space is {}".format(self.observation_space["action_mask"].shape))
        # else:
        print("+++++ Environment created, observation space is {}".format(self.observation_space))
        print("+++++ Environment created, action space is {}".format(self.action_space))

   



    def step(self, actions):

        turn_basis = False

      

        observations = {}
        rewards = {}
        terminateds = {}
        infos = {}
        self.timestep += 1
        action_dict_game = {} # maps agent objects to action
        


        #see if object is attached to more than one agent
        objects_attached = [i for i in self.game.objects if i.attachable != True]

        #if object is attahced by one agent or more
        obj_agent = [i for i in self.game.objects if len(i.carriers)>0]

        carriers = [] # the list of object carriers, if the object is attached to more than one carrier

       
###no specidic observation for an agent, sop using same observaTION FOR  all agents

        #random and heuristic agent uses vector observation for path finding
        default_obs = list(self.render(mode="human", image_observation= False))


 
      
       

         #if object is attached to more than one agent
        for obj in objects_attached:

            

            

            #select move for herustic algorithm for composite object(object attached to two agents)

            move = self.h_agent.next_move(observation = default_obs , composite = True, goals_area = [(x.loc[0],x.loc[1]) for x in self.game.goals])


            action_dict_game[self.game.agent_dict['heuristic_agent']] = Actions_extended[move] #only heuristic agent will determine the moves

            #to get policy_agent item; remove heuristic agent from carrier copy list and the remaining one is policy agent
            obj_carr = obj.carriers.copy()
            obj_carr.remove(self.game.agent_dict['heuristic_agent'])
            for obj_carrier in obj_carr:
                policy_agent = obj_carrier

            
                action_dict_game[policy_agent] = Actions_extended[6] #policy agent will do nothing
                carriers.extend(obj.carriers)
    


        for agent in self.all_agents:

            agent_obj = self.game.agent_dict[agent]
    

            # img =self.render(mode="rgb_array", image_observation= True)

            # fig, ax = plt.subplots()
            # mpl_img = ax.imshow(img)
            # mpl_img.set_data(img)
            # fig.canvas.draw()
            # plt.savefig("test_img.jpg")
         

    
            if agent != 'random_agent':
                if agent_obj not in carriers and agent_obj.attached:
                    # agent_obj is attached to the object but the object is not attached by all agents yet

                    if self.goal_area and agent != "heuristic_agent":

                        agent_index = int(agent.split('_')[-1]) - 1 
                        
                        action_dict_game[agent_obj] = Actions_extended[actions[agent_index][0]] #get move from policy 
                    else:
                        action_dict_game[agent_obj] = Actions_extended[6] #do nothing if agent_obj is attached but not in the carriers list





                elif agent_obj not in carriers:
                    #if agent is not attached and not in carriers list

                    
                    
                    if agent == 'heuristic_agent':
                        #if turn basis on then heuristic agent will do nothing if the other agent is having a move; heuristic agent will only move when other agent is not moving
                        #turn basis True means only one step at a time, if heuristic agent is passed as doing nothing and none of the agent is attached to the object
                        if turn_basis and action_dict['heuristic_agent'] == 6 and len(obj_agent)<1:
                            move = 6 #heuristic agent does nothing
                        else:
  
                            move = self.h_agent.next_move(observation = default_obs , composite = False, goals_area = [(x.loc[0],x.loc[1]) for x in self.game.goals]) #get move from heruistic algorithm
                                
                        action_dict_game[agent_obj] = Actions_extended[move]

                    else:
                        #turn basis True means only one step at a time, if heuristic agent is making move and none of the agent is attached to the object
                        if turn_basis and action_dict['heuristic_agent'] != 6 and len(obj_agent)<1:
                            action_dict_game[agent_obj] = Actions_extended[6] #policy agents does nothing
                        else:

                            agent_index = int(agent.split('_')[-1]) - 1 #-1 beacuse we are using agent id starts from 1 and we use it for indexing

   
                            
                        
                            action_dict_game[agent_obj] = Actions_extended[actions[agent_index][0]] #get move from policy for other agents
                
              

                            
            else:

                    #if agent is random agent
                    

                    #get random agent move from heuristic pass goal loc and observation 
 
                    random_agent_move= self.r_agent.next_move(observation = default_obs)
                     
                    action_dict_game[agent_obj] =Actions_extended[random_agent_move] 

        

                        
    
        self.game.step(action_dict_game)


    
                    
        terminateds["__all__"] = self.game.done



        #after a step is taken swap the observations if obs stacking is used 

        obs = self.get_obs_agent(0)

        truncateds = {}

        rewards = []
        dones = []
        infos = []
        for agent in self._agent_ids:
        
            rewards.append([self.game.reward_history[agent][-1]])
            dones.append( self.game.step_data[agent]['terminated'])
            infos.append( self.game.step_data[agent]['infos'])


                #reset the game and start new if game is terminated
        if self.game.done:
            
            self.reset()

         
    
            # observations[agent] = obs[agent]
            # rewards[agent] = self.game.reward_history[agent][-1]
            # terminateds[agent] = self.game.step_data[agent]['terminated'] 

            # truncateds[agent] =  self.game.step_data[agent]["infos"]["steps_exceeded"] == -1 #equals to -1 means step exceeded so episode is truncated
            # infos[agent] = self.game.step_data[agent]['infos']
            # truncateds["__all__"] = truncateds[agent] 
            # for debugging
        #     # print("agent {} action mask: {}".format(agent, observations[agent]["action_mask"]))
        
        # # return reward_n, done_n, info
        # rewards = [[reward_n]] * self.n_agents
        # dones = [done_n] * self.n_agents
        # infos = [info for _ in range(self.n_agents)]
        # # print("obs: ", self.get_obs())
        # # print("state: ", self.get_state())

      


        # if self.game.step_data[agent]['terminated']:
        #     self.game.reset(deterministic_game= self.deterministic_game, goal_area=self.goal_area)



        return self.get_obs(), self.get_state(), rewards, dones, infos, self.get_avail_actions()





    def reset(self, *, seed= 42, options=None) -> dict:
        self.terminateds = set()
        self.truncateds = set()
        self.timestep = 0
        infos = {}
        obs_env = {}
        if self.deterministic_game == True:
            self.game.reset(deterministic_game=True, seed = self.seed , goal_area=self.goal_area)
        else:
            self.game.reset(deterministic_game=False, seed= self.seed , goal_area=self.goal_area)
        #print("+++++ RESET returning:")
        #print(self._gen_obs())

        if self.obs_stacking: #reset stacked obs when game is reset
            self.obs_stacked = []

         #reset goal weights for dynamic obstacle/random agent whenever environment is reset   
        if self.dynamic and self.obstacle: 
            self.r_agent.goals_weights =  [(self.r_agent.goals_coord.index(i)+1)/10 for i in self.r_agent.goals_coord]




        # for agent in self._agent_ids:
        #     obs_env[agent], infos[agent] = obs[agent],  ''
        

        return self.get_obs(), self.get_state(), self.get_avail_actions()















    # def get_obs(self):
    #     """ Returns all agent observat3ions in a list """
    #     state = self.env._get_obs()
    #     obs_n = []
    #     for a in range(self.n_agents):
    #         agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)
    #         agent_id_feats[a] = 1.0
    #         # obs_n.append(self.get_obs_agent(a))
    #         # obs_n.append(np.concatenate([state, self.get_obs_agent(a), agent_id_feats]))
    #         # obs_n.append(np.concatenate([self.get_obs_agent(a), agent_id_feats]))
    #         obs_i = np.concatenate([state, agent_id_feats])
    #         obs_i = (obs_i - np.mean(obs_i)) / np.std(obs_i)
    #         obs_n.append(obs_i)
    #     return obs_n

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
        
        return self.render(mode="rgb_array", image_observation=self.image_observation)


            # return build_obs(self.env,
            #                  self.k_dicts[agent_id],
            #                  self.k_categories,
            #                  self.mujoco_globals,
            #                  self.global_categories)

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
        agent_obj = self.game.agent_dict[f'agent_{agent_id + 1}']
        action_mask = self.gen_action_mask(agent_obj)
        action_mask = np.asarray(action_mask, dtype='float32')
        

        return action_mask


    def gen_action_mask(self, agent_obj):
        """
        generates the action mask, corresponding to the current status of param agent_obj;
        """
        if agent_obj.attached:
            # object already attached => no pick action available
            if len(agent_obj._reachable_goals(agent_obj.picked_object)) > 0:
                # a goal is reachable, allow drop action
                action_mask = [1., 1., 1., 1., 0., 1., 1.]  # [left, right, up, down, pick, drop, do_nothing]
            else:
                # no goal is reachable, mask out drop action
                action_mask = [1., 1., 1., 1., 0., 0., 1.]
    
            if agent_obj.picked_object.attachable:
                action_mask = [0., 0., 0., 0., 0., 1., 1.] #mask out everything except drop and do nothing if the agent is attached but object still need more agents 
                
        else:
            # no object attached => no drop action available
            if len(agent_obj._reachable_objects()) > 0:
                # at least one object is reachable, allow pick action
                action_mask = [1., 1., 1., 1., 1., 0., 1.]
            else:
                # no object reachable, mask out pick action
                action_mask = [1., 1., 1., 1., 0., 0., 1.]
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
     

    def render(self, mode="human", image_observation:bool=True):
        state = self.game.render(image_observation=image_observation) # state can be an image or a matrix
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
