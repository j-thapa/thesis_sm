import ray
#from ray.rllib import evaluation
#from ray.rllib.agents import trainer
from  ray.rllib.algorithms.ppo import PPOConfig
import numpy as np
import  csv
import random

from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec

from concert.gyms.warehouse_env import WarehouseEnv_1, WarehouseEnv_2, WarehouseEnv_3
from concert.gyms.examples.concert_visionnet import VisionNetwork_1
from concert.gyms.examples.torch_network import TorchCentralizedCriticModel
import matplotlib
import matplotlib.pyplot as plt
from ray.rllib.examples.models.action_mask_model import ActionMaskModel

from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

"""
Evaluation of a trained agent by stepping through a single random game

1) restore trained agent policy from checkpoint
2) create random game => render
3) step through the game by calling the policy => render after each step
"""
ray.shutdown()
ray.init()

image_observation = True #whether the agent was trained using images or vectorized represenations
action_masking = True



agents = ['agent_1', 'agent_2']

ModelCatalog.register_custom_model("visionnet", VisionNetwork_1)


#ModelCatalog.register_custom_model("vn_central_critic", TorchCentralizedCriticModel)

goal_area = True
num_objects = 1

    # if goal_area and num_objects <2 :
    #     raise ValueError("Num of objects should be greater than 1 in case of goal_area")
    
register_env("WarehouseEnv_3",
                      lambda config: WarehouseEnv_3(config,
                                                    agent_ids=agents, max_steps=90, deterministic_game=False,
                                                    shape=(10,10),
                                                    obs_stacking= False, stacked_layers= 4,
                                                    image_observation=image_observation, action_masking=action_masking, 
                                                    num_objects= num_objects, obstacle = False, goal_area = goal_area,
                                                    dynamic= False, mode="training", three_grid_object= True,
                                                    goals_coord = [(8,2),(5,2),(7,4),(2,5),(3,8)]))


    # creating policies
policies = {"policy_1": PolicySpec(),
                "policy_2": PolicySpec(),
                "policy_3" : PolicySpec(),
                "policy_4" : PolicySpec()}  # default PPO policies, observation and action spaces are inferred from environment # default PPO policies, observation and action spaces are inferred from environment


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id == "agent_1":
            return "policy_1"
        else:
            return "policy_2"
        
if image_observation:
    model_config = {
        "custom_model": "visionnet",
        "custom_model_config": {
        }
    }

if image_observation == False:
    model_config = {
        "custom_model": ActionMaskModel,
        "custom_model_config": {
            "no-masking": False,
        },
    
        # "use_lstm": True,
        # "max_seq_len": 5,
        #"lstm_use_prev_action": True,
    }
configuration =(PPOConfig()
    .framework( "tf",
        eager_tracing = False)
    .training(
        model = model_config,
        use_gae = True,
        lambda_ = 0.87,
        sgd_minibatch_size = 128,
        clip_param = 0.16,
        train_batch_size = 1500,
        entropy_coeff = 0.079,
        vf_loss_coeff = 0.82
        )
    .environment("WarehouseEnv_3",
    #seed = 1, # seed RNGs on RLlib level; for more information on why seeding is a good idea: https://docs.ray.io/en/latest/tune/faq.html#how-can-i-reproduce-experiments; https://github.com/ray-project/ray/blob/master/rllib/examples/deterministic_training.py
    env_config = {
        "reward_game_success": 1.0, # default: 1.0
        "reward_each_action": -0.03, # default: -0.01
        "reward_illegal_action": -0.067, # default: -0.05
        "dynamic_obstacle_randomness": 0, #tune.grid_search([0.00, 0.20, 0.40, 0.60, 0.80, 1]), #defualt 0.25
        "heuristic_agent_randomness": 0.3, #tune.grid_search([0.00, 0.20, 0.40, 0.60, 0.80, 1]) #defualt 0.25
        "seed":  2,
        #"worker_index" : 2,
        },
        
    disable_env_checking = True)
    #.callbacks(MyCallbacks)
    #.log_level("WARN")
    .experimental(
    _disable_preprocessor_api = True
    )
    .resources(
        num_cpus_per_worker = 1,
        num_gpus_per_worker = 0,
        num_gpus = 0
        )
    .rollouts(
        rollout_fragment_length = 750,
        num_rollout_workers=0,
        batch_mode = "complete_episodes"
        )
    .multi_agent(
            policies = policies,
        policy_mapping_fn =  policy_mapping_fn
    #     # "observation_fn": central_critic_observer,
    )
    .debugging(
        log_level = 'WARN',
        seed = 2
    )

)

# checkpoint_path = "D:/Profactor/ray_results/checkpoint_006507/"
# import time
# algo = Algorithm.from_checkpoint(checkpoint_path)
# print("algo loaded +==+++++++++++++++++")
# time.sleep(2300)

# configuration = {
#         "env": "WarehouseEnv",
#         "log_level": "ERROR",
#         #"num_gpus": 0.2, ### gpu share assigned to local trainer
#         "num_workers": 0,
#         "train_batch_size": 400,
#         "rollout_fragment_length": 400,
#         "batch_mode": "complete_episodes",
#         # "sgd_minibatch_size": 64,
#         #"num_gpus_per_worker": 0.8,
#         "num_cpus_per_worker": 1,
#         "explore": False,
#         "framework": "tf", #tf for tensorflow, tfe for tensorflow eager
#         "disable_env_checking": True,
#         #turn this off (False) if want to use actionmaskmodel of rllib, (turn on parameter to use action masking along with image obeservation(pass environment observation as it is))
#         "_disable_preprocessor_api" : True,
#         #"lr": 0.00005,
#         # "lr": tune.grid_search([0.0001, 0.00005]),
#         # "clip_param": tune.grid_search([0.3, 0.25, 0.2, 0.15, 0.1]), decreasing values produce ever worse rewards, see PPO_2021-07-06_12-14-57
#         # "lambda": tune.grid_search([1.0, 0.975, 0.95]), # decreasing values produce ever worse rewards, see PPO_2021-07-06_10-51-14
#         # "entropy_coeff": tune.grid_search([0.0, 0.025, 0.05, 0.075, 0.1]), # increasing values produce ever worse rewards, see PPO_2021-07-05_17-11-12
#         "multiagent": {
#             "policies": policies,
#             "policy_mapping_fn": policy_mapping_fn,
#             # "observation_fn": central_critic_observer,
#         },
#         # "model": {
#         #     #"dim": 45,
#         #     #"conv_filters": [[16, [5, 5], 1]], # [32, [3, 3], 2], [512, [3, 3], 1]],
#         #     #"conv_activation": "relu",
#         #     #"post_fcnet_hiddens": [256],
#         #     #"post_fcnet_activation": "relu",
#         #     "custom_model": "visionnet",
#         #     "custom_model_config": {
#         #     #    "conv_filters": [[16, [5, 5], 1], [32, [5, 5], 1]], # [128, [5, 5], 1]],
#         #     #    "conv_activation": "relu",
#         #     #    "post_fcnet_hiddens": [256],
#         #     #    "post_fcnet_activation": "relu",
#         #     #    "no_final_linear": True,
#         #     #    #"vf_share_layers":,
#         #     }
#         # },
#         "model":model_config,
#         #"num_sgd_iter": 1,
#         #"vf_loss_coeff": 0.0001,
#         #"callbacks": MyCallbacks,
#     }

evaluator = configuration.build()
checkpoint_path = "D:/Profactor/ray_results/three_grid/checkpoint_003996"

evaluator.restore(checkpoint_path)



# gans
#evaluator.restore('D:/Profactor/marl/concert/gyms/examples/chk/checkpoint-5000')

#evaluator.restore('D:/Profactor/ray_results/conv/checkpoint-6627')

# gpusrv
# evaluator.restore("D:/Profactor/path_finder/hp/checkpoint_005000/checkpoint-5000")

def evaluate(num_episodes = 1):
    #matplotlib.use('Agg')

    # create the environment for evaluation
    env_config = {
        "reward_game_success": 1.0, 
        "reward_each_action": -0.01,
        "reward_move_failure": -0.05,
        "heuristic_agent_randomness": 0,
        "dynamic_obstacle_randomness": 0,
        "seed": 13
    }



    
    env =  WarehouseEnv_3(env_config= env_config, shape=(10, 10), agent_ids=agents, max_steps = 60, 
                          deterministic_game= False, obs_stacking= False,
                                image_observation=image_observation, action_masking=action_masking, num_objects=1, obstacle = False, goal_area = True, dynamic= False, mode="training",
                                 goals_coord = [(8,2),(5,2),(7,4),(2,5),(3,8)], three_grid_object=True)

    #policy = evaluator.get_policy('policy_1')
    #model = policy.model
    #model.internal_model.base_model.summary()

    for i in range(num_episodes):
        observations = env.reset() [0]  # reset the environment
        print("++++++ observation after reset ++++++++")

        fig, ax = plt.subplots()
 

        mpl_img = ax.imshow(env.game.render())
        mpl_img.set_data(env.game.render())
        fig.canvas.draw()
        plt.ion()
        plt.show()

            # cv2.imshow('game',env.game.render())

        episode_reward = 0
        timestep = 0

        terminateds= {}
        terminateds["__all__"] = False

        # import pdb; pdb.set_trace()

        # step through one episode
        while not terminateds["__all__"]:
            # compute actions
            actions_dict = {}

    
            for agent in env._agent_ids:
                
                if agent != 'heuristic_agent' and agent!= "random_agent":



                    actions_dict[agent] = evaluator.compute_single_action(observations[agent], policy_id= "policy_1")

            # for debugging
            # print("timestep {}".format(timestep))
            # print("action agent 1: {}".format(actions_dict[agents[0]]))
            # #print("action agent 2: {}".format(actions_dict[agents[1]]))
            #input("Press [enter] to continue. ")
            
            # execute one step
            observations, rewards, terminateds, truncateds, infos = env.step(actions_dict)
   
            # print(observations)

            # show updated game after step
            mpl_img = ax.imshow(env.game.render())
        
            mpl_img.set_data(env.game.render())
            fig.canvas.draw()
            # plt.savefig(f"D:/Profactor/demo_dynamic/{i}_{timestep}.png")
            input("Press [enter] to continue. ")

            
            #cv2.imshow('game',env.game.render())

            # print("after step: producer_1 load: {}".format(env.agents_dict['producer_1'].load))

            for reward in rewards.values():
                episode_reward += reward
            timestep += 1

        print("episode reward: " + str(episode_reward))
        #input("Press [enter] to continue. ")
        plt.close()
    # endof evaluate()

def turn_basis_evaluation(num_episodes=10):
    '''

    either heuristic or policy agent makes the step in each timestep unless one is attached to the object; turn by turn basis evaluation 


    '''
    # create the environment for evaluation
    env_config = {
        "reward_game_success": 1.0,
        "reward_each_action": -0.01,
        "reward_move_failure": -0.1,
        "heuristic_agent_randomness": 0,
    }

    env = WarehouseEnv_3(env_config, agent_ids=agents, max_steps=30, deterministic_game=False,
                        image_observation=image_observation, action_masking=action_masking, num_objects=1, obstacle= False, mode="evaluation",
                        dynamic=False,  goals_coord = [(2,7),(4,2),(6,5),(7,3)])

    policy = evaluator.get_policy('policy_1')
    model = policy.model
    model.internal_model.base_model.summary()


    for i in range(num_episodes):
        observations = env.reset()  # reset the environment
        print("++++++ observation after reset ++++++++")

        fig, ax = plt.subplots()

        mpl_img = ax.imshow(env.game.render())
        mpl_img.set_data(env.game.render())
        fig.canvas.draw()
        plt.ion()
        plt.show()
        # cv2.imshow('game',env.game.render())

        episode_reward = 0
        timestep = 0

        dones = {}
        dones["__all__"] = False

    #function to choose whether or heuristic or policy agent as initial turn
        choose_agent = lambda: "heuristic_agent" if random.randint(0, 1) == 0 else "policy_agent"
        step_turn = choose_agent()
    

        # import pdb; pdb.set_trace()

        # step through one episode
        while not dones["__all__"]:
            # compute actions
            actions_dict = {}

    
            for agent in env._agent_ids:
                
                if agent != 'heuristic_agent' and agent!= "random_agent":

                    #if its not heuristic agent turn
                    if step_turn != "heuristic_agent":

                        #evaluate policy agent action
                        actions_dict[agent] = evaluator.compute_action(observations[agent], policy_id= "policy_1")
                        #6 is do nothing; heuristic agent does nothing
                        actions_dict["heuristic_agent"] = 6
                        #next turn is turn of heuristic agent
                        step_turn = "heuristic_agent"
                    else:
                        #evaluate and send policy move but wont be used if object is not attached to heuristic agent as it is heuristic agent turn
                        actions_dict[agent] = evaluator.compute_action(observations[agent], policy_id= "policy_1")
                        #send 1 as heuristic agent move; heuristic move will later be decided in environment step method
                        actions_dict["heuristic_agent"] = 1
                        #next is policy agent turn
                        step_turn = "policy_agent"
                
    


            # execute one step

            # input("Press [enter] to continue. ")
            observations, rewards, dones, infos = env.step(actions_dict, turn_basis = True)
            # print(observations)

            # show updated game after step
            mpl_img = ax.imshow(env.game.render())
            mpl_img.set_data(env.game.render())
            fig.canvas.draw()
            
            # cv2.imshow('game',env.game.render())

            # print("after step: producer_1 load: {}".format(env.agents_dict['producer_1'].load))

            for reward in rewards.values():
                episode_reward += reward
            timestep += 1

        print("episode reward: " + str(episode_reward))
        plt.close()

    


def evaluate_to_csv(episodes=300, name="evaluation_csv", obs_stack = False):
    """
    writes agent observations, actions and success into a csv file;
    """

    header = ['Observation', 'Next_Observation', 'Agent', 'Action', 'Action_success','Object_attached','First Object Attached','Complete_Episode','Episode','Timestep']
    
    with open(name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        env_config = {
            "reward_game_success": 1.0, 
            "reward_each_action": -0.01,
            "reward_move_failure": -0.05,
            "heuristic_agent_randomness": 0,
            "dynamic_obstacle_randomness": 0,
            "seed": 12
        }



        
        env =  WarehouseEnv_3(env_config= env_config, agent_ids=agents, max_steps=60, deterministic_game=False, obs_stacking= False,
                                    image_observation=image_observation, action_masking=action_masking, num_objects =2, obstacle = False, goal_area = True, dynamic= False, mode="evaluation",
                                    goals_coord = [(8,2),(5,2),(7,4),(2,5),(3,8)])

        
        policy = evaluator.get_policy('policy_1')
        model = policy.model

    
        for eps in range(episodes):


            observations = env.reset()[0]  # reset the environment


            episode_reward = 0
            timestep = 0
            attach_timestep = 100

            terminateds = {}
            terminateds["__all__"] = False

            # import pdb; pdb.set_trace()

            # step through one episode
            while not terminateds["__all__"]:
                # compute actions
                actions_dict = {}

                first_object_attached = 0



 


                #get previous observation
                obs = env._gen_obs(image_observation = False)
                previous_obs =  obs[env._agent_ids[0]]['observations']
             

                

               

                for agent in env._agent_ids:
                    
                    if agent != 'heuristic_agent':
                        actions_dict[agent] = evaluator.compute_single_action(observations[agent], policy_id= "policy_1")

                # execute one step and get the infos and observation after
             
                observations, rewards, terminateds,truncateds, infos = env.step(actions_dict)
               
                

                print("running episode", eps+1)
                print("--------------------")

                #gives the step info for that time step
                step_info = env.game.step_data

                if step_info[agent]["infos"]["first_object_attached"] == 1:
                    if timestep < attach_timestep:
                        first_object_attached = 1
                        attach_timestep = timestep


                #attached gives info whether the object was attached to both agents or not in the time step
                object_attached = [i for i in env.game.objects if len(i.carriers) == 2]

                if len(object_attached) == len(env.game.objects):
                    attached = 1
                else:
                    attached = 0

                obs = env._gen_obs(image_observation = False)

                default_obs = obs[env._agent_ids[0]]['observations']
              

             
            
                
                for agent in env._agent_ids:


                     #eps+1 to get initial episode as 1 not 0

     
                    if obs_stack:

                    #in observation stacking case previous obseervation is stakced on top of latest observation i.e observations[-2] = previous observation


                        if step_info[agent]['action'] == 'drop': #drop is the 
                        
                            writer.writerow([default_obs [-2] , default_obs [-1] , 
                            agent, step_info[agent]['action'],
                            step_info[agent]['infos']['drop_failure'], attached ,first_object_attached,
                            step_info[agent]["infos"]["steps_exceeded"], eps+1, timestep])
                




                        elif step_info[agent]['action'] == 'pick': #pick is the move

                            writer.writerow([default_obs [-2] , default_obs [-1] , 
                            agent, step_info[agent]['action'], 
                            step_info[agent]['infos']['pick_failure'], attached ,first_object_attached ,
                            step_info[agent]["infos"]["steps_exceeded"], eps+1, timestep])
                


                        else: #other moves
                            writer.writerow([default_obs [-2] , default_obs [-1] , 
                            agent, step_info[agent]['action'], 
                            step_info[agent]['infos']['move_failure'], attached , first_object_attached ,
                            step_info[agent]["infos"]["steps_exceeded"], eps+1, timestep])
                       
                    else:
                       

                        if step_info[agent]['action'] == 'drop': #drop is the 
                        
                            writer.writerow([previous_obs, default_obs, 
                            agent, step_info[agent]['action'],
                            step_info[agent]['infos']['drop_failure'], attached ,first_object_attached,
                            step_info[agent]["infos"]["steps_exceeded"], eps+1, timestep])
                




                        elif step_info[agent]['action'] == 'pick': #pick is the move

                            writer.writerow([previous_obs, default_obs, 
                            agent, step_info[agent]['action'], 
                            step_info[agent]['infos']['pick_failure'], attached , first_object_attached ,
                            step_info[agent]["infos"]["steps_exceeded"], eps+1, timestep])
                


                        else: #other moves
                            writer.writerow([previous_obs, default_obs, 
                            agent, step_info[agent]['action'], 
                            step_info[agent]['infos']['move_failure'], attached , first_object_attached ,
                            step_info[agent]["infos"]["steps_exceeded"], eps+1, timestep])
                
  
                timestep += 1
   

     
                print("++Current timestamp++++++", timestep)
                
         


def turn_basis_evaluate_to_csv(episodes=300, name="evaluation_csv",obs_stack=True):
    """
    writes agent observations, actions and success into a csv file with turn basis; 
    one turn for heuristic agent and another turn for policy agents
    """

    #header for csv file
    header = ['Observation', 'Next_Observation', 'Agent', 'Action', 'Action_success','Object_attached','Complete_Episode','Episode','Timestep']
    
    with open(name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        env_config = {
            "reward_game_success": 1.0,
            "reward_each_action": -0.01,
            "reward_move_failure": -0.1,
            "heuristic_agent_randomness": 0.3
        }

        env = WarehouseEnv_3(env_config, agent_ids=agents, max_steps=20, deterministic_game=False, obs_stacking = obs_stack,
                            image_observation=image_observation, action_masking=action_masking, num_objects=1, obstacle= False, mode="evaluation",
                            dynamic=False,  goals_coord = [(2,7),(4,2),(6,5),(7,3)])

        
        policy = evaluator.get_policy('policy_1')
        model = policy.model


 
    #loop for number of episodes
        
        for eps in range(episodes):

            #function to choose whether or heuristic or policy agent as initial turn
            choose_agent = lambda: "heuristic_agent" if random.randint(0, 1) == 0 else "policy_agent"
            step_turn = choose_agent()




            observations = env.reset()  # reset the environment


            timestep = 0

            dones = {}
            dones["__all__"] = False


            # step through one episode
            while not dones["__all__"]:
                # compute actions
                actions_dict = {}

                obs = env._gen_obs(image_observation = False)
                previous_obs =  obs[env._agent_ids[0]]['observations']

                for agent in env._agent_ids:

            
                    if agent != 'heuristic_agent' and agent!= "random_agent":

                        #if its not heuristic agent turn
                        if step_turn != "heuristic_agent":

                            #evaluate policy agent action
                            actions_dict[agent] = evaluator.compute_action(observations[agent], policy_id= "policy_1")
                            #6 is do nothing; heuristic agent does nothing
                            actions_dict["heuristic_agent"] = 6
                            #next turn is turn of heuristic agent
                            step_turn = "heuristic_agent"
                        else:
                            #evaluate and send policy move but wont be used if object is not attached to heuristic agent as it is heuristic agent turn
                            actions_dict[agent] = evaluator.compute_action(observations[agent], policy_id= "policy_1")
                            #send 1 as heuristic agent move; heuristic move will later be decided in environment step method
                            actions_dict["heuristic_agent"] = 1
                            #next is policy agent turn
                            step_turn = "policy_agent"

  
                # execute one step and get the infos and observation after
                observations, rewards, dones, infos = env.step(actions_dict, turn_basis = True)

                print("running episode", eps+1, 'timestep',timestep)
                print("--------------------")

                #gives the step info for that time step
                step_info = env.game.step_data



                #attached gives info whether the object was attached to both agents or not in the time step
                object_attached = [i for i in env.game.objects if len(i.carriers) == 2]

                if len(object_attached) == len(env.game.objects):
                    attached = 1
                else:
                    attached = 0


                obs = env._gen_obs(image_observation = False)

                default_obs = obs[env._agent_ids[0]]['observations']

                
                #loop through all agents
                
                for agent in env._agent_ids:


                    


                    #eps+1 to get initial episode as 1 not 0

                    if obs_stack:

                        #in observation stacking case previous obseervation is stakced on top of latest observation i.e observations[-2] = previous observation


                        if step_info[agent]['action'] == 'drop': #drop is the 
                        
                            
                            writer.writerow([default_obs [-2] , default_obs [-1] , 
                            agent, step_info[agent]['action'],
                            step_info[agent]['infos']['drop_failure'], attached , step_info[agent]["infos"]["steps_exceeded"], eps+1, timestep])
                




                        elif step_info[agent]['action'] == 'pick': #pick is the move

                            writer.writerow([default_obs [-2] , default_obs [-1] , 
                            agent, step_info[agent]['action'], 
                            step_info[agent]['infos']['pick_failure'], attached , step_info[agent]["infos"]["steps_exceeded"], eps+1, timestep])
                


                        else: #other moves
                            writer.writerow([default_obs [-2] , default_obs [-1] , 
                            agent, step_info[agent]['action'], 
                            step_info[agent]['infos']['move_failure'], attached , step_info[agent]["infos"]["steps_exceeded"], eps+1, timestep])
                    
                    else:

                        if step_info[agent]['action'] == 'drop': #drop is the 
                        
                            writer.writerow([previous_obs, default_obs, 
                            agent, step_info[agent]['action'],
                            step_info[agent]['infos']['drop_failure'], attached , step_info[agent]["infos"]["steps_exceeded"], eps+1, timestep])
                




                        elif step_info[agent]['action'] == 'pick': #pick is the move

                            writer.writerow([previous_obs, default_obs, 
                            agent, step_info[agent]['action'], 
                            step_info[agent]['infos']['pick_failure'], attached , step_info[agent]["infos"]["steps_exceeded"], eps+1, timestep])
                


                        else: #other moves
                            writer.writerow([previous_obs, default_obs, 
                            agent, step_info[agent]['action'], 
                            step_info[agent]['infos']['move_failure'], attached , step_info[agent]["infos"]["steps_exceeded"], eps+1, timestep])
    
                timestep += 1
         


evaluate(num_episodes= 7)

#turn_basis_evaluation(num_episodes=5)

# # # #episodes: number of episodes to run , name: path and name of generated csv
# evaluate_to_csv(episodes=300, name="no_extended_reward", obs_stack=False)


# # turn_basis_evaluate_to_csv(episodes=300, name="turn_csv1", obs_stack=False)
ray.shutdown()








"""
def on_press(event):
    if event.key == "a":
        pass
    else:
        pass
    mpl_img.set_data(env.game.render())
    fig.canvas.draw()
"""

