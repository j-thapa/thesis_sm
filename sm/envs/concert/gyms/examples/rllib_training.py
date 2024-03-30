from ray.rllib.models import ModelCatalog
from concert_visionnet import VisionNetwork_1
from concert.gyms.warehouse_env import WarehouseEnv_1, WarehouseEnv_2, WarehouseEnv_3
from callbacks import MyCallbacks, MyCallbacks_singleAgent
#from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.rllib.policy.policy import PolicySpec
from ray.air.config import RunConfig, ScalingConfig
from ray.tune.tuner import Tuner



from ray.tune.registry import register_env
#import concert.gyms
#import concert.gridworld.items

from concert.gyms.find_goal_env import FindGoalEnv_1
from concert.gridworld.items import MoveAgent
import ray
from  ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from ray.rllib.examples.models.action_mask_model import ActionMaskModel
#import configuration
from ray.tune.registry import register_env
import tensorflow as tf


def train_one_agent():
    #import configuration
    from ray.tune.registry import register_env
    import tensorflow as tf
    ray.shutdown()
    ray.init() # for debugging: local_mode=True
    #ray.init(include_dashboard=True)
    #ray.init()

    #agent = MoveAgent((0,0), impassable=True)
    #agents_dict = {'agent_1': agent}
    agents = ['agent_1']
    image_observation = True
    action_masking = False
    if image_observation and action_masking:
        print("+++++ ERROR action_masking and image_observation both are True; such a custom model is currently not implemented, aborting training +++++++++")
        ray.shutdown()


    #env = FindGoalEnv_1(agents)
    env = WarehouseEnv_1(agents, max_steps=400, deterministic_game=False, image_observation=image_observation, action_masking=action_masking)
    #env = FlattenObservation(env)
    #env = FlattenedVectorObsEnv(env, concert.gyms.warehouse_env.flattenVectorObs)
    # env = GrayscaleEnv(env, concert.gyms.warehouse_env.rgb_to_grayscale)
    tune.register_env("WarehouseEnv_1", lambda _: env)
    ModelCatalog.register_custom_model("visionnet", VisionNetwork_1)

    policies = {agent: (None, env.observation_space, env.action_space, {}) for agent in env.agents}

    #for debugging
    #print("environment created; environment observation space shape: {}, action space shape: {}".format(env.observation_space.shape, env.action_space.shape))

    model_config = {}
    if image_observation:
        model_config = {
                #"dim": 45,
                #"conv_filters": [[16, [5, 5], 1]], # [32, [3, 3], 2], [512, [3, 3], 1]],
                #"conv_activation": "relu",
                #"post_fcnet_hiddens": [256],
                #"post_fcnet_activation": "relu",
                "custom_model": "visionnet",
                "custom_model_config": {
                #    "conv_filters": [[16, [5, 5], 1], [32, [5, 5], 1]], # [128, [5, 5], 1]],
                #    "conv_activation": "relu",
                #    "post_fcnet_hiddens": [256],
                #    "post_fcnet_activation": "relu",
                #    "no_final_linear": True,
                #    #"vf_share_layers":,
                }
            }

    if action_masking:
        model_config = {
            "custom_model": ActionMaskModel,
            "custom_model_config": {
                "no-masking": False,
            }
        }

    configuration = {
            "env": "WarehouseEnv_1",
            "log_level": "WARN",
            #"num_gpus": 0.2, ### gpu share assigned to local trainer
            "num_workers": 8,
            "train_batch_size": 4000,
            "rollout_fragment_length": 500,
            "batch_mode": "complete_episodes",
            "sgd_minibatch_size": 128,
            #"num_gpus_per_worker": 0.8,
            "num_cpus_per_worker": 1,
            "framework": "tf", #tf for tensorflow, tfe for tensorflow eager (allows debugging rllib code)
            "eager_tracing": False, # if True: allows debugging rllib code
            "disable_env_checking": True,
            #"lr": 0.00005,
            # "lr": tune.grid_search([0.0001, 0.00005]),
            # "clip_param": tune.grid_search([0.3, 0.25, 0.2, 0.15, 0.1]), decreasing values produce ever worse rewards, see PPO_2021-07-06_12-14-57
            # "lambda": tune.grid_search([1.0, 0.975, 0.95]), # decreasing values produce ever worse rewards, see PPO_2021-07-06_10-51-14
            # "entropy_coeff": tune.grid_search([0.0, 0.025, 0.05, 0.075, 0.1]), # increasing values produce ever worse rewards, see PPO_2021-07-05_17-11-12
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": (lambda agent_id: agent_id),
                # "observation_fn": central_critic_observer,
            },
            "model": model_config,
            #"num_sgd_iter": 1,
            #"vf_loss_coeff": 0.0001,
            "callbacks": MyCallbacks,
            #"output": "./episode_traces/", # logs episode traces to the experiment's ray results folder
            #"output_compress_columns": [],
        }

    results = tune.run(
        ppo.PPOTrainer,
        stop={
            #"episodes_total": 100,
            "training_iteration": 2000,
            #"timesteps_total": 1000000
            },
        reuse_actors=False,  # False is required for the tune grid_search to run
        config=configuration,
        # checkpoint_freq=25,
        checkpoint_at_end=True
    )

    ray.shutdown()


def train_two_agents_ppo():
    ray.shutdown()
    ray.init(num_cpus=25, local_mode=False)  # for debugging: local_mode=True

    # alg = BayesOptSearch(metric="episode_reward_mean", mode="max")

    agents = ['agent_1', 'agent_2']
    image_observation = False
    action_masking = True
    if image_observation and action_masking:
        print(
            "+++++ ERROR action_masking and image_observation both are True; such a custom model is currently not implemented, aborting training +++++++++")
        ray.shutdown()

    # env = env.with_agent_groups(env, {"group1": [agents[0], agents[1]]}) # required for QMIX

    tune.register_env("WarehouseEnv_2",
                      lambda config: WarehouseEnv_2(config, agent_ids=agents, max_steps=400, deterministic_game=False,
                                                    image_observation=image_observation,
                                                    action_masking=action_masking, num_objects=2, random_items=[1,0,0]))
    ModelCatalog.register_custom_model("visionnet", VisionNetwork_1)

    # creating policies
    policies = {"agent_1": PolicySpec(),
                "agent_2": PolicySpec()}  # default PPO policies, observation and action spaces are inferred from environment

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id == "agent_1":
            return "agent_1"
        elif agent_id == "agent_2":
            return "agent_2"
        else:
            return "agent_1"

    model_config = {}
    if image_observation:
        model_config = {
            "custom_model": "visionnet",
            "custom_model_config": {}
        }

    if action_masking:
        model_config = {
            "custom_model": ActionMaskModel,
            "custom_model_config": {
                "no-masking": False,
            },
            "use_lstm": True,
            "max_seq_len": 5,
            #"lstm_use_prev_action": True,
        }

    configuration = {
        "env": "WarehouseEnv_2",
        "env_config": {
            "reward_game_success": 1.0, #tune.uniform(0.5, 1.5), # default: 1.0
            "reward_each_action": -0.01043, #tune.uniform(-0.02, -0.01), # default: -0.01
            "reward_illegal_action": -0.090846, #tune.uniform(-0.2,0.1), # default: -0.1
            "seed": 3, #tune.grid_search([1,2,3,4,5]), # seed RNG on environment level
            },
        "seed": 1, #    tune.grid_search([1,2,3,4,5]), # seed RNGs on RLlib level; for more information on why seeding is a good idea: https://docs.ray.io/en/latest/tune/faq.html#how-can-i-reproduce-experiments; https://github.com/ray-project/ray/blob/master/rllib/examples/deterministic_training.py
        "log_level": "WARN",
        # "num_gpus": 0.2, ### gpu share assigned to local trainer
        "num_workers": 4,
        "train_batch_size": 2000,
        "rollout_fragment_length": 500,
        "batch_mode": "complete_episodes",
        "sgd_minibatch_size": 128,
        # "num_gpus_per_worker": 0.8,
        "num_cpus_per_worker": 1,
        "disable_env_checking": True,
        "framework": "tf",  # tf for tensorflow, tfe for tensorflow eager (allows debugging rllib code)
        "eager_tracing": False,  # if True: allows debugging into rllib code
        "entropy_coeff": 0.00175791, # tune.uniform(0.0, 0.01), # tune.uniform(lower,upper) triggers hyperparameter search (a simple random search)
        # "lr": 0.00005,
        # "lr": tune.grid_search([0.0001, 0.00005]),
        # "clip_param": tune.grid_search([0.3, 0.25, 0.2, 0.15, 0.1]), decreasing values produce ever worse rewards, see PPO_2021-07-06_12-14-57
        # "lambda": tune.grid_search([1.0, 0.975, 0.95]), # decreasing values produce ever worse rewards, see PPO_2021-07-06_10-51-14
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
            # "observation_fn": central_critic_observer,
        },
        "model": model_config,
        # "num_sgd_iter": 1,
        # "vf_loss_coeff": 0.0001,
        "callbacks": MyCallbacks,
        #"output": "./episode_traces/", # logs episode traces to the experiment's ray results folder
        #"output_compress_columns": [],
    }

    results = tune.run(
        ppo,
        metric="episode_reward_mean",
        mode="min",
        stop={
            # "episodes_total": 100,
            "training_iteration": 1000,
            # "timesteps_total": 1000000
        },
        reuse_actors=False,  # False is required for the tune grid_search to run
        config=configuration,
        # checkpoint_freq=25,
        checkpoint_at_end=True,
        # search_alg=alg,
        num_samples=5 # number of samples in the hyperparameter search space
    )

    #print("best hyperparameters: ", results.best_config)
    #print("best result: ", results.best_result)

    ray.shutdown()

def train_h_agent_agent_ppo():
    ray.shutdown()
    #ray.init(num_cpus= 16, num_gpus=1)  # for debugging: local_mode=True
    ray.init(num_cpus= 10)

    # alg = BayesOptSearch(metric="episode_reward_mean", mode="max")

    agents = ['agent_1']
    image_observation = True
    action_masking = True

    # if image_observation and action_masking:
    #     print(
    #         "+++++ ERROR action_masking and image_observation both are True; such a custom model is currently not implemented, aborting training +++++++++")
    #     ray.shutdown()

    # env = env.with_agent_groups(env, {"group1": [agents[0], agents[1]]}) # required for QMIX

    ModelCatalog.register_custom_model("visionnet", VisionNetwork_1)
    
    
    
    register_env("WarehouseEnv_3",
                      lambda config: WarehouseEnv_3(config, agent_ids=agents, max_steps=70, deterministic_game=False, obs_stacking= False,
                                image_observation=image_observation, action_masking=action_masking, num_objects=2, obstacle = False, goal_area = True, dynamic= False, mode="training",
                                 goals_coord = [(8,2),(5,2),(7,4),(2,5),(3,8)]))


    

    # creating policies
    policies = {"policy_1": PolicySpec()}  # default PPO policies, observation and action spaces are inferred from environment

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id == "agent_1":
            return "policy_1"
        else:
            return "policy_1"
        

      

    model_config = {}
    if image_observation:
        model_config = {
            "custom_model": "visionnet",
            # "custom_model_config": {
            # }
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
            "worker_index" : 2,
            },
           
        disable_env_checking = True)
        .callbacks(MyCallbacks)
        #.log_level("WARN")
        .experimental(
        _disable_preprocessor_api = True
        )
        .resources(
            num_cpus_per_worker = 1,
            num_gpus_per_worker = 0.25,
            num_gpus = 1
            )
        .rollouts(
            rollout_fragment_length = 750,
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


    run_config=RunConfig(
        name="70_steps_goal_area_0.3",
        stop = {"timesteps_total": 10000000},
        checkpoint_config=ray.air.CheckpointConfig(checkpoint_frequency=2500,
                                                    checkpoint_at_end=True)
    
    )

    tuner = tune.Tuner(
        "PPO",
        #metric="episode_len_mean",
        run_config=run_config,
        param_space=configuration.to_dict(),
    )

    #print("best hyperparameters: ", results.best_config)

    results = tuner.fit()

    ray.shutdown()

if __name__ == "__main__":
    train_h_agent_agent_ppo()
    
