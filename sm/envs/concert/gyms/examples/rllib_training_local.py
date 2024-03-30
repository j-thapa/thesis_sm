from ray.rllib import SampleBatch
from ray.rllib.algorithms.qmix import QMixConfig
from ray.rllib.evaluation import compute_advantages
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.examples.models.centralized_critic_models import CentralizedCriticModel
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.tf_utils import make_tf_callable
from ray.rllib.utils.tf_utils import explained_variance
import numpy as np

import concert.gyms.warehouse_env
from concert_visionnet import VisionNetwork_1
from concert.gyms.warehouse_env import WarehouseEnv_1, WarehouseEnv_2, WarehouseEnv_3
from callbacks import MyCallbacks, MyCallbacks_singleAgent
from gym.wrappers import FlattenObservation
from gym.spaces import Tuple

#import concert.gyms
#import concert.gridworld.items

from concert.gyms.find_goal_env import FindGoalEnv_1
from concert.gridworld.items import MoveAgent
from concert.gyms.warehouse_env import FlattenedVectorObsEnv
import ray
import ray.rllib.agents.ppo as ppo
from ray import tune, air
from ray.rllib.examples.models.action_mask_model import ActionMaskModel
#from ray.rllib.algorithms.qmix import QMixConfig

def train_one_agent():
    #import configuration
    from ray.tune.registry import register_env
    import tensorflow as tf
    ray.shutdown()
    ray.init(local_mode=True) # for debugging: local_mode=True
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
            "num_workers": 4,
            "train_batch_size": 1600,
            "rollout_fragment_length": 400,
            "batch_mode": "complete_episodes",
            "sgd_minibatch_size": 128,
            #"num_gpus_per_worker": 0.8,
            "num_cpus_per_worker": 1,
            "framework": "tfe", #tf for tensorflow, tfe for tensorflow eager (allows debugging rllib code)
            "eager_tracing": False, # if True: allows debugging rllib code
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
            "training_iteration": 100,
            #"timesteps_total": 1000000
            },
        reuse_actors=False,  # False is required for the tune grid_search to run
        config=configuration,
        # checkpoint_freq=25,
        checkpoint_at_end=True
    )

    ray.shutdown()

def train_two_agents_qmix():
    """
    NOTE: use qmix_test.py for testing qmix;

    the code follows the example in https://github.com/ray-project/ray/blob/master/rllib/algorithms/qmix/qmix.py as of may 24 2022; however, at that time
    the example was not compatible with ray's latest wheel, so the code did not work;
    """
    ray.shutdown()
    ray.init(local_mode=True)  # for debugging: local_mode=True
    # ray.init(include_dashboard=True)
    # ray.init()

    #agents = ['agent_1', 'agent_2']

    env = WarehouseEnv_2({}) # this is just used to extract observation and action space
    #env = env.with_agent_groups(env, {"group1": [agents[0], agents[1]]}) # required for QMIX
    grouping = {
        "group_1": ['agent_1', 'agent_2'],
    }
    tune.register_env(
        "WarehouseEnv",
        lambda config: WarehouseEnv_2(config, max_steps=400, deterministic_game=False, num_objects=2).with_agent_groups(grouping,
                                                                                                 obs_space=Tuple((env.observation_space, env.observation_space)),
                                                                                                 act_space=Tuple((env.action_space, env.action_space)),
                                                                                                 )
    )
    ModelCatalog.register_custom_model("visionnet", VisionNetwork_1)

    # policies = {agent: (None, env.observation_space, env.action_space, {}) for agent in env._agent_ids}

    # for debugging
    # print("environment created; environment observation space shape: {}, action space shape: {}".format(env.observation_space.shape, env.action_space.shape))

    config = QMixConfig()
    # Update the config object.
    #config.training(lr=tune.grid_search([0.001, 0.0001]), optim_alpha=0.97)
    config.training(mixer="qmix", train_batch_size=32)
    config.environment(env="WarehouseEnv",
                       env_config={
                           "reward_game_success": 1.0,  # tune.uniform(0.5, 1.5), # default: 1.0
                           "reward_each_action": -0.01, #tune.uniform(-0.1, -0.01),  # default: -0.01
                           "reward_illegal_move": -0.1,  # default: -0.05
                       },
                       disable_env_checking=True)
    """
        if image_observation:
        config.training(model={
            "custom_model": "visionnet",
        })
    if action_masking:
        config.training(model={
            "custom_model": ActionMaskModel,
            "custom_model_config": {
                "no-masking": False,
            }
        })
    """

    #config.framework(framework_str="torch") #QMIX requires pytorch
    # config.training(log_level="DEBUG") # throws exception
    config.rollouts(num_rollout_workers=0, rollout_fragment_length=500, batch_mode="complete_episodes")
    config.evaluation(evaluation_config={"explore": False})
    config.exploration(
        exploration_config={
            "final_epsilon": 0.0,
        }
    )

    stop = {
        #"episode_reward_mean": args.stop_reward,
        #"timesteps_total": args.stop_timesteps,
        "training_iteration": 100,
    }

    #tune.run("QMIX", stop={"training_iteration": 10}, config=config.to_dict())
    results = tune.Tuner(
        "QMIX",
        run_config=air.RunConfig(stop=stop, verbose=2),
        param_space=config.to_dict(),
    ).fit()

    ray.shutdown()


def train_two_agents_ppo():
    ray.shutdown()
    ray.init(local_mode=True)  # for debugging: local_mode=True
    # ray.init(include_dashboard=True)
    # ray.init()

    agents = ['agent_1', 'agent_2']
    image_observation = False
    action_masking = True
    if image_observation and action_masking:
        print(
            "+++++ ERROR action_masking and image_observation both are True; such a custom model is currently not implemented, aborting training +++++++++")
        ray.shutdown()

    # env = env.with_agent_groups(env, {"group1": [agents[0], agents[1]]}) # required for QMIX

    tune.register_env("WarehouseEnv_2", lambda config: WarehouseEnv_2(config, agent_ids=agents, max_steps=400, deterministic_game=False, image_observation=image_observation,
                         action_masking=action_masking, num_objects=2, seed=11))
    ModelCatalog.register_custom_model("visionnet", VisionNetwork_1)

    # dummy_env = WarehouseEnv_2({}, agent_ids=agents) # just used for extraction of observation and action spaces when creating policies

    # deprecated way of creating policies
    # policies = {agent: (None, dummy_env.observation_space, dummy_env.action_space, {}) for agent in dummy_env._agent_ids}

    """
    new way of creating policies (from https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent_cartpole.py)
    # Each policy can have a different configuration (including custom model).
    def gen_policy(i):
        config = {
            "model": {
                "custom_model": ["model1", "model2"][i % 2],
            },
            "gamma": random.choice([0.95, 0.99]),
        }
        return PolicySpec(config=config)

    # Setup PPO with an ensemble of `num_policies` different policies.
    policies = {"policy_{}".format(i): gen_policy(i) for i in range(args.num_policies)}
    policy_ids = list(policies.keys())

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        pol_id = random.choice(policy_ids)
        return
    # endof new way for creating policies
    """

    # creating policies
    policies = {"policy_1": PolicySpec(), "policy_2": PolicySpec()} # default PPO policies, observation and action spaces are inferred from environment

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        #return "policy_1" # both agents use the same policy: "parameter sharing" approach
        if agent_id == 'agent_1':
            return 'policy_1'
        else:
            return 'policy_2'
    # for debugging
    # print("environment created; environment observation space shape: {}, action space shape: {}".format(env.observation_space.shape, env.action_space.shape))

    model_config = {}
    if image_observation:
        model_config = {
            "custom_model": "visionnet",
            "custom_model_config": {
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
        "env": "WarehouseEnv_2",
        "env_config": {
            "reward_game_success": 1.0,
            "reward_each_action": -0.0254,
            "reward_illegal_action": -0.1,
        },
        "log_level": "WARN",
        "seed": 42, # for more information on why seeding is a good idea: https://docs.ray.io/en/latest/tune/faq.html#how-can-i-reproduce-experiments; https://github.com/ray-project/ray/blob/master/rllib/examples/deterministic_training.py        # TODO include "seed" parameter in hyperparameter search
        # "num_gpus": 0.2, ### gpu share assigned to local trainer
        "num_workers": 4,
        "train_batch_size": 2000,
        "rollout_fragment_length": 500,
        "batch_mode": "complete_episodes",
        "sgd_minibatch_size": 2000,
        # "num_gpus_per_worker": 0.8,
        "num_cpus_per_worker": 1,
        "disable_env_checking": True,
        "framework": "tfe",  # tf for tensorflow, tfe for tensorflow eager (allows debugging rllib code)
        "eager_tracing": True,  # if True: allows debugging rllib code
        # "lr": 0.00005,
        # "lr": tune.grid_search([0.0001, 0.00005]),
        # "clip_param": tune.grid_search([0.3, 0.25, 0.2, 0.15, 0.1]), decreasing values produce ever worse rewards, see PPO_2021-07-06_12-14-57
        # "lambda": tune.grid_search([1.0, 0.975, 0.95]), # decreasing values produce ever worse rewards, see PPO_2021-07-06_10-51-14
        "entropy_coeff": 0.015, # tune.grid_search([0.0, 0.025, 0.05, 0.075, 0.1]), # increasing values produce ever worse rewards, see PPO_2021-07-05_17-11-12
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
        ppo.PPOTrainer,
        stop={
            # "episodes_total": 100,
            "training_iteration": 5,
            # "timesteps_total": 1000000
        },
        reuse_actors=False,  # False is required for the tune grid_search to run
        config=configuration,
        # checkpoint_freq=25,
        checkpoint_at_end=True,
        num_samples=3
    )

    ray.shutdown()

def train_two_agents_ppo_cc():


    """
    PPO extended with centralised critic (cc); cc code from ray/rllib/examples/centralized_critic.py
    """
    OPPONENT_OBS = "opponent_obs"
    OPPONENT_ACTION = "opponent_action"

    class CentralizedValueMixin:
        """Add method to evaluate the central value function from the model."""

        def __init__(self):
            if self.config["framework"] != "torch":
                self.compute_central_vf = make_tf_callable(self.get_session())(
                    self.model.central_value_function
                )
            else:
                self.compute_central_vf = self.model.central_value_function

    # Grabs the opponent obs/act and includes it in the experience train_batch,
    # and computes GAE using the central vf predictions.
    def centralized_critic_postprocessing(
            policy, sample_batch, other_agent_batches=None, episode=None
    ):
        pytorch = policy.config["framework"] == "torch"
        if (pytorch and hasattr(policy, "compute_central_vf")) or (
                not pytorch and policy.loss_initialized()
        ):
            assert other_agent_batches is not None
            [(_, opponent_batch)] = list(other_agent_batches.values())

            # also record the opponent obs and actions in the trajectory
            sample_batch[OPPONENT_OBS] = opponent_batch[SampleBatch.CUR_OBS]
            sample_batch[OPPONENT_ACTION] = opponent_batch[SampleBatch.ACTIONS]

            # overwrite default VF prediction with the central VF
            sample_batch[SampleBatch.VF_PREDS] = convert_to_numpy(
                policy.compute_central_vf(
                    sample_batch[SampleBatch.CUR_OBS],
                    sample_batch[OPPONENT_OBS],
                    sample_batch[OPPONENT_ACTION],
                )
            )
        else:
            # Policy hasn't been initialized yet, use zeros.
            sample_batch[OPPONENT_OBS] = np.zeros_like(sample_batch[SampleBatch.CUR_OBS])
            sample_batch[OPPONENT_ACTION] = np.zeros_like(sample_batch[SampleBatch.ACTIONS])
            sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
                sample_batch[SampleBatch.REWARDS], dtype=np.float32
            )

        completed = sample_batch["dones"][-1]
        if completed:
            last_r = 0.0
        else:
            last_r = sample_batch[SampleBatch.VF_PREDS][-1]

        train_batch = compute_advantages(
            sample_batch,
            last_r,
            policy.config["gamma"],
            policy.config["lambda"],
            use_gae=policy.config["use_gae"],
        )
        return train_batch

    # Copied from PPO but optimizing the central value function.
    def loss_with_central_critic(policy, base_policy, model, dist_class, train_batch):
        # Save original value function.
        vf_saved = model.value_function

        # Calculate loss with a custom value function.
        model.value_function = lambda: policy.model.central_value_function(
            train_batch[SampleBatch.CUR_OBS],
            train_batch[OPPONENT_OBS],
            train_batch[OPPONENT_ACTION],
        )
        policy._central_value_out = model.value_function()
        loss = base_policy.loss(model, dist_class, train_batch)

        # Restore original value function.
        model.value_function = vf_saved

        return loss

    def central_vf_stats(policy, train_batch):
        # Report the explained variance of the central value function.
        return {
            "vf_explained_var": explained_variance(
                train_batch[Postprocessing.VALUE_TARGETS], policy._central_value_out
            )
        }

    def get_ccppo_policy(base):
        class CCPPOTFPolicy(CentralizedValueMixin, base):
            def __init__(self, observation_space, action_space, config):
                base.__init__(self, observation_space, action_space, config)
                CentralizedValueMixin.__init__(self)

            @override(base)
            def loss(self, model, dist_class, train_batch):
                # Use super() to get to the base PPO policy.
                # This special loss function utilizes a shared
                # value function defined on self, and the loss function
                # defined on PPO policies.
                return loss_with_central_critic(
                    self, super(), model, dist_class, train_batch
                )

            @override(base)
            def postprocess_trajectory(
                    self, sample_batch, other_agent_batches=None, episode=None
            ):
                return centralized_critic_postprocessing(
                    self, sample_batch, other_agent_batches, episode
                )

            @override(base)
            def stats_fn(self, train_batch: SampleBatch):
                stats = super().stats_fn(train_batch)
                stats.update(central_vf_stats(self, train_batch))
                return stats

        return CCPPOTFPolicy

    CCPPOStaticGraphTFPolicy = get_ccppo_policy(PPOTF1Policy)
    CCPPOEagerTFPolicy = get_ccppo_policy(PPOTF2Policy)

    class CCPPOTorchPolicy(CentralizedValueMixin, PPOTorchPolicy):
        def __init__(self, observation_space, action_space, config):
            PPOTorchPolicy.__init__(self, observation_space, action_space, config)
            CentralizedValueMixin.__init__(self)

        @override(PPOTorchPolicy)
        def loss(self, model, dist_class, train_batch):
            return loss_with_central_critic(self, super(), model, dist_class, train_batch)

        @override(PPOTorchPolicy)
        def postprocess_trajectory(
                self, sample_batch, other_agent_batches=None, episode=None
        ):
            return centralized_critic_postprocessing(
                self, sample_batch, other_agent_batches, episode
            )

    class CCTrainer(PPO):
        @override(PPO)
        def get_default_policy_class(self, config):
            if config["framework"] == "torch":
                return CCPPOTorchPolicy
            elif config["framework"] == "tf":
                return CCPPOStaticGraphTFPolicy
            else:
                return CCPPOEagerTFPolicy

    # ++++++++++++  endof centralised critic code
    ray.shutdown()
    ray.init(local_mode=True)  # for debugging: local_mode=True

    agents = ['agent_1', 'agent_2']
    image_observation = False
    action_masking = True
    if image_observation and action_masking:
        print(
            "+++++ ERROR action_masking and image_observation both are True; such a custom model is currently not implemented, aborting training +++++++++")
        ray.shutdown()

    tune.register_env("WarehouseEnv_2",
                      lambda config: WarehouseEnv_2(config, agent_ids=agents, max_steps=400, deterministic_game=True,
                                                    image_observation=image_observation,
                                                    action_masking=action_masking, num_objects=2))
    ModelCatalog.register_custom_model("visionnet", VisionNetwork_1)
    ModelCatalog.register_custom_model("cc_model", CentralizedCriticModel)

    # creating policies
    policies = {"policy_1": PolicySpec(),
                "policy_2": PolicySpec()}  # default PPO policies, observation and action spaces are inferred from environment

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return "policy_1"  # both agents use the same policy: "parameter sharing" approach

    # for debugging
    # print("environment created; environment observation space shape: {}, action space shape: {}".format(env.observation_space.shape, env.action_space.shape))

    model_config = {}
    if image_observation:
        model_config = {
            "custom_model": "visionnet",
            "custom_model_config": {
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
        "env": "WarehouseEnv_2",
        "env_config": {
            "reward_game_success": 1.0,
            "reward_each_action": -0.01,
            "reward_illegal_move": -0.05,
        },
        "log_level": "DEBUG",
        # "num_gpus": 0.2, ### gpu share assigned to local trainer
        "num_workers": 0,
        "train_batch_size": 10,
        "rollout_fragment_length": 10,
        "batch_mode": "complete_episodes",
        "sgd_minibatch_size": 10,
        # "num_gpus_per_worker": 0.8,
        "num_cpus_per_worker": 1,
        "disable_env_checking": True,
        "framework": "tfe",  # tf for tensorflow, tfe for tensorflow eager (allows debugging rllib code)
        "eager_tracing": True,  # if True: allows debugging rllib code
        # "lr": 0.00005,
        # "lr": tune.grid_search([0.0001, 0.00005]),
        # "clip_param": tune.grid_search([0.3, 0.25, 0.2, 0.15, 0.1]), decreasing values produce ever worse rewards, see PPO_2021-07-06_12-14-57
        # "lambda": tune.grid_search([1.0, 0.975, 0.95]), # decreasing values produce ever worse rewards, see PPO_2021-07-06_10-51-14
        # "entropy_coeff": tune.grid_search([0.0, 0.025, 0.05, 0.075, 0.1]), # increasing values produce ever worse rewards, see PPO_2021-07-05_17-11-12
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
            # "observation_fn": central_critic_observer,
        },
        "model": {
            "custom_model": "cc_model",
        },
        # "num_sgd_iter": 1,
        # "vf_loss_coeff": 0.0001,
        "callbacks": MyCallbacks,
        # "output": "./episode_traces/", # logs episode traces to the experiment's ray results folder
        # "output_compress_columns": [],
    }

    results = tune.run(
        CCTrainer,
        stop={
            # "episodes_total": 100,
            "training_iteration": 10,
            # "timesteps_total": 1000000
        },
        reuse_actors=False,  # False is required for the tune grid_search to run
        config=configuration,
        # checkpoint_freq=25,
        checkpoint_at_end=True
    )

    ray.shutdown()

def train_h_agent_agent_ppo():
    ray.shutdown()
    ray.init(local_mode=True)  # for debugging: local_mode=True

    # alg = BayesOptSearch(metric="episode_reward_mean", mode="max")

    agents = ['agent_1']
    image_observation = False
    action_masking = True
    if image_observation and action_masking:
        print(
            "+++++ ERROR action_masking and image_observation both are True; such a custom model is currently not implemented, aborting training +++++++++")
        ray.shutdown()

    # env = env.with_agent_groups(env, {"group1": [agents[0], agents[1]]}) # required for QMIX

    tune.register_env("WarehouseEnv_3",
                      lambda config: WarehouseEnv_3(config, agent_ids=agents, max_steps=400, deterministic_game=False,
                                                    image_observation=image_observation,
                                                    action_masking=action_masking, num_objects=1))


    ModelCatalog.register_custom_model("visionnet", VisionNetwork_1)

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
            "custom_model_config": {
            }
        }

    if action_masking:
        model_config = {
            "custom_model": ActionMaskModel,
            "custom_model_config": {
                "no-masking": False,
            },
            #"use_lstm": True,
            #"max_seq_len": 5,
            #"lstm_use_prev_action": True,
        }

    configuration = {
        "env": "WarehouseEnv_3",
        "env_config": {
            "reward_game_success": 1.0, # default: 1.0
            "reward_each_action": -0.01, # default: -0.01
            "reward_illegal_action": -0.05, # default: -0.05
            },
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
        #choosing red values entropy and vf
        "entropy_coeff": 0.019, #0.015 # tune.uniform(lower,upper) triggers hyperparameter search (a simple random search)
        # "lr": 0.00005,
        # "lr": tune.grid_search([0.0001, 0.00005]),
        "use_gae": True,
        "vf_loss_coeff": 0.82,
        "lambda": 0.87,
        "clip_param":  0.16, #decreasing values produce ever worse rewards, see PPO_2021-07-06_12-14-57
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
        ppo.PPOTrainer,
        name= "random_object_on_the_goal_agent_h_agent",
        metric="episode_len_mean",
        # resume=True
        # #mode="min",
        stop={
            # "episodes_total": 100,
            "training_iteration": 100,
            # "timesteps_total": 1000000
        },
        #reuse_actors=False,  # False is required for the tune grid_search to run
        config=configuration,
        checkpoint_freq=500,
        checkpoint_at_end=True,
        #search_alg=alg,
        #num_samples=30 # number of samples in the hyperparameter search space
    )

    #print("best hyperparameters: ", results.best_config)

    ray.shutdown()

def train_only_robot():
    ray.shutdown()
    ray.init(local_mode=True)  # for debugging: local_mode=True
    # ray.init(include_dashboard=True)
    # ray.init()

    image_observation = False
    action_masking = True
    if image_observation and action_masking:
        print(
            "+++++ ERROR action_masking and image_observation both are True; such a custom model is currently not implemented, aborting training +++++++++")
        ray.shutdown()

    # env = env.with_agent_groups(env, {"group1": [agents[0], agents[1]]}) # required for QMIX

    tune.register_env("WarehouseEnv_3", lambda config: WarehouseEnv_3(config, max_steps=400, deterministic_game=False, image_observation=image_observation,
                                                                              action_masking=action_masking, num_objects=1, seed=11))
    ModelCatalog.register_custom_model("visionnet", VisionNetwork_1)

    # dummy_env = WarehouseEnv_2({}, agent_ids=agents) # just used for extraction of observation and action spaces when creating policies

    # deprecated way of creating policies
    # policies = {agent: (None, dummy_env.observation_space, dummy_env.action_space, {}) for agent in dummy_env._agent_ids}

    """
    new way of creating policies (from https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent_cartpole.py)
    # Each policy can have a different configuration (including custom model).
    def gen_policy(i):
        config = {
            "model": {
                "custom_model": ["model1", "model2"][i % 2],
            },
            "gamma": random.choice([0.95, 0.99]),
        }
        return PolicySpec(config=config)

    # Setup PPO with an ensemble of `num_policies` different policies.
    policies = {"policy_{}".format(i): gen_policy(i) for i in range(args.num_policies)}
    policy_ids = list(policies.keys())

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        pol_id = random.choice(policy_ids)
        return
    # endof new way for creating policies
    """

    # creating policies
    #policies = {"policy_1": PolicySpec(), "policy_2": PolicySpec()} # default PPO policies, observation and action spaces are inferred from environment

    #def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    #    return "policy_1" # both agents use the same policy: "parameter sharing" approach

    # for debugging
    # print("environment created; environment observation space shape: {}, action space shape: {}".format(env.observation_space.shape, env.action_space.shape))

    model_config = {}
    if image_observation:
        model_config = {
            "custom_model": "visionnet",
            "custom_model_config": {
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
        "env": "WarehouseEnv_3",
        "env_config": {
            "reward_game_success": 1.0,
            "reward_each_action": -0.0254,
            "reward_illegal_action": -0.1,
        },
        "log_level": "DEBUG",
        "seed": 42, # for more information on why seeding is a good idea: https://docs.ray.io/en/latest/tune/faq.html#how-can-i-reproduce-experiments; https://github.com/ray-project/ray/blob/master/rllib/examples/deterministic_training.py        # TODO include "seed" parameter in hyperparameter search
        # "num_gpus": 0.2, ### gpu share assigned to local trainer
        "num_workers": 0,
        "train_batch_size": 2000,
        "rollout_fragment_length": 2000,
        "batch_mode": "complete_episodes",
        "sgd_minibatch_size": 128,
        # "num_gpus_per_worker": 0.8,
        "num_cpus_per_worker": 1,
        "disable_env_checking": True,
        "framework": "tfe",  # tf for tensorflow, tfe for tensorflow eager (allows debugging rllib code)
        "eager_tracing": True,  # if True: allows debugging rllib code
        # "lr": 0.00005,
        # "lr": tune.grid_search([0.0001, 0.00005]),
        # "clip_param": tune.grid_search([0.3, 0.25, 0.2, 0.15, 0.1]), decreasing values produce ever worse rewards, see PPO_2021-07-06_12-14-57
        # "lambda": tune.grid_search([1.0, 0.975, 0.95]), # decreasing values produce ever worse rewards, see PPO_2021-07-06_10-51-14
        "entropy_coeff": 0.015, # tune.grid_search([0.0, 0.025, 0.05, 0.075, 0.1]), # increasing values produce ever worse rewards, see PPO_2021-07-05_17-11-12
        #"multiagent": {
        #    "policies": policies,
        #    "policy_mapping_fn": policy_mapping_fn,
        #    # "observation_fn": central_critic_observer,
        #},
        "model": model_config,
        # "num_sgd_iter": 1,
        # "vf_loss_coeff": 0.0001,
        "callbacks": MyCallbacks_singleAgent,
        #"output": "./episode_traces/", # logs episode traces to the experiment's ray results folder
        #"output_compress_columns": [],
    }

    results = tune.run(
        ppo.PPOTrainer,
        stop={
            # "episodes_total": 100,
            "training_iteration": 5,
            # "timesteps_total": 1000000
        },
        reuse_actors=False,  # False is required for the tune grid_search to run
        config=configuration,
        # checkpoint_freq=25,
        checkpoint_at_end=True,
        #num_samples=3
    )

    ray.shutdown()

if __name__ == "__main__":
    train_h_agent_agent_ppo()