"""An example of implementing a centralized critic with ObservationFunction.
The advantage of this approach is that it's very simple and you don't have to
change the algorithm at all -- just use callbacks and a custom model.
However, it is a bit less principled in that you have to change the agent
observation spaces to include data that is only used at train time.
See also: centralized_critic.py for an alternative approach that instead
modifies the policy to add a centralized value function.
"""

import numpy as np
import ray
from gymnasium.spaces import Dict, Discrete, Box
import argparse
import os

from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.examples.models.centralized_critic_models import (
    YetAnotherCentralizedCriticModel,
    YetAnotherTorchCentralizedCriticModel,
)
from ray.rllib.examples.env.two_step_game import TwoStepGame
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune import register_env
from concert.gyms.examples.callbacks import MyCallbacks
from concert.gyms.warehouse_env import WarehouseEnv_2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="tf",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=100, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=7.99, help="Reward at which we stop training."
)


class CentralisedCriticModel_for_warehouseEnv(YetAnotherCentralizedCriticModel):
    # no action masking
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CentralisedCriticModel_for_warehouseEnv, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.action_model = FullyConnectedNetwork(
            Box(low=0, high=1, shape=(700,)), # WarehouseEnv_2 observation has shape 10x10x7
            action_space,
            num_outputs,
            model_config,
            name + "_action",
        )

class CCModel_warehouseEnv_actionmasking(YetAnotherCentralizedCriticModel):
    # with action masking
    # arg obs_space reflects the output from the observer function (dict with keys: own_obs, opponent_obs, opponent_actions
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
                isinstance(orig_space, Dict)
                and "action_mask" in orig_space.spaces['own_obs'].spaces
                and "action_mask" in orig_space.spaces['opponent_obs'].spaces
                and "observations" in orig_space.spaces['own_obs'].spaces
                and "observations" in orig_space.spaces['opponent_obs'].spaces
        )

        super(YetAnotherCentralizedCriticModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        self.action_model = FullyConnectedNetwork(
            orig_space["own_obs"]["observations"], # the action model only takes the agent's observation (without action mask) as input
            action_space,
            num_outputs,
            model_config,
            name + "_action",
        )

        self.value_model = FullyConnectedNetwork(
            obs_space, action_space, 1, model_config, name + "_vf"
        )

        # disable action masking --> will likely lead to invalid actions
        self.no_masking = model_config["custom_model_config"].get("no_masking", False)

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["own_obs"]["action_mask"]

        # Compute the unmasked logits.

        logits, _ = self.action_model({"obs": input_dict["obs"]["own_obs"]["observations"]})

        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        masked_logits = logits + inf_mask

        self._value_out, _ = self.value_model(
            {"obs": input_dict["obs_flat"]}, state, seq_lens # feed observations and action masks of both agents, and the opponent agent's actions
        )
        # Return masked logits.
        return masked_logits, state


class FillInActions_twostepgame(DefaultCallbacks):
    """Fills in the opponent actions info in the training batches."""

    def on_postprocess_trajectory(
        self,
        worker,
        episode,
        agent_id,
        policy_id,
        policies,
        postprocessed_batch,
        original_batches,
        **kwargs
    ):
        to_update = postprocessed_batch[SampleBatch.CUR_OBS]
        other_id = 1 if agent_id == 0 else 0
        action_encoder = ModelCatalog.get_preprocessor_for_space(Discrete(2))

        # set the opponent actions into the observation
        _, opponent_batch = original_batches[other_id]
        opponent_actions = np.array(
            [action_encoder.transform(a) for a in opponent_batch[SampleBatch.ACTIONS]]
        )
        to_update[:, -2:] = opponent_actions


class FillInActions_warehouseenv(MyCallbacks):
    """Fills in the opponent actions info in the training batches."""

    def on_postprocess_trajectory(
        self,
        worker,
        episode,
        agent_id,
        policy_id,
        policies,
        postprocessed_batch,
        original_batches,
        **kwargs
    ):
        to_update = postprocessed_batch[SampleBatch.CUR_OBS]
        other_id = 'agent_2' if agent_id == 'agent_1' else 'agent_1'
        action_encoder = ModelCatalog.get_preprocessor_for_space(Discrete(7))

        # set the opponent actions into the observation
        _, opponent_batch = original_batches[other_id]
        opponent_actions = np.array(
            [action_encoder.transform(a) for a in opponent_batch[SampleBatch.ACTIONS]]
        )
        to_update[:, 0:7] = opponent_actions # the other agent's actions are to be inserted in slice 0:7 of the agent's total observation
        print()

class FillInActions_warehouseenv_actionmasking(MyCallbacks):
    """Fills in the opponent actions info in the training batches."""

    def on_postprocess_trajectory(
        self,
        worker,
        episode,
        agent_id,
        policy_id,
        policies,
        postprocessed_batch,
        original_batches,
        **kwargs
    ):
        to_update = postprocessed_batch[SampleBatch.CUR_OBS]
        other_id = 'agent_2' if agent_id == 'agent_1' else 'agent_1'
        action_encoder = ModelCatalog.get_preprocessor_for_space(Discrete(7))

        # set the opponent actions into the observation
        if other_id in original_batches.keys():
            _, opponent_batch = original_batches[other_id]
            opponent_actions = np.array(
                [action_encoder.transform(a) for a in opponent_batch[SampleBatch.ACTIONS]]
            )
            if opponent_actions.shape[0] < to_update.shape[0]:
                to_update[0:(opponent_actions.shape[0]), 0:7] = opponent_actions # the other agent's actions are to be inserted in slice 0:7 of the agent's total observation
            else:
                to_update[0:to_update.shape[0], 0:7] = opponent_actions[0:to_update.shape[0], 0:7]
        else:
            # no other agent, do nothing
            pass
        # print()


def run_twostepgame():
    global args, config, stop
    args = parser.parse_args()
    ModelCatalog.register_custom_model(
        "cc_model",
        YetAnotherTorchCentralizedCriticModel
        if args.framework == "torch"
        else YetAnotherCentralizedCriticModel,
    )

    def central_critic_observer(agent_obs, **kw):
        """Rewrites the agent obs to include opponent data for training."""

        new_obs = {
            0: {
                "own_obs": agent_obs[0],
                "opponent_obs": agent_obs[1],
                "opponent_action": 0,  # filled in by FillInActions
            },
            1: {
                "own_obs": agent_obs[1],
                "opponent_obs": agent_obs[0],
                "opponent_action": 0,  # filled in by FillInActions
            },
        }
        return new_obs

    action_space = Discrete(2)
    observer_space = Dict(
        {
            "own_obs": Discrete(6),
            # These two fields are filled in by the CentralCriticObserver, and are
            # not used for inference, only for training.
            "opponent_obs": Discrete(6),
            "opponent_action": Discrete(2),
        }
    )
    env = register_env("TwoStepGame", lambda config: TwoStepGame(config))
    config = (
        PPOConfig()
            .environment(env="TwoStepGame")
            .framework(args.framework)
            .rollouts(batch_mode="complete_episodes", num_rollout_workers=0)
            .callbacks(FillInActions_twostepgame)
            .training(model={"custom_model": "cc_model"})
            .multi_agent(
            policies={
                "pol1": (None, observer_space, action_space, {}),
                "pol2": (None, observer_space, action_space, {}),
            },
            policy_mapping_fn=lambda aid, **kwargs: "pol1" if aid == 0 else "pol2",
            observation_fn=central_critic_observer,
        )
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )
    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }
    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop=stop, verbose=1),
    )
    results = tuner.fit()
    if args.as_test:
        check_learning_achieved(results, args.stop_reward)


def run_warehouseenv_2():
    ray.shutdown()
    ray.init(local_mode=True)

    ModelCatalog.register_custom_model("cc_model", CCModel_warehouseEnv_actionmasking)

    agents = ['agent_1', 'agent_2']
    env = WarehouseEnv_2({}, action_masking=True)
    tune.register_env("WarehouseEnv_2",
                      lambda config: WarehouseEnv_2(config, agent_ids=agents, max_steps=400, deterministic_game=False,
                                                    image_observation=False,
                                                    action_masking=True, num_objects=2, random_items=[1,0,0]))
    def central_critic_observer(agent_obs:dict, **kw):
        """Rewrites the agent obs to include opponent data for training."""
        if "agent_1" in agent_obs.keys() and "agent_2" in agent_obs.keys():
            new_obs = {
                    'agent_1': {
                    "own_obs": agent_obs['agent_1'],
                    "opponent_obs":agent_obs['agent_2'], # without action masking, both agents do have the same global observation (without action masking), so we just feed a dummy observation for opponent_obs
                    "opponent_action": 6,  # irrelevant, filled in by callback FillInActions
                },
                'agent_2': {
                    "own_obs": agent_obs['agent_2'],
                    "opponent_obs": agent_obs['agent_1'],
                    "opponent_action": 6,  # filled in by FillInActions
                },
            }
        elif "agent_1" in agent_obs.keys() and not "agent_2" in agent_obs.keys():
            new_obs = {
                'agent_1': {
                    "own_obs": agent_obs['agent_1'],
                    "opponent_obs": agent_obs['agent_1'], # no agent_2 anymore, but still keep the observation space consistent
                    "opponent_action": 6,  # irrelevant, filled in by callback FillInActions
                },
            }
        elif not "agent_1" in agent_obs.keys() and "agent_2" in agent_obs.keys():
            new_obs = {
                'agent_2': {
                    "own_obs": agent_obs['agent_2'],
                    "opponent_obs": agent_obs['agent_2'],
                    # no agent_1 anymore, but still keep the observation space consistent
                    "opponent_action": 6,  # irrelevant, filled in by callback FillInActions
                },
            }
        return new_obs

    action_space = Discrete(7)
    observer_space = Dict(
        {
            "own_obs": env.observation_space,
            "opponent_obs": env.observation_space, # env.observation_space,
            "opponent_action": Discrete(7),
        }
    )

    config = (
        PPOConfig()
            .environment(env="WarehouseEnv_2", env_config={
                "reward_game_success": 1.0,
                "reward_each_action": -0.01,
                "reward_illegal_action": -0.1,
                }, disable_env_checking=True)
            .framework("tfe", eager_tracing=True)
            .rollouts(batch_mode="complete_episodes", num_rollout_workers=0, rollout_fragment_length=500)
            .callbacks(FillInActions_warehouseenv_actionmasking)
            .training(model={"custom_model": "cc_model"}, train_batch_size=500, entropy_coeff=0.)
            .multi_agent(
            policies={
                "agent_1": (None, observer_space, action_space, {}),
                "agent_2": (None, observer_space, action_space, {}),
            },
            policy_mapping_fn=lambda aid, **kwargs: "agent_1" if aid == "agent_1" else "agent_2",
            observation_fn=central_critic_observer,
        )
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            .resources(num_cpus_per_worker=1)
    )
    stop = {
        "training_iteration": 50,
    }
    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop=stop, verbose=3)
    )
    results = tuner.fit()
    ray.shutdown()


if __name__ == "__main__":
    run_warehouseenv_2()