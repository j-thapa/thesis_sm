"""
Original code from https://github.com/ray-project/ray/blob/master/rllib/examples/two_step_game.py, 11.10.2022

Uses the MADDPG algorithm and WarehouseEnv_2 => DOES NOT WORK, as the MADDPG implementation does not support the observation
space from WarehouseEmv_2;

The two-step game from QMIX: https://arxiv.org/pdf/1803.11485.pdf
Configurations you can try:
    - normal policy gradients (PG)
    - MADDPG
    - QMIX
See also: centralized_critic.py for centralized critic PPO on this game.
"""

import argparse
from gymnasium.spaces import Dict, Discrete, Tuple, MultiDiscrete
import logging
import os

import ray
from ray import air, tune
from ray.tune import register_env
from ray.rllib.algorithms.qmix import QMixConfig
from ray.rllib.env.multi_agent_env import ENV_STATE
from ray.rllib.examples.env.two_step_game import TwoStepGame
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.test_utils import check_learning_achieved
from concert.gyms.warehouse_env import WarehouseEnv_2

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    ray.init(num_cpus=None, local_mode=True)
    tune.register_env("WarehouseEnv_2",
                      lambda config: WarehouseEnv_2(config,
                                                    agent_ids=[1, 2],
                                                    max_steps=400,
                                                    deterministic_game=False,
                                                    image_observation=False,
                                                    action_masking=False,
                                                    num_objects=2,
                                                    seed=11))
    env = WarehouseEnv_2({}, action_masking=False)
    obs_space = env.observation_space
    act_space = env.action_space
    config = {
        "env": "WarehouseEnv_2",
        "env_config": {
            "actions_are_logits": True, # FIXME this has to be implemented in WarehouseEnv_2
        },
        #"num_steps_sampled_before_learning_starts": 100,
        "multiagent": {
            "policies": {
                "pol1": PolicySpec(
                    observation_space=obs_space,
                    action_space=act_space,
                    config={"agent_id": 1},
                ),
                "pol2": PolicySpec(
                    observation_space=obs_space,
                    action_space=act_space,
                    config={"agent_id": 2},
                ),
            },
            "policy_mapping_fn": (lambda agent_id: "pol2" if agent_id == 2 else "pol1"),
        },
        "framework": "tf",
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "disable_env_checking": True,
    }

    stop = {
        #"episode_reward_mean": args.stop_reward,
        #"timesteps_total": args.stop_timesteps,
        "training_iteration": 100,
    }

    results = tune.Tuner(
        "MADDPG",
        run_config=air.RunConfig(stop=stop, verbose=3),
        param_space=config,
    ).fit()

    ray.shutdown()