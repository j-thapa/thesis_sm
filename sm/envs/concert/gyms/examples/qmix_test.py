"""
Original code from https://github.com/ray-project/ray/blob/master/rllib/examples/two_step_game.py, 11.10.2022

Modifications: use WarehouseEnv_2 as environment
"""

import argparse
from gym.spaces import Dict, Discrete, Tuple, MultiDiscrete
import logging
import os

import ray
from ray import air, tune
from ray.tune import register_env
from ray.rllib.algorithms.qmix import QMixConfig
from ray.rllib.env.multi_agent_env import ENV_STATE
# from ray.rllib.examples.env.two_step_game import TwoStepGame
from ray.rllib.policy.policy import PolicySpec
from callbacks import MyCallbacks
from ray.rllib.utils.test_utils import check_learning_achieved

from concert.gyms.warehouse_env import WarehouseEnv_2

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="tf",
    help="The DL framework specifier.",
)
parser.add_argument("--num-cpus", type=int, default=0)
parser.add_argument(
    "--mixer",
    type=str,
    default="qmix",
    choices=["qmix", "vdn", "none"],
    help="The mixer model to use.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
#parser.add_argument(
#    "--stop-iters", type=int, default=200, help="Number of iterations to train."
#)
#parser.add_argument(
#    "--stop-timesteps", type=int, default=70000, help="Number of timesteps to train."
#)
#parser.add_argument(
#    "--stop-reward", type=float, default=8.0, help="Reward at which we stop training."
#)
#parser.add_argument(
#    "--local-mode",
#    action="store_true",
#    help="Init Ray in local mode for easier debugging.",
#)

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init(num_cpus=args.num_cpus or None, local_mode=True)

    grouping = {
        "group_1": ['agent_1'],
        "group_2": ['agent_2']
    }

    env = WarehouseEnv_2({})
    register_env(
        "WarehouseEnv",
        # action_masking=False does not work with QMIX
        lambda config: WarehouseEnv_2(config, max_steps=1500, deterministic_game=False, num_objects=2, action_masking=True).with_agent_groups(
            grouping,
            obs_space=Tuple([env.observation_space]),
            act_space=Tuple([env.action_space]),
            )
    )

    config = (
        QMixConfig()
        .training(mixer=args.mixer, train_batch_size=32)
        .rollouts(num_rollout_workers=4, rollout_fragment_length=250, batch_mode="complete_episodes")
        .exploration(
            exploration_config={
                "initial_epsilon": 1.0,
                "final_epsilon": 0.01,
                "epsilon_timesteps": 10000000,
            }
        )
        .environment(
            env="WarehouseEnv",
            env_config={
                "reward_game_success": 1.0,  # tune.uniform(0.5, 1.5), # default: 1.0
                "reward_each_action": -0.01,  # tune.uniform(-0.1, -0.01),  # default: -0.01
                "reward_illegal_move": -0.1,  # default: -0.05
            },
            disable_env_checking=True
        )
        #.resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )
    config = config.to_dict()
    config["simple_optimizer"] = True # no training on multiple GPUs
    multiagent_config =  {
        "policies": {
            # the first tuple value is None -> uses default policy
            "pol_1": (None, Tuple([env.observation_space]), Tuple([env.action_space]), {"agent_id": "group_1"}),
            "pol_2": (None, Tuple([env.observation_space]), Tuple([env.action_space]), {"agent_id": "group_2"}),
        },
        "policy_mapping_fn": lambda agent_id: "pol_1" if agent_id == "group_1" else "pol_2"
    }
    config["multiagent"] = multiagent_config
    # config["callbacks"] = MyCallbacks # TODO MyCallbacks has to be adapted to work with QMIX

    stop = {
        #"episode_reward_mean": args.stop_reward,
        #"timesteps_total": args.stop_timesteps,
        "training_iteration": 10000,
    }

    trainer = tune.Tuner(
        "QMIX",
        run_config=air.RunConfig(stop=stop, verbose=3),
        param_space=config,
    )

    results = trainer.fit()

    ray.shutdown()