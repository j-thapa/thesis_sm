"""
test the deployment of trained policies with ray serve; see https://docs.ray.io/en/latest/serve/tutorials/rllib.html
"""
import json
import jsonpickle

from starlette.requests import Request
from ray import serve
from ray import tune
from ray.rllib.models import ModelCatalog
from concert.gyms.warehouse_env import WarehouseEnv_3
from concert.gyms.examples.concert_visionnet import VisionNetwork_1
from ray.rllib.examples.models.action_mask_model import ActionMaskModel
from ray.rllib.policy.policy import PolicySpec
import ray.rllib.agents.ppo as ppo
import requests



@serve.deployment
class ServePPOModel:
    def __init__(self, checkpoint_path) -> None:
        # Re-create the originally used config for training the model.
        image_observation = False  # whether the agent was trained using images or vectorized representations
        action_masking = True
        if image_observation:
            model_config = {
                "custom_model": "visionnet",
                "custom_model_config": {
                }
            }
        else:
            model_config = {
                "custom_model": ActionMaskModel,
                "custom_model_config": {
                    "no-masking": False,
                }
            }
        agents = ['agent_1']
        tune.register_env("WarehouseEnv",
                          lambda config: WarehouseEnv_3(config, agent_ids=agents, max_steps=400,
                                                        deterministic_game=False, obs_stacking=False,
                                                        image_observation=image_observation, mode="evaluation",
                                                        action_masking=action_masking, num_objects=1, obstacle=False,
                                                        dynamic=False, goals_coord=[(2, 4), (4, 2), (6, 5), (7, 3)]))
        ModelCatalog.register_custom_model("visionnet", VisionNetwork_1)

        # creating policies
        policies = {
            "policy_1": PolicySpec()}  # default PPO policies, observation and action spaces are inferred from environment

        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            if agent_id == "agent_1":
                return "policy_1"
            else:
                return "policy_1"
        configuration = {
            "env": "WarehouseEnv",
            "log_level": "ERROR",
            # "num_gpus": 0.2, ### gpu share assigned to local trainer
            "num_workers": 0,
            "train_batch_size": 400,
            "rollout_fragment_length": 400,
            "batch_mode": "complete_episodes",
            # "sgd_minibatch_size": 64,
            # "num_gpus_per_worker": 0.8,
            "num_cpus_per_worker": 1,
            "explore": False,
            "framework": "tf",  # tf for tensorflow, tfe for tensorflow eager
            "disable_env_checking": True,
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
            # "model": {
            #     #"dim": 45,
            #     #"conv_filters": [[16, [5, 5], 1]], # [32, [3, 3], 2], [512, [3, 3], 1]],
            #     #"conv_activation": "relu",
            #     #"post_fcnet_hiddens": [256],
            #     #"post_fcnet_activation": "relu",
            #     "custom_model": "visionnet",
            #     "custom_model_config": {
            #     #    "conv_filters": [[16, [5, 5], 1], [32, [5, 5], 1]], # [128, [5, 5], 1]],
            #     #    "conv_activation": "relu",
            #     #    "post_fcnet_hiddens": [256],
            #     #    "post_fcnet_activation": "relu",
            #     #    "no_final_linear": True,
            #     #    #"vf_share_layers":,
            #     }
            # },
            "model": model_config,
            # "num_sgd_iter": 1,
            # "vf_loss_coeff": 0.0001,
            # "callbacks": MyCallbacks,
        }

        # Build the Algorithm instance using the config.
        self.algorithm = ppo.PPOTrainer(config=configuration, env="WarehouseEnv")

        # Restore the algo's state from the checkpoint.
        self.algorithm.restore(checkpoint_path)

    async def __call__(self, request: Request):
        json_input = await request.json()
        obs = json_input["observation"]
        obs_decoded = jsonpickle.decode(obs, keys=True)
        action = self.algorithm.compute_action(obs_decoded, policy_id="policy_1")
        return {"action": int(action)}


if __name__ == "__main__":
    # start the policy server
    ppo_model = ServePPOModel.bind('C:/Users/ahaemm/ray_results_archive/PPO_2023-03-06_14-10-33/PPO_WarehouseEnv_3_4725a_00000_0_2023-03-06_14-10-33/checkpoint_003305/checkpoint-3305')
    serve.run(ppo_model)

    # create the environment for querying the served ppo model
    env_config = {
        "reward_game_success": 1.0,
        "reward_each_action": -0.01,
        "reward_move_failure": -0.05,
        "heuristic_agent_randomness": 0.0,
        "dynamic_obstacle_randomness": 0.0,
    }

    agents = ['agent_1']
    image_observation = False
    action_masking = True
    env = WarehouseEnv_3(env_config, agent_ids=agents, max_steps=60, deterministic_game=False, obs_stacking=False,
                         image_observation=image_observation, action_masking=action_masking, num_objects=1,
                         obstacle=True, mode="evaluation",
                         dynamic=True, goals_coord=[(8, 2), (5, 2), (7, 4), (2, 5), (3, 8)])

    # interface point; in the lab demonstrator the observation is provided by the vision system
    obs = env.reset()
    print("-> Sending observation {}".format(obs["agent_1"]))
    json_obs = jsonpickle.encode(obs["agent_1"], keys=True)
    resp = requests.get(
        "http://localhost:8000/", json={"observation": json_obs}
    )
    print(f"<- Received response {resp.json()}")