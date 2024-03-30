import ray
#from ray.rllib import evaluation
from ray.rllib.agents import trainer
import ray.rllib.agents.ppo as ppo
import numpy as np
import  csv

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec

from concert.gyms.examples.test_centralised_critic import CCModel_warehouseEnv_actionmasking, \
    FillInActions_warehouseenv_actionmasking
from concert.gyms.warehouse_env import WarehouseEnv_1, WarehouseEnv_2, WarehouseEnv_3
from concert.gyms.examples.concert_visionnet import VisionNetwork_1
import matplotlib.pyplot as plt
from ray.rllib.examples.models.action_mask_model import ActionMaskModel
from gymnasium.spaces import Dict, Discrete

"""
Evaluation of 2 trained agents (multi-agent RL) by stepping through episodes;
uses either centralised critic approach, or 2 independent policy models; 

1) restore trained agent policies from checkpoint
2) create game => render
3) step through the game by calling the policy => render after each step
"""

use_CC = False # set centralised critic to be used or not, must be consistent with the trained model to be restored

ray.shutdown()
ray.init()

ModelCatalog.register_custom_model("cc_model", CCModel_warehouseEnv_actionmasking)

agents = ['agent_1', 'agent_2']
env = WarehouseEnv_2({}, action_masking=True)
tune.register_env("WarehouseEnv_2",
                  lambda config: WarehouseEnv_2(config, agent_ids=agents, max_steps=400,
                                                deterministic_game=True,
                                                image_observation=False,
                                                action_masking=True,
                                                num_objects=2,
                                                seed=11))


def central_critic_observer(agent_obs, **kw):
    """Rewrites the agent obs to include opponent data for training."""
    new_obs = {
        'agent_1': {
            "own_obs": agent_obs['agent_1'],
            "opponent_obs": agent_obs['agent_2'],
            # without action masking, both agents do have the same global observation (without action masking), so we just feed a dummy observation for opponent_obs
            "opponent_action": 2,  # irrelevant, filled in by callback FillInActions
        },
        'agent_2': {
            "own_obs": agent_obs['agent_2'],
            "opponent_obs": agent_obs['agent_1'],
            "opponent_action": 2,  # filled in by FillInActions
        },
    }
    return new_obs


action_space = Discrete(7)
observer_space = Dict(
    {
        "own_obs": env.observation_space,
        "opponent_obs": env.observation_space,  # env.observation_space,
        "opponent_action": Discrete(7),
    }
)

if use_CC:
    model_config = {
        "custom_model": "cc_model"
    }
    obs_space = observer_space
else:
    model_config = {
        "custom_model": ActionMaskModel,
        "custom_model_config": {
            "no_masking": False,
        }
    }
    obs_space = env.observation_space

config = (
    PPOConfig()
        .environment(env="WarehouseEnv_2", env_config={
            "reward_game_success": 1.0,
            "reward_each_action": -0.01,
            "reward_illegal_action": -0.1,
            },
            disable_env_checking=True)
        .framework("tf", eager_tracing=False)
        .rollouts(batch_mode="complete_episodes", num_rollout_workers=0, rollout_fragment_length=500)
        #.callbacks(FillInActions_warehouseenv_actionmasking)
        .training(model=model_config, train_batch_size=500, entropy_coeff=0.005)
        .multi_agent(
            policies={
                "agent_1": (None, obs_space, action_space, {}),
                "agent_2": (None, obs_space, action_space, {}),
                },
            policy_mapping_fn=lambda aid, **kwargs: "agent_1" if aid == "agent_1" else "agent_2",
            observation_fn=central_critic_observer,
        )
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_cpus_per_worker=1)
)

config = config.to_dict()
config["explore"] = False

evaluator = ppo.PPOTrainer(config=config, env="WarehouseEnv_2")

# gans
#evaluator.restore('D:/Profactor/marl/concert/gyms/examples/chk/checkpoint-3800')

evaluator.restore('C:/Users/ahaemm/ray_results_archive/PPO_WarehouseEnv_2_cf13e_00004_4_2022-12-19_12-19-25/checkpoint_001000/checkpoint-1000')

# gpusrv
#evaluator.restore("D:\Profactor\path_finder\checkpoint_008000\checkpoint-8000")

def evaluate(seed):
    # create the environment for evaluation
    env = WarehouseEnv_2({"reward_game_success": 1.0, "reward_each_action": -0.01, "reward_move_failure": -0.1, "seed": seed}, # seeding the RNG for random item locations at env.reset()
                         agent_ids=['agent_1', 'agent_2'],
                         max_steps=400,
                        deterministic_game=False,
                        image_observation=False,
                        action_masking=True,
                        num_objects=2,
                        seed=11,
                        random_items=[1,0,0])

    observations = env.reset()  # reset the environment
    if use_CC:
        observations = central_critic_observer(observations) # make observations compatible with CC
    #print("+++++++ observation after reset ++++++++")
    #print(observations)
    fig, ax = plt.subplots()

    mpl_img = ax.imshow(env.game.render())
    mpl_img.set_data(env.game.render())
    fig.canvas.draw()
    plt.show()

    episode_reward = 0
    timestep = 0

    dones = {}
    dones["__all__"] = False
    dones_agents = {}
    for agent in env._agent_ids:
        dones_agents[agent] = False

    # import pdb; pdb.set_trace()

    # step through one episode
    while not dones["__all__"]:
        # compute actions
        actions_dict = {}
        for agent in env._agent_ids:

            if not dones_agents[agent] and agent != 'heuristic_agent':
                actions_dict[agent] = evaluator.compute_action(observations[agent], policy_id=agent)

        # for debugging
        # print("timestep {}".format(timestep))
        # print("action agent 1: {}".format(actions_dict[agents[0]]))
        # #print("action agent 2: {}".format(actions_dict[agents[1]]))

        # execute one step
        observations, rewards, dones, infos = env.step(actions_dict)
        for agent in env._agent_ids:
            if agent in dones.keys(): # only enter that clause for active agents
                dones_agents[agent] = dones[agent]
                if infos[agent]['move_failure'] < 0.:
                    print("move failure agent {}".format(agent))
        if use_CC:
            observations = central_critic_observer(observations)
        # print(observations)

        # show updated game after step
        mpl_img = ax.imshow(env.game.render())
        mpl_img.set_data(env.game.render())
        fig.canvas.draw()
        plt.show()

        # print("after step: producer_1 load: {}".format(env.agents_dict['producer_1'].load))

        for reward in rewards.values():
            episode_reward += reward
        timestep += 1

    print("episode reward: " + str(episode_reward))
    # endof evaluate()

def evaluate_to_csv(episodes=300, name="evaluation_csv"):
    """
    writes agent observations, actions and success into a csv file;
    """

    header = ['Observation', 'Next Observation',  'Agent', 'Action', 'Action_success','Episode','Timestep']
    with open(name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    
        for eps in range(episodes):
            env_config = {
                "reward_game_success": 1.0,
                "reward_each_action": -0.01,
                "reward_move_failure": -0.1,
            }
            env = WarehouseEnv_3(env_config, agent_ids=agents, max_steps=400, deterministic_game=False,
                                image_observation=image_observation, action_masking=action_masking, num_objects=1, obstacle= True, dynamic=True)

            policy = evaluator.get_policy('policy_1')
            model = policy.model
            #model.internal_model.base_model.summary()


            observations = env.reset()  # reset the environment
    

            
            episode_reward = 0
            timestep = 0

            dones = {}
            dones["__all__"] = False

            # import pdb; pdb.set_trace()

            # step through one episode
            while not dones["__all__"]:
                # compute actions
                actions_dict = {}
                previous_obs = observations
                for agent in env._agent_ids:
                    
                    if agent != 'heuristic_agent':
                        actions_dict[agent] = evaluator.compute_action(observations[agent], policy_id= "policy_1")

                    
                

                # for debugging
                # print("timestep {}".format(timestep))
                # print("action agent 1: {}".format(actions_dict[agents[0]]))
                # #print("action agent 2: {}".format(actions_dict[agents[1]]))

                # execute one step
                observations, rewards, dones, infos = env.step(actions_dict)
                print("running episode", eps+1)
                print("--------------------")
                step_info = env.game.step_data
                for agent in env._agent_ids:
                    if step_info[agent]['action'] == 'drop': #drop is the move
                        writer.writerow([previous_obs[agent]['observations'], observations[agent]['observations'], agent, step_info[agent]['action'],
                        step_info[agent]['infos']['drop_failure'],eps+1, timestep])
                    elif step_info[agent]['action'] == 'pick': #pick is the move
                        writer.writerow([previous_obs[agent]['observations'], observations[agent]['observations'], agent, step_info[agent]['action'], 
                        step_info[agent]['infos']['pick_failure'],eps+1, timestep])
                    else: #other moves
                        writer.writerow([previous_obs[agent]['observations'], observations[agent]['observations'], agent, step_info[agent]['action'], 
                        step_info[agent]['infos']['move_failure'],eps+1, timestep])

                for reward in rewards.values():
                    episode_reward += reward
                timestep += 1


seed = 1234 # seed for game's RNG (used for random item locations at reset)
for i in range(10):
    evaluate(seed)
    seed += 10

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


