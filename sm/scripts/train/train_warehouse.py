#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
import argparse
sys.path.append("../../")
sys.path.append("../../sm/envs/")
from sm.config import get_config
from sm.envs.concert.warehouse_env import WarehouseMultiEnv #import warehoisebev
from sm.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from sm.runner.shared.warehouse_runner import WarehouseRunner as Runner
from sm.runner.separated.warehouse_runner import WarehouseRunner as Sep_Runner


"""Train script for concert Warehouse."""

    # register_env("WarehouseEnv_3",
    #                   lambda config: WarehouseEnv_3(config, agent_ids=agents, max_steps=70, deterministic_game=False, obs_stacking= False,
    #                             image_observation=image_observation, action_masking=action_masking, num_objects=2, obstacle = False, goal_area = True, dynamic= False, mode="training",
    #                              goals_coord = [(8,2),(5,2),(7,4),(2,5),(3,8)]))

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "WarehouseEnv":
                

                # Convert to dictionary
                args_dict = vars(all_args)

                # Modify the parameter you want to change. For example, changing 'parameter_name'
                args_dict['seed'] = all_args.seed * 50000 + rank * 10000

                # Create a new Namespace object with the modified arguments
                all_args_seeded = argparse.Namespace(**args_dict)

                print("Seeded with", all_args.seed * 50000 + rank * 10000)

                # Now pass all_args_seeded with the modified parameter
                env = WarehouseMultiEnv(all_args = all_args_seeded)
         
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError


            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "WarehouseEnv":

                # Convert to dictionary
                args_dict = vars(all_args)

                # Modify the parameter you want to change. For example, changing 'parameter_name'
                args_dict['seed'] = all_args.seed * 50000 + rank * 10000

                # Create a new Namespace object with the modified arguments
                all_args_seeded = argparse.Namespace(**args_dict)

                # Now pass all_args_seeded with the modified parameter
                env = WarehouseMultiEnv(all_args = all_args_seeded)


            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError

            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):

    # register_env("WarehouseEnv_3",
    #                   lambda config: WarehouseEnv_3(config, agent_ids=agents, max_steps=70, deterministic_game=False, obs_stacking= False,
    #                             image_observation=image_observation, action_masking=action_masking, num_objects=2, obstacle = False, goal_area = True, dynamic= False, mode="training",
    #                              goals_coord = [(8,2),(5,2),(7,4),(2,5),(3,8)]))


        # .environment("WarehouseEnv_3",
        # #seed = 1, # seed RNGs on RLlib level; for more information on why seeding is a good idea: https://docs.ray.io/en/latest/tune/faq.html#how-can-i-reproduce-experiments; https://github.com/ray-project/ray/blob/master/rllib/examples/deterministic_training.py
        # env_config = {
        #     "reward_game_success": 1.0, # default: 1.0
        #     "reward_each_action": -0.03, # default: -0.01
        #     "reward_illegal_action": -0.067, # default: -0.05
        #     "dynamic_obstacle_randomness": 0, #tune.grid_search([0.00, 0.20, 0.40, 0.60, 0.80, 1]), #defualt 0.25
        #     "heuristic_agent_randomness": 0.3, #tune.grid_search([0.00, 0.20, 0.40, 0.60, 0.80, 1]) #defualt 0.25
        #     "seed":  2,
        #     "worker_index" : 2,
        #     },

        
    # parser.add_argument('--reward_game_success', type=float, default=1.0)
    # parser.add_argument('--reward_each_action', type=float, default=-0.01)
    # parser.add_argument('--reward_illegal_action', type=float, default=-0.07)
    #works as available actions
    parser.add_argument("--action_masking", action='store_true', default=True)
    parser.add_argument("--image_observation", action='store_true', default=False)
    parser.add_argument("--obs_stacking", action='store_true', default=False)
    parser.add_argument("--num_objects", type = int, default=1)
    parser.add_argument("--pair_agents", type = int, default=2)
    parser.add_argument("--max_steps", type = int, default=100)
    parser.add_argument("--stacked_layers", type = int, default=3)
    parser.add_argument("--obstacle", action='store_true', default=False)
    parser.add_argument("--goal_area", action='store_true', default= False)
    parser.add_argument("--deterministic_game", action='store_true', default= False)
    parser.add_argument("--dynamic", action='store_true', default=False)
    parser.add_argument("--mode", type = str, default='training')
    parser.add_argument("--random_agent_order", action='store_true', default=False)
    parser.add_argument("--three_grid_object", action='store_true', default=False)
    parser.add_argument("--grid_shape", default=(15,15))
    parser.add_argument("--heuristic_agent", action='store_true', default= True)
    parser.add_argument("--random_agent", action='store_true', default= False)
    parser.add_argument("--goals_coord", default=  [(8,2),(5,2),(7,4),(2,5),(3,8)])
    parser.add_argument("--use_single_network", action='store_true', default=False)
    parser.add_argument("--partial_observation", action='store_true', default=True)


    #aadd here params

    all_args = parser.parse_known_args(args)[0]

    #number of agents
    all_args.num_agents = 2 * all_args.pair_agents

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)


    if all_args.algorithm_name == "mat_dec":
        all_args.dec_actor = True
        all_args.share_actor = True

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.algorithm_name 
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                         group=all_args.map_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "@" + str(
            all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env
    num_agents = all_args.num_agents
    all_args.run_dir = run_dir
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    if all_args.separated:
        runner = Sep_Runner(config)
    else:
        runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
