def config(policies):
    configuration = {
        "env": "FindGoalEnv_1",
        "log_level": "DEBUG",
        #"num_gpus": 1.0, ### this slows down the training duration (any value from 1 - 0.1)
        "num_workers": 1,
        "train_batch_size": 2000,
        "rollout_fragment_length": 2000,
        "batch_mode": "complete_episodes",
        #"num_gpus_per_worker": 0.125, ## this is not working (no gpu available)
        "num_cpus_per_worker": 1,
        "framework": "tfe", #tf for tensorflow, tfe for tensorflow eager
        "lr": 0.00005,
        # "lr": tune.grid_search([0.0001, 0.00005]),
        # "clip_param": tune.grid_search([0.3, 0.25, 0.2, 0.15, 0.1]), decreasing values produce ever worse rewards, see PPO_2021-07-06_12-14-57
        # "lambda": tune.grid_search([1.0, 0.975, 0.95]), # decreasing values produce ever worse rewards, see PPO_2021-07-06_10-51-14
        # "entropy_coeff": tune.grid_search([0.0, 0.025, 0.05, 0.075, 0.1]), # increasing values produce ever worse rewards, see PPO_2021-07-05_17-11-12
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": (lambda agent_id: agent_id),
            # "observation_fn": central_critic_observer,
        },
        "model": {
            #"dim": 45,
            #"conv_filters": [[16, [3, 3], 2], [32, [3, 3], 2], [512, [3, 3], 1]],
            #"conv_activation": "relu",
            "custom_model": "visionnet",
            "custom_model_config": {
                "conv_filters": [[16, [5, 5], 1]], # [32, [5, 5], 1], [128, [5, 5], 1]],
                "conv_activation": "relu",
                "post_fcnet_hiddens": [256],
                "post_fcnet_activation": "relu",
                #"no_final_linear":,
                #"vf_share_layers":,
            }
        },
        #"num_sgd_iter": 1,
        #"vf_loss_coeff": 0.0001,
        # "callbacks": MyCallbacks
    }
    return configuration