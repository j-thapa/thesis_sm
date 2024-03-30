import copy
import numpy as np
import unittest

import ray
from ray.rllib.agents.callbacks import DefaultCallbacks
import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.ppo.ppo_tf_policy import ppo_surrogate_loss as \
    ppo_surrogate_loss_tf
from ray.rllib.agents.ppo.ppo_torch_policy import ppo_surrogate_loss as \
    ppo_surrogate_loss_torch
from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch, \
    Postprocessing
from ray.rllib.models.tf.tf_action_dist import Categorical
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch
#from ray.rllib.utils.metrics.learner_info import LEARNER_INFO, \
#    LEARNER_STATS_KEY
from ray.rllib.utils.numpy import fc
from ray.rllib.utils.test_utils import check, check_compute_single_action, \
    framework_iterator

# Fake CartPole episode of n time steps.
FAKE_BATCH = SampleBatch({
    SampleBatch.OBS: np.array(
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]],
        dtype=np.float32),
    SampleBatch.ACTIONS: np.array([0, 1, 1]),
    SampleBatch.PREV_ACTIONS: np.array([0, 1, 1]),
    SampleBatch.REWARDS: np.array([1.0, -1.0, .5], dtype=np.float32),
    SampleBatch.PREV_REWARDS: np.array([1.0, -1.0, .5], dtype=np.float32),
    SampleBatch.DONES: np.array([False, False, True]),
    SampleBatch.VF_PREDS: np.array([0.5, 0.6, 0.7], dtype=np.float32),
    SampleBatch.ACTION_DIST_INPUTS: np.array(
        [[-2., 0.5], [-3., -0.3], [-0.1, 2.5]], dtype=np.float32),
    SampleBatch.ACTION_LOGP: np.array([-0.5, -0.1, -0.2], dtype=np.float32),
    SampleBatch.EPS_ID: np.array([0, 0, 0]),
    SampleBatch.AGENT_INDEX: np.array([0, 0, 0]),
})

ray.init(local_mode=True)

"""Test whether a PPOTrainer can be built with all frameworks."""
config = copy.deepcopy(ppo.DEFAULT_CONFIG)
# For checking lr-schedule correctness.
#config["callbacks"] = MyCallbacks

config["num_workers"] = 1
config["num_sgd_iter"] = 2
# Settings in case we use an LSTM.
#config["model"]["lstm_cell_size"] = 10
#config["model"]["max_seq_len"] = 20
# Use default-native keras models whenever possible.
# config["model"]["_use_default_native_models"] = True

# Setup lr- and entropy schedules for testing.
#config["lr_schedule"] = [[0, config["lr"]], [128, 0.0]]
# Set entropy_coeff to a faulty value to proof that it'll get
# overridden by the schedule below (which is expected).
#config["entropy_coeff"] = 100.0
#config["entropy_coeff_schedule"] = [[0, 0.1], [256, 0.0]]

config["train_batch_size"] = 128
# Test with compression.
config["compress_observations"] = True
num_iterations = 2

for fw in framework_iterator(config):
    for env in ["MsPacmanNoFrameskip-v4"]:
        print("Env={}".format(env))
        for lstm in [False]:
            print("LSTM={}".format(lstm))
            config["model"]["use_lstm"] = lstm
            config["model"]["lstm_use_prev_action"] = lstm
            config["model"]["lstm_use_prev_reward"] = lstm

            trainer = ppo.PPOTrainer(config=config, env=env)
            policy = trainer.get_policy()
            entropy_coeff = trainer.get_policy().entropy_coeff
            lr = policy.cur_lr
            if fw == "tf":
                entropy_coeff, lr = policy.get_session().run(
                    [entropy_coeff, lr])
            check(entropy_coeff, 0.1)
            check(lr, config["lr"])

            for i in range(num_iterations):
                results = trainer.train()
                #check_train_results(results)
                print(results)

            check_compute_single_action(
                trainer,
                include_prev_action_reward=True,
                include_state=lstm)
            trainer.stop()
ray.shutdown()

