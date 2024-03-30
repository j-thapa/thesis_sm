
from ray.rllib.algorithms.ppo.ppo import PPO
from ray.rllib.algorithms.ppo.ppo_tf_policy import (
    PPOTF1Policy,
    PPOTF2Policy,
)
import numpy as np

import torch

from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
import time 
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.numpy import convert_to_numpy

from ray.rllib.utils.tf_utils import explained_variance, make_tf_callable

from ray.rllib.utils.torch_utils import convert_to_torch_tensor

OPPONENT_OBS = "opponent_obs"
OPPONENT_ACTION = "opponent_action"

class CentralizedValueMixin:
    """Add method to evaluate the central value function from the model."""

    def __init__(self):

        self.compute_central_vf = self.model.central_value_function


# Grabs the opponent obs/act and includes it in the experience train_batch,
# and computes GAE using the central vf predictions.
def centralized_critic_postprocessing(
    policy, sample_batch, other_agent_batches=None, episode=None
):

    pytorch = True
    if (pytorch and hasattr(policy, "compute_central_vf")) or (
        not pytorch and policy.loss_initialized()
    ):
        assert other_agent_batches is not None


        if policy.config["enable_connectors"]:
            #[(_, _, opponent_batch)] = list(other_agent_batches.values())
            concat_opponent_batch = [opponent_n_batch for _, _,opponent_n_batch in other_agent_batches.values()]
        else:
           # [(_, opponent_batch)] = list(other_agent_batches.values())
            concat_opponent_batch = [opponent_n_batch for  _,opponent_n_batch in other_agent_batches.values()]

                   

        #concating observations and actions of opponent batches to create one single batch
        #here opponent and agent observations are stacked upon one another and  action vectors are later concat in model
        concat_obs = [x['obs']['observations'] for x in concat_opponent_batch]
        concat_actions = [x['actions'].reshape(-1,1) for x in concat_opponent_batch]
       
        concat_opp_obs_batch = np.concatenate(concat_obs, axis=1)
        concat_opp_action_batch = np.concatenate(concat_actions, axis=1)
     
        # also record the opponent obs and actions in the trajectory
        sample_batch[OPPONENT_OBS] =concat_opp_obs_batch
        sample_batch[OPPONENT_ACTION] = concat_opp_action_batch

        # overwrite default VF prediction with the central VF
        # overwrite default VF prediction with the central VF
        if pytorch:
            sample_batch[SampleBatch.VF_PREDS] = (
                policy.compute_central_vf(
                    convert_to_torch_tensor(
                        sample_batch[SampleBatch.CUR_OBS]['observations'], policy.device
                    ),
                    convert_to_torch_tensor(sample_batch[OPPONENT_OBS], policy.device),
                    convert_to_torch_tensor(
                        sample_batch[OPPONENT_ACTION], policy.device
                    ),
                )
                .cpu()
                .detach()
                .numpy()
            )


        else:
            sample_batch[SampleBatch.VF_PREDS] = convert_to_numpy(
                policy.compute_central_vf(
                    sample_batch[SampleBatch.CUR_OBS]['observations'],
                    sample_batch[OPPONENT_OBS],
                    sample_batch[OPPONENT_ACTION],
                )
            )
    else:

 
        # Policy hasn't been initialized yet, use zeros.
  

        num_opp_agents = len(sample_batch[SampleBatch.CUR_OBS]['num_agents'][0]) - 1

        #create zero value obs and action sample if the policy is not initialized

        sample_obs_ = np.concatenate([sample_batch[SampleBatch.CUR_OBS]['observations'] for x in range(num_opp_agents)], axis=1)
        sample_action_ = np.concatenate([sample_batch[SampleBatch.ACTIONS].reshape(-1,1) for x in range(num_opp_agents)], axis=1)
        
        
        sample_batch[OPPONENT_OBS] = np.zeros_like(sample_obs_)
        sample_batch[OPPONENT_ACTION] = np.zeros_like(sample_action_)
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
            sample_batch[SampleBatch.REWARDS], dtype=np.float32
        )

    completed = sample_batch[SampleBatch.TERMINATEDS][-1]
    if completed:
        last_r =0.0
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
    #CentralizedValueMixin.__init__(policy)
    # Save original value function.
    vf_saved = model.value_function

    # Calculate loss with a custom value function.

    model.value_function = lambda: policy.model.central_value_function(
        train_batch[SampleBatch.CUR_OBS]['observations'],
        train_batch[OPPONENT_OBS],
        train_batch[OPPONENT_ACTION],
    )
    policy._central_value_out = model.value_function()
    loss = base_policy.loss(model, dist_class, train_batch)

    # Restore original value function.
    model.value_function = vf_saved

    return loss

# def setup_tf_mixins(policy, obs_space, action_space, config):
#     # Copied from PPOTFPolicy (w/o ValueNetworkMixin).
#     KLCoeffMixin.__init__(policy, config)
#     EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
#                                   config["entropy_coeff_schedule"])
#     LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])

def central_vf_stats(policy, train_batch):
# Report the explained variance of the central value function.
    return {
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], policy._central_value_out
        )
    }

def get_ccppo_policy(base):
    class CCPPOTFPolicy(CentralizedValueMixin, base):
    
        # def __init__(self, observation_space, action_space, config, existing_inputs = None, existing_model = None):
        #     base.__init__(self, observation_space, action_space, config,
        #     existing_inputs = None,
        #     existing_model = None)
        #     CentralizedValueMixin.__init__(self)
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

class CentralizedCritic(PPO):
    @classmethod
    @override(PPO)
    def get_default_policy_class(cls, config):
    
         return CCPPOTorchPolicy