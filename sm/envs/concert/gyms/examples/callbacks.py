from typing import Dict
import argparse
import numpy as np
import os
import ray
from ray import tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv, GroupAgentsWrapper
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker, Episode
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch


class MyCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        # Make sure this episode has just been started (only initial obs
        # logged so far).

        assert episode.length  -1, \
            "ERROR: `on_episode_start()` callback should be called right " \
            "after env reset!; episodeV2 starts with -1"

        episode.user_data["move_failure"] = []
        episode.user_data["pick_failure"] = []
        episode.user_data["drop_failure"] = []
        episode.user_data["game_success"] = []
        episode.user_data["steps_exceeded"] = []
        episode.user_data["pick_success"] = []
        # episode.hist_data["pole_angles"] = []

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs):
        # Make sure this episode is ongoing.
        assert episode.length > 0, \
            "ERROR: `on_episode_step()` callback should not be called right " \
            "after env reset!"
        # total reward contributions over all agents
        sum_move_failure = 0.
        sum_pick_failure = 0.
        sum_drop_failure = 0.
        sum_game_success = 0.
        sum_steps_exceeded = 0.
        sum_pick_success = 0.

        env = base_env.get_sub_environments()[0] # get_unwrapped() gets a list of environments
        if isinstance(env, GroupAgentsWrapper):
            game = env.env.game
        else:
            game = env.game
        #doesn't add heuristic or random agents callbacks
  
        game_agents = [x for x in list(game.agent_dict.keys()) if "heuristic" not in x  and "random" not in x]
        num_agents = len(game_agents)
        for agent in game_agents:

            if 'move_failure' in list(episode._last_infos[agent].keys()):
                sum_move_failure += episode._last_infos[agent]['move_failure']
            else:
                sum_move_failure += 0.0
            if 'pick_failure' in list(episode._last_infos[agent].keys()):
                sum_pick_failure += episode._last_infos[agent]['pick_failure']
            else:
                sum_pick_failure += 0.
            if 'drop_failure' in list(episode._last_infos[agent].keys()):
                sum_drop_failure += episode._last_infos[agent]['drop_failure']
            else:
                sum_drop_failure += 0.
            if 'game_success' in list(episode._last_infos[agent].keys()):
                sum_game_success += episode._last_infos[agent]['game_success'] / num_agents # count game success only once
            else:
                sum_game_success += 0.
            if 'steps_exceeded' in list(episode._last_infos[agent].keys()):
                sum_steps_exceeded += episode._last_infos[agent]['steps_exceeded'] / num_agents
            else:
                sum_steps_exceeded += 0.
            if 'pick_success' in list(episode._last_infos[agent].keys()):
                sum_pick_success += episode._last_infos[agent]['pick_success']
            else:
                sum_pick_success += 0.

        episode.user_data["move_failure"].append(sum_move_failure)
        episode.user_data["pick_failure"].append(sum_pick_failure)
        episode.user_data["drop_failure"].append(sum_drop_failure)
        episode.user_data["game_success"].append(sum_game_success)
        episode.user_data["steps_exceeded"].append(sum_steps_exceeded)
        episode.user_data["pick_success"].append(sum_pick_success)

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        # Make sure this episode is really done.
        # assert episode.batch_builder.policy_collectors[
        #    "default_policy"].buffers["dones"][-1], \
        #    "ERROR: `on_episode_end()` should only be called " \
        #    "after episode is done!"

        # total reward contribs for this episode
        sum_move_failure = sum(episode.user_data["move_failure"])
        sum_pick_failure = sum(episode.user_data["pick_failure"])
        sum_drop_failure = sum(episode.user_data["drop_failure"])
        sum_game_success = sum(episode.user_data["game_success"])
        sum_steps_exceeded = sum(episode.user_data["steps_exceeded"])
        sum_pick_success = sum(episode.user_data["pick_success"])

        episode.custom_metrics["move_failure"] = sum_move_failure
        episode.custom_metrics["pick_failure"] = sum_pick_failure
        episode.custom_metrics["drop_failure"] = sum_drop_failure
        episode.custom_metrics["game_success"] = sum_game_success
        episode.custom_metrics["steps_exceeded"] = sum_steps_exceeded
        episode.custom_metrics["pick_success"] = sum_pick_success
        # episode.hist_data["pole_angles"] = episode.user_data["pole_angles"]

"""
    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):
        print("returned sample batch of size {}".format(samples.count))

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        print("trainer.train() result: {} -> {} episodes".format(
            trainer, result["episodes_this_iter"]))
        # you can mutate the result dict to add new fields to return
        result["callback_ok"] = True

    def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch,
                          result: dict, **kwargs) -> None:
        result["sum_actions_in_train_batch"] = np.sum(train_batch["actions"])
        print("policy.learn_on_batch() result: {} -> sum actions: {}".format(
            policy, result["sum_actions_in_train_batch"]))

    def on_postprocess_trajectory(
            self, *, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        print("postprocessed {} steps".format(postprocessed_batch.count))
        if "num_batches" not in episode.custom_metrics:
            episode.custom_metrics["num_batches"] = 0
        episode.custom_metrics["num_batches"] += 1
"""

class MyCallbacks_singleAgent(DefaultCallbacks):
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: Episode, env_index: int, **kwargs):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        assert episode.length == 0, \
            "ERROR: `on_episode_start()` callback should be called right " \
            "after env reset!"

        episode.user_data["move_failure"] = []
        episode.user_data["pick_failure"] = []
        episode.user_data["drop_failure"] = []
        episode.user_data["game_success"] = []
        episode.user_data["steps_exceeded"] = []
        episode.user_data["pick_success"] = []
        # episode.hist_data["pole_angles"] = []

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: Episode, env_index: int, **kwargs):
        # Make sure this episode is ongoing.
        assert episode.length > 0, \
            "ERROR: `on_episode_step()` callback should not be called right " \
            "after env reset!"
        # total reward contributions over all agents

        env = base_env.get_unwrapped()[0] # get_unwrapped() gets a list of environments

        if 'move_failure' in list(episode._last_infos().keys()):
            move_failure = episode._last_infos()['move_failure']
        else:
            move_failure = 0.0
        if 'pick_failure' in list(episode._last_infos().keys()):
            pick_failure = episode._last_infos()['pick_failure']
        else:
            pick_failure = 0.
        if 'drop_failure' in list(episode._last_infos().keys()):
            drop_failure = episode._last_infos()['drop_failure']
        else:
            drop_failure = 0.
        if 'game_success' in list(episode._last_infos().keys()):
            game_success = episode._last_infos()['game_success']
        else:
            game_success = 0.
        if 'steps_exceeded' in list(episode._last_infos().keys()):
            steps_exceeded = episode._last_infos()['steps_exceeded']
        else:
            steps_exceeded = 0.
        if 'pick_success' in list(episode._last_infos().keys()):
            pick_success = episode._last_infos()['pick_success']
        else:
            pick_success = 0.

        episode.user_data["move_failure"].append(move_failure)
        episode.user_data["pick_failure"].append(pick_failure)
        episode.user_data["drop_failure"].append(drop_failure)
        episode.user_data["game_success"].append(game_success)
        episode.user_data["steps_exceeded"].append(steps_exceeded)
        episode.user_data["pick_success"].append(pick_success)

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: Episode,
                       env_index: int, **kwargs):
        # Make sure this episode is really done.
        # assert episode.batch_builder.policy_collectors[
        #    "default_policy"].buffers["dones"][-1], \
        #    "ERROR: `on_episode_end()` should only be called " \
        #    "after episode is done!"

        # total reward contribs for this episode
        sum_move_failure = sum(episode.user_data["move_failure"])
        sum_pick_failure = sum(episode.user_data["pick_failure"])
        sum_drop_failure = sum(episode.user_data["drop_failure"])
        sum_game_success = sum(episode.user_data["game_success"])
        sum_steps_exceeded = sum(episode.user_data["steps_exceeded"])
        sum_pick_success = sum(episode.user_data["pick_success"])

        episode.custom_metrics["move_failure"] = sum_move_failure
        episode.custom_metrics["pick_failure"] = sum_pick_failure
        episode.custom_metrics["drop_failure"] = sum_drop_failure
        episode.custom_metrics["game_success"] = sum_game_success
        episode.custom_metrics["steps_exceeded"] = sum_steps_exceeded
        episode.custom_metrics["pick_success"] = sum_pick_success
        # episode.hist_data["pole_angles"] = episode.user_data["pole_angles"]
