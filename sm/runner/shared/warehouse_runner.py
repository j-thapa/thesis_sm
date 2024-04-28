import time
import wandb
import numpy as np
from functools import reduce
import torch
from sm.runner.shared.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

class WarehouseRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for Warehouse. See parent class for details."""
    def __init__(self, config):
        super(WarehouseRunner, self).__init__(config)

    def run2(self):
        for episode in range(1):
            self.eval(episode)

    def run(self):
        self.warmup()

        start = time.time()
        policy_updates = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_terminated = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_games_success = np.zeros(self.n_rollout_threads, dtype=np.float32)
        


        for episode in range(policy_updates):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, policy_updates)

            #total_games = np.zeros(self.n_rollout_threads, dtype=np.float32)
            game_success = []
            reward_hist = []

            for step in range(self.episode_length):
            #no fix episode length, episode ends and resets in this length; not concrete episode length
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)
                dones_env = np.all(dones, axis=1)
               
 
                for t in range(self.n_rollout_threads):

                    if dones_env[t]:

                        # if np.mean(infos[t][0]['reward_till']) > -0.0098:
                        #     # print("lets see reward now thaT env is done-----------------")
                        #     # print(len(infos[t][0]['reward_till']), np.mean(infos[t][0]['reward_till']))
                        #     # print("+++++++++++++++++++++++++++++++++")
                    
                        #     print(infos[t][0]['reward_till'])
                        reward_hist.append(np.mean(infos[t][0]['reward_till']))
                        game_success.append(infos[t][0]['game_success'])
    


                #print(len(game_success), "step is", step)
                    
                    # add one more game if a new game is started
                      

                #reset if an env is terminated or done and replace obs with next obs, availkable action, share obs



                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic 
                
                
                
                # insert data into buffer
                self.insert(data)

            success_rate = np.mean(np.array(game_success))
     
            reward_rate =  np.mean(np.array(reward_hist))

             
            #print(" train game success rate is {}.".format(success_rate))

            # compute return and update network
            self.compute()
            train_infos = self.train()

          
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads          
            # save model
            if (total_num_steps % self.save_interval == 0 or episode == policy_updates - 1):
                self.save(episode)

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Algo {} updates {}/{}, total num timesteps {}/{}, FPS {}.\n"
                        .format(
                                self.algorithm_name,
                                episode,
                                policy_updates,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                print(" reward rate is {}.".format(reward_rate))
                print(" Success rate is {}.".format(success_rate))

                # games_success = []
                # terminated = []
                # incre_games_success = []
                # incre_terminated = []

                # # for i, info in enumerate(infos):
                #     if 'games_success' in info[0].keys():
                #         games_success.append(info[0]['game_success'])
                #         incre_games_success.append(info[0]['game_success']-last_games_success[i])
                #     if 'terminated' in info[0].keys():
                #         terminated.append(info[0]['terminated'])
                #         incre_terminated.append(info[0]['terminated']-last_terminated[i])


                if self.use_wandb:
                    wandb.log({"incre_win_rate": incre_win_rate}, step=total_num_steps)
                else:
                    self.writter.add_scalars("train_success_rate", {"train_success_rate": success_rate}, total_num_steps)
                    self.writter.add_scalars("eps_reward_rate", {"train_reward_rate": reward_rate}, total_num_steps)
                  

                # last_terminated = terminated
                # last_games_success = games_success

                # train_infos['dead_ratio'] = 1 - self.buffer.active_masks.sum() / reduce(lambda x, y: x*y, list(self.buffer.active_masks.shape)) 
                
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()


       

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs
        



  
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                            np.concatenate(self.buffer.obs[step]),
                                            np.concatenate(self.buffer.rnn_states[step]),
                                            np.concatenate(self.buffer.rnn_states_critic[step]),
                                            np.concatenate(self.buffer.masks[step]),
                                            np.concatenate(self.buffer.available_actions[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # print(active_masks, dones)
        # active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        # active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        #bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])
        
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks, None, None, available_actions)

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)

        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)


###add in the environment that if a game is success or terminated then next game begins ###
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_games_success = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:

            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = \
                self.trainer.policy.act(np.concatenate(eval_share_obs),
                                        np.concatenate(eval_obs),
                                        np.concatenate(eval_rnn_states),
                                        np.concatenate(eval_masks),
                                        np.concatenate(eval_available_actions),
                                        deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            one_episode_rewards.append(eval_rewards)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []
                    if eval_infos[eval_i][0]['game_success']:
                        eval_games_success += 1

            if eval_episode >= self.all_args.eval_episodes:
                # self.eval_envs.save_replay()
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards}                
                self.log_env(eval_env_infos, total_num_steps)
                eval_game_success = eval_games_success/eval_episode
                print("eval success rate is {}.".format(eval_game_success))
                if self.use_wandb:
                    wandb.log({"eval_game_success": eval_game_success}, step=total_num_steps)
                else:
                    self.writter.add_scalars("eval_game_success", {"eval_game_success": eval_game_success}, total_num_steps)
                break
