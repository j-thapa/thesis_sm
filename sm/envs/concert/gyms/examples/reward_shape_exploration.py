import time
import gymnasium
import matplotlib.pyplot as plt
from gymnasium.core import Wrapper
from gymnasium.envs.registration import register

from .find_goal_agent import train_agent, _eval_agent
from concert.gyms import FindGoalEnv

from stable_baselines3.common.callbacks import BaseCallback


class SuccessRateCallback(BaseCallback):
    def __init__(
        self, eval_env, verbose: int = 0, eval_freq: int = 1000, num_eps: int = 100
    ):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.num_eps = num_eps
        self.avg_success_rates = []
        self.avg_episode_lengths = []
        self.timesteps = []

    def _on_step(self) -> bool:
        if self.model.num_timesteps % self.eval_freq == 0:
            srate, elen = _eval_agent(self.model, self.eval_env, self.num_eps)
            self.avg_success_rates.append(srate)
            self.avg_episode_lengths.append(elen)
            self.timesteps.append(self.model.num_timesteps)


class RewardShape1(gym.Wrapper):
    """Wraps FindGoalEnv to return no intermediate rewards."""

    def __init__(self, env: FindGoalEnv):
        super().__init__(env)
        self.env = env
        self.reward_range = (0.0, 1.0)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if not done:
            reward = 0.0
        else:
            tinfo = info["terminal_info"]
            if tinfo["goal_reached"]:
                reward = 1.0
            else:
                reward = 0.0
        return obs, reward, done, info


class RewardShape2(gym.Wrapper):
    """Wraps FindGoalEnv to return no intermediate rewards."""

    def __init__(self, env: FindGoalEnv):
        super().__init__(env)
        self.env = env
        self.reward_range = (
            1 - ((self.env.game.max_steps + 1) / self.env.game.max_steps),
            1.0,
        )

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if not done:
            reward = 0.0
        else:
            beta = 1.0
            reward = 1 - beta * (
                (self.env.game.num_steps + 1) / self.env.game.max_steps
            )
        return obs, reward, done, info


if __name__ == "__main__":
    variants = [
        lambda: gym.make("FindGoalEnv-v0"),
        lambda: RewardShape1(gym.make("FindGoalEnv-v0")),
        lambda: RewardShape2(gym.make("FindGoalEnv-v0")),
    ]
    num_timesteps = 50000
    num_eval_eps = 1000
    eval_env = variants[0]()
    eval_data = []
    for vidx, create_fn in enumerate(variants):
        print("variant", vidx)
        cb = SuccessRateCallback(eval_env, eval_freq=1000, num_eps=num_eval_eps)
        train_agent(
            create_fn,
            num_timesteps=num_timesteps,
            save_path=f"./tmp/agent_v{vidx:2d}",
            callback=cb,
        )
        eval_data.append((cb.avg_success_rates, cb.avg_episode_lengths, cb.timesteps))

    fig, axs = plt.subplots(1, 2)
    axs[0].set_xlabel("timestep")
    axs[0].set_ylabel("avg. success rate")
    axs[1].set_xlabel("timestep")
    axs[1].set_ylabel("avg. episode length")
    for vidx, data in enumerate(eval_data):
        axs[0].plot(data[2], data[0], label=f"variant {vidx:2d}")
        axs[1].plot(data[2], data[1], label=f"variant {vidx:2d}")
    plt.legend()
    plt.show()

    # for vidx, create_fn in enumerate(variants):
    #     success_rate = eval_agent(
    #         create_fn, num_eps=num_eval_eps, load_path=f"./tmp/agent_v{vidx:2d}"
    #     )
    #     print("variant", vidx, "success rate", success_rate)
