import time
import numpy as np
import gymnasium
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

import concert.gyms  # needed to find gyms


def setup_env(env_create_fn, vectorized: bool = True):
    if vectorized:
        env = DummyVecEnv([lambda: Monitor(env_create_fn())])
        env = VecTransposeImage(env)
    else:
        env = env_create_fn()
    return env


def train_agent(
    env_create_fn,
    num_timesteps: int = 100000,
    save_path: str = "./tmp/agent",
    callback=None,
):
    check_env(env_create_fn())
    env = setup_env(env_create_fn)
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=1e-3,
        batch_size=64,
    )
    model.learn(total_timesteps=num_timesteps, callback=callback)
    model.save(save_path)
    env.close()


def render_agent_gif(
    env_create_fn, num_steps: int = 300, load_path: str = "./tmp/agent"
):
    model = PPO.load(load_path)
    env = setup_env(env_create_fn, vectorized=False)

    imgs = []
    obs = env.reset()
    for _ in range(num_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        imgs.append(env.render(mode="rgb_array"))
        if done:
            obs = env.reset()
    env.close()
    imageio.mimsave("./tmp/agent.gif", imgs, fps=5)


def _eval_agent(model, env, num_eps: int = 100):
    num_successes = 0
    episode_lengths = []
    for e in range(num_eps):
        obs = env.reset()
        for s in range(100):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done:
                tinfo = info["terminal_info"]
                num_successes += int(tinfo["goal_reached"])
                episode_lengths.append(s)
                break
    env.close()
    return num_successes / num_eps, np.mean(episode_lengths)


def eval_agent(
    env_create_fn, num_eps: int = 100, load_path: str = "./tmp/agent"
) -> float:
    model = PPO.load(load_path)
    env = setup_env(env_create_fn, vectorized=False)
    return _eval_agent(model, env, num_eps=num_eps)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["train", "render_gif", "eval"])
    parser.add_argument(
        "-env", type=str, help="environment name", default="FindGoalEnv-v0"
    )
    args = parser.parse_args()

    if args.mode == "train":
        train_agent(env_create_fn=lambda: gymnasium.make(args.env))
    elif args.mode == "render_gif":
        render_agent_gif(env_create_fn=lambda: gymnasium.make(args.env))
    else:
        print(
            "Fraction of times goal reached: ",
            eval_agent(env_create_fn=lambda: gymnasium.make(args.env)),
        )
