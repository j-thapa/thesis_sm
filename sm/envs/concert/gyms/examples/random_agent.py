import gym
import time

import concert.gyms

# from gym.wrappers.record_video import RecordVideo


def main():
    envs = concert.gyms.registered_concert_envs()
    print("Available concert envs", envs)
    env = gym.make(envs[0])
    for i_episode in range(10):
        observation = env.reset()
        for t in range(100):
            env.render(mode="human")
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
            time.sleep(0.01)
    env.close()


if __name__ == "__main__":
    main()