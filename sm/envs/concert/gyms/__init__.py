# from gym.envs.registration import register
# from gym import envs

# from .find_goal_env import FindGoalEnv
# from .warehouse_env import WarehouseEnv

# register(
#     id="FindGoalEnv-v0",
#     entry_point="concert.gyms:FindGoalEnv",
# )

# register(
#     id="WarehouseEnv-v0",
#     entry_point="concert.gyms:WarehouseEnv",
# )

# register(
#     id="WarehouseEnv-Hard-v0",
#     entry_point="concert.gyms:WarehouseEnv",
#     kwargs=dict(num_objects=3, shape=(16, 16)),
# )


# def registered_concert_envs():
#     """Returns a list of concert envs"""
#     all_envs = envs.registry.all()
#     env_ids = [
#         env_spec.id
#         for env_spec in all_envs
#         if env_spec.entry_point.startswith("concert.gyms")
#     ]
#     return env_ids
