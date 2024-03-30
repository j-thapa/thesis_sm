from concert.gyms.find_goal_env import FindGoalEnv_1
from concert.gyms.warehouse_env import WarehouseEnv_1, WarehouseEnv_2, WarehouseEnv_3
import matplotlib.pyplot as plt

Actions = ["left", "right", "up", "down", "pick", "drop", "do_nothing"]


def test_warehouse_env_1():
    global action
    fig, ax = plt.subplots()
    image_observation = True
    agents = ['agent_1']
    # env = FindGoalEnv_1(agents)
    env = WarehouseEnv_1(agents, max_steps=100, deterministic_game=False, image_observation=image_observation,
                         action_masking=False, num_obstacles=1)
    result_reset = env.reset()
    if image_observation:
        mpl_img = ax.imshow(env.game.render())
        mpl_img.set_data(env.game.render())
        fig.canvas.draw()
        plt.show()
    else:
        print("reset")
        print(result_reset)
    # **************************************************************************
    # step through environment, applying a sequence of actions
    actions = [4, 5, 1, 2, 3, 0, 5, 2, 4, 5]
    for action in actions:
        results = env.step({'agent_1': action})
        if image_observation:
            mpl_img = ax.imshow(env.game.render())
            mpl_img.set_data(env.game.render())
            fig.canvas.draw()
            plt.show()
        else:
            print("step, action is {}".format(action))
            print(results)
    # ***************************************************************************

def test_warehouse_env_2():
    global action
    fig, ax = plt.subplots()
    image_observation = False
    action_masking = True
    agents = ['agent_1', 'agent_2']
    env = WarehouseEnv_2({},agent_ids=agents, max_steps=100, deterministic_game=True, image_observation=image_observation,
                         action_masking=action_masking, num_objects=2)
    result = env.reset()
    if image_observation:
        mpl_img = ax.imshow(env.game.render())
        mpl_img.set_data(env.game.render())
        fig.canvas.draw()
        plt.show()
    else:
        print("reset")
        print(result)
    # **************************************************************************
    # step through environment, applying a sequence of actions for both agents
    # actions = [(4,2), (1,4), (3,2)] # tuple = (action agent_1, action agent_2)
    actions = [(4,4)]
    for action_tuple in actions:
        results = env.step({'agent_1': action_tuple[0], 'agent_2': action_tuple[1]})
        if image_observation:
            mpl_img = ax.imshow(env.game.render())
            mpl_img.set_data(env.game.render())
            fig.canvas.draw()
            plt.show()
        else:
            print("step, action tuple is {}".format(action_tuple))
            print(results)
    # ***************************************************************************
    print("")

def test_warehouse_env_3():
    """
    test WarehouseEnv_3: one trained agent, one heuristic agent;
    creates a deterministic game, the game configuration is set in WarehouseGame_3._create_game_deterministic();
    """
    global action
    fig, ax = plt.subplots()
    env = WarehouseEnv_3({},
                         agent_ids=['agent_1'],
                         max_steps=400,
                         deterministic_game=False,
                         image_observation=False,
                         action_masking=True,
                         num_objects=2,
                         goal_area=True)
    result = env.reset()

    mpl_img = ax.imshow(env.game.render())
    mpl_img.set_data(env.game.render())
    fig.canvas.draw()
    plt.show()

    print("reset")
    print(result)

    # **************************************************************************
    # step through environment, applying a sequence of actions
    actions = [2,4,6,6,6,6,6,6,6,6,6,6,6,6,6] # actions: ["left", "right", "up", "down", "pick", "drop", "do_nothing"]
    for action in actions:
        results = env.step({'agent_1': action})
        mpl_img = ax.imshow(env.game.render())
        mpl_img.set_data(env.game.render())
        fig.canvas.draw()
        plt.show()
        print("step, action is {}".format(action))
        print(results)
    # ***************************************************************************

    print("")

if __name__ == "__main__":
    test_warehouse_env_3()















