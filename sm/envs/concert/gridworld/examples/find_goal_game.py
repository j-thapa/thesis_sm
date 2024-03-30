import numpy as np
from .. import engine, items, defaults, games



# environment just uses agent IDs, agent objects are created and used in the game
class FindGoalGame_1(games.GameBase):
    def __init__(self, agent_id, seed: int = None, shape=(9, 9), max_steps: int = 100) -> None:
        super().__init__(seed=seed)
        self.shape = shape
        self.max_steps = max_steps
        self.agent_id = agent_id
        self.agent: items.MoveAgent = None
        self.agent_dict = {} # mapping of agent ID to agent object

    def _create_game(self):
        self.engine = engine.GridEngine(self.shape)
        w = items.WallItem.create_border(self.shape)
        self.engine.add(w)
        self.items = [
            items.MoveAgent((-1, -1)),
            items.GoalItem((-1, -1)),
        ]
        self.engine._add_randomly(self.items)
        self.agent = self.items[0]
        self.agent_dict = {self.agent_id: self.agent}

    def _step(self, step_data: games.StepData):
        actions = step_data.actions # mapping of agent objects to actions
        # Move the agent
        if not self.agent.move(actions[self.agent]):
            step_data.add_reward(-0.1)
        # Check game end
        if self._check_goal():
            step_data.add_reward(1.0)
            step_data.terminate_game()
        elif step_data.steps_exceeded(self.max_steps):
            step_data.add_reward(-1.0)
            step_data.terminate_game()

    def _check_goal(self) -> bool:
        the_items = self.engine.items(locs=self.agent.loc)
        the_items = [i for i in the_items if i.kind == items.ItemKind.GOAL]
        return len(the_items) > 0

    def render(self) -> np.ndarray:
        return defaults.render(self.engine)

"""
def play_findgoalgame():
    import matplotlib.pyplot as plt

    print("Use w,a,s,d to move")

    game = FindGoalGame(max_steps=10)
    game.reset()

    fig, ax = plt.subplots()
    mpl_img = ax.imshow(game.render())

    def on_press(event):
        if event.key == "w":
            game.step({game.agent: "up"})
        elif event.key == "s":
            game.step({game.agent: "down"})
        elif event.key == "a":
            game.step({game.agent: "left"})
        elif event.key == "d":
            game.step({game.agent: "right"})
        if game.done:
            print(game.reward_history)
            game.reset()
        mpl_img.set_data(game.render())
        fig.canvas.draw()

    fig.canvas.mpl_connect("key_press_event", on_press)
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    plt.show()


if __name__ == "__main__":
    play_findgoalgame()
"""

class FindGoalGame(games.GameBase):
    def __init__(
        self,
        seed: int = None,
        shape=(9, 9),
        max_steps: int = 100,
        initial_agent_loc=(-1, -1),
        initial_goal_loc=(-1, -1),
    ) -> None:
        super().__init__(seed=seed)
        self.shape = shape
        self.max_steps = max_steps
        self.initial_agent_loc = initial_agent_loc
        self.initial_goal_loc = initial_goal_loc
        self.agent: items.MoveAgent = None

    def _create_game(self):
        self.engine = engine.GridEngine(self.shape)
        w = items.WallItem.create_border(self.shape)
        self.engine.add(w)
        self.items = [
            items.MoveAgent(self.initial_agent_loc),
            items.GoalItem(self.initial_goal_loc),
        ]
        self.engine.add(self.items)
        self.agent = self.items[0]

    def _step(self, step_data: games.StepData):
        actions = step_data.actions
        # Move the agent
        if not self.agent.move(actions[self.agent]):
            step_data.add_reward(-0.01)
        # Check game end
        if self._check_goal():
            step_data.add_reward(1.0)
            step_data.terminate_game({"goal_reached": True})
        elif step_data.steps_exceeded(self.max_steps):
            step_data.add_reward(-1.0)
            step_data.terminate_game({"goal_reached": False})

    def _check_goal(self) -> bool:
        the_items = self.engine.items(locs=self.agent.loc)
        the_items = [i for i in the_items if i.kind == items.ItemKind.GOAL]
        return len(the_items) > 0

    def render(self) -> np.ndarray:
        return defaults.render(self.engine)


def play_findgoalgame():
    import matplotlib.pyplot as plt

    print("Use w,a,s,d to move")

    game = FindGoalGame_1('agent_1', max_steps=10)
    game.reset()

    fig, ax = plt.subplots()
    mpl_img = ax.imshow(game.render())

    def on_press(event):
        if event.key == "w":
            game.step({game.agent: "up"})
        elif event.key == "s":
            game.step({game.agent: "down"})
        elif event.key == "a":
            game.step({game.agent: "left"})
        elif event.key == "d":
            game.step({game.agent: "right"})
        if game.done:
            print(game.reward_history)
            game.reset()
        mpl_img.set_data(game.render())
        fig.canvas.draw()

    fig.canvas.mpl_connect("key_press_event", on_press)
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    plt.show()


if __name__ == "__main__":
    play_findgoalgame()
