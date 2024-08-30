import numpy as np
from typing import List
from .items import ItemKind, ItemKind_encode, ItemKind_onehot
from . import art
import matplotlib
import matplotlib.pyplot as plt

DEFAULT_COLORS = {
    "#": (127, 127, 127),
    ".": (255, 255, 255),
    "o": (0, 200, 0),
    "m": (0, 100, 0),
    "a": (200, 0, 0),
    "g": (200, 0, 200),
    "t": (0, 0, 0),
    "b":(0, 0, 200),
    "y":(255,255,0)
}

DEFAULT_SPRITES = {
    ItemKind.WALL: art.Sprite.from_text(
        [
            "####",
            "####",
            "####",
            "####",
        ],
        DEFAULT_COLORS,
    ),
    ItemKind.OBJECT: art.Sprite.from_text(
        [
            "oooo",
            "o..o",
            "o..o",
            "oooo",
        ],
        DEFAULT_COLORS,
    ),
    ItemKind.OBJECT_ATTACHED: art.Sprite.from_text(
        [
            "oooo",
            "otto",
            "otto",
            "oooo",
        ],
        DEFAULT_COLORS,
    ),
    ItemKind.AGENT: art.Sprite.from_text(
        [
            ".aa.",
            "aaaa",
            "aaaa",
            ".aa.",
        ],
        DEFAULT_COLORS,
    ),
    ItemKind.H_AGENT: art.Sprite.from_text(
        [
            ".bb.",
            "bbbb",
            "bbbb",
            ".bb.",
        ],
        DEFAULT_COLORS,
    ),
    ItemKind.H_AGENT_ATTACHED: art.Sprite.from_text(
        [
            ".bb.",
            "bttb",
            "bttb",
            ".bb.",
        ],
        DEFAULT_COLORS,
    ),
    ItemKind.AGENT_ATTACHED: art.Sprite.from_text(
        [
            ".aa.",
            "atta",
            "atta",
            ".aa.",
        ],
        DEFAULT_COLORS,
    ),
    ItemKind.GOAL: art.Sprite.from_text(
        [
            "gggg",
            "gggg",
            "gggg",
            "gggg",
        ],
        DEFAULT_COLORS,
    ),
}

DEFAULT_ORDER = [ItemKind.GOAL, ItemKind.WALL,    ItemKind.OBJECT,  ItemKind.AGENT, ItemKind.H_AGENT, ItemKind.OBJECT_ATTACHED, ItemKind.AGENT_ATTACHED, ItemKind.H_AGENT_ATTACHED]





def render(engine, order: List[ItemKind] = None, image_observation:bool=True, partial_observation = False, loc = None) -> np.ndarray:
    """
    returns the current state of the world being represented by param "engine"; the state may be returned as an image
    (if param "image_observation" is True) or as a (flattened) 3-dim array, with one-hot encoded items on grid locations;
    """
    H, W = engine.shape
    state = None # return value; the actual engine state
    if order is None:
        order = DEFAULT_ORDER

    if partial_observation:

        #only 3x3 observation with loc being center is the state

        partial_state = []
        center_x, center_y = loc
        partial_coordinates = [(x, y) for x in range(center_x - 1, center_x + 2)
                     for y in range(center_y - 1, center_y + 2)]

        state = np.zeros(shape=(H,W,1), dtype='int32')
        for it in engine.items(order=order):

            state[it.loc[0]][it.loc[1]] = ItemKind_encode.encoding[it.kind]
        
      

        partial_state =  np.array([ItemKind_onehot.encode_one_hot[int(state[x[0]][x[1]])] for x in partial_coordinates])



        return partial_state.flatten()




   

    if image_observation:
        img = art.create_image(engine.shape)
        for it in engine.items(order=order):
            if any(isinstance(loc, np.ndarray) for loc in list(it.loc)):
                for loc_ in it.loc:
                    DEFAULT_SPRITES[it.kind].draw(img[loc_[1], loc_[0]])
            else:
                DEFAULT_SPRITES[it.kind].draw(img[it.loc[1], it.loc[0]])


        img = art.finalize_image(img)
        state = img
    else:
        state = np.zeros(shape=(H,W,ItemKind_onehot.num_itemkind), dtype='int32')


        # state = np.zeros(shape=(H,W,1), dtype='int32')
        for it in engine.items(order=order):

            # state[it.loc[0]][it.loc[1]] = ItemKind_encode.encoding[it.kind]

            #one hot encoding
            state[it.loc[0]][it.loc[1]] = ItemKind_onehot.encoding[it.kind]

        
            


        state = state.flatten()
   

    return state

