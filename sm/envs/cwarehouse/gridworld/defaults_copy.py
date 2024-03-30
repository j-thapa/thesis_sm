import numpy as np
from typing import List
from .items import ItemKind
from concert.gyms import art

DEFAULT_COLORS = {
    "#": (127, 127, 127),
    ".": (255, 255, 255),
    "o": (0, 200, 0),
    "a": (200, 0, 0),
    "g": (200, 0, 200),
    "t": (0, 0, 0) # marks attached objects / agents
}

DEFAULT_SPRITES = {
    ItemKind.WALL: art.Sprite.from_text(
        [
            "#####",
            "#####",
            "#####",
            "#####",
            "#####",
        ],
        DEFAULT_COLORS,
    ),
    ItemKind.OBJECT: art.Sprite.from_text(
        [
            "ooooo",
            "o...o",
            "o...o",
            "o...o",
            "ooooo",
        ],
        DEFAULT_COLORS,
    ),
    ItemKind.OBJECT_ATTACHED: art.Sprite.from_text(
        [
            "ooooo",
            "o.t.o",
            "ottto",
            "o.t.o",
            "ooooo",
        ],
        DEFAULT_COLORS,
    ),
    ItemKind.AGENT: art.Sprite.from_text(
        [
            "..a..",
            ".aaa.",
            "aaaaa",
            ".aaa.",
            "..a..",
        ],
        DEFAULT_COLORS,
    ),
    ItemKind.AGENT_ATTACHED: art.Sprite.from_text(
        [
            "..a..",
            ".ata.",
            "attta",
            ".ata.",
            "..a..",
        ],
        DEFAULT_COLORS,
    ),
    ItemKind.GOAL: art.Sprite.from_text(
        [
            "ggggg",
            "ggggg",
            "ggggg",
            "ggggg",
            "ggggg",
        ],
        DEFAULT_COLORS,
    ),
}

DEFAULT_ORDER = [ItemKind.WALL, ItemKind.GOAL, ItemKind.OBJECT, ItemKind.AGENT, ItemKind.OBJECT_ATTACHED, ItemKind.AGENT_ATTACHED]


def render(engine, order: List[ItemKind] = None) -> np.ndarray:
    H, W = engine.shape
    if order is None:
        order = DEFAULT_ORDER
    img = art.create_image(engine.shape)
    for it in engine.items(order=order):
        DEFAULT_SPRITES[it.kind].draw(img[it.loc[1], it.loc[0]])

    img = art.finalize_image(img)

    return img
