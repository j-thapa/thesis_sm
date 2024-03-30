import numpy as np
from typing import List
from .items import ItemKind, ItemKind_onehot
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
    ItemKind.R_AGENT: art.Sprite.from_text(
        [
            ".yy.",
            "yyyy",
            "yyyy",
            ".yy.",
        ],
        DEFAULT_COLORS,
    ),
}

DEFAULT_ORDER = [ItemKind.WALL,   ItemKind.GOAL, ItemKind.OBJECT,  ItemKind.AGENT, ItemKind.H_AGENT, ItemKind.OBJECT_ATTACHED, ItemKind.AGENT_ATTACHED, ItemKind.R_AGENT, ItemKind.H_AGENT_ATTACHED]





def render(engine, order: List[ItemKind] = None, image_observation:bool=True) -> np.ndarray:
    """
    returns the current state of the world being represented by param "engine"; the state may be returned as an image
    (if param "image_observation" is True) or as a (flattened) 3-dim array, with one-hot encoded items on grid locations;
    """
    H, W = engine.shape
    state = None # return value; the actual engine state
    if order is None:
        order = DEFAULT_ORDER

   

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
        for it in engine.items(order=order):
            if any(isinstance(loc, np.ndarray) for loc in list(it.loc)):
                for loc_ in it.loc:
                    
                    state[loc_[0]][loc_[1]] = ItemKind_onehot.encoding[it.kind]
            else:
                state[it.loc[0]][it.loc[1]] = ItemKind_onehot.encoding[it.kind]
            
            

   

        state = state.flatten()

    return state


#function to render one hot encoded encoded observation from evaluation csv to images
def render_observation(observation, shape):

    #string formatting to format and convert obs to int
    temp_grid = []
    obs = observation.replace('\n', '').replace('\r','')
    obs = obs.replace('[','').replace(']','').replace('.','')
    obs = obs.split(' ')
    observation = [int(x) for x in obs]


 
    
    item_num = len(DEFAULT_ORDER)
    for x in range(0, len(observation), item_num):

        if observation [x:x + item_num] [0] == 1 : 
            temp_grid.append(0) #wall


        elif observation [x:x + item_num] [1] == 1 :
            temp_grid.append(2) #goal

        elif observation [x:x + item_num] [2] == 1 :
            temp_grid.append(1) #object



        elif observation [x:x + item_num] [3] == 1 :
            temp_grid.append(3) #agent

        elif observation [x:x + item_num] [4] == 1 :
            temp_grid.append(4) #heuristic_agent


        elif observation [x:x + item_num] [5] == 1 :
            temp_grid.append(5) #attached object

        elif observation [x:x + item_num] [6] == 1 :
            temp_grid.append(6) #attached agent

        elif observation [x:x + item_num] [7] == 1 :
            temp_grid.append(7) #random agent/ dynamic obstacle

        elif observation [x:x + item_num] [8] == 1 :
            temp_grid.append(8) #heuristic_agent_attached


        else:
            temp_grid.append(-1) #free grid cells
    
    #reshape matrix into shape of observation
    matrix = np.array(temp_grid).reshape([shape[0],shape[1]])
    items =[]

    for r_idx,row in enumerate(matrix): #loop through rows of matrix
        for c_idx,column in enumerate(row): #loop through columns of the row
       
            if column != -1: #if it is not grid cell
          
                items.append((DEFAULT_ORDER[column],(r_idx,c_idx))) #append the item type based on column value along wih its position set

   #create empty image
    img = art.create_image([shape[0],shape[1]])


    #add items in the image based on items kind and their position set
    for it in items:
        DEFAULT_SPRITES[it[0]].draw(img[it[1][1], it[1][0]])
    
    #final image
    img = art.finalize_image(img)
    return img

