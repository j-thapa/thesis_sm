"""
This script contains various utility functions.
"""
import concert.gridworld.items
from concert.gridworld.items import ItemBase
import numpy as np
import random



def handle_not_element(config, arg, base_value):
    if arg in config:
        return config[arg]
    else:
        return base_value

def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# def tf_seed(seed):
#     tf.random.set_seed(seed)
#     tf.keras.utils.set_random_seed(seed)
