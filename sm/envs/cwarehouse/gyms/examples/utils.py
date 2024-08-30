"""
This script contains various utility functions.
"""

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


