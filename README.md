 This repository is part of a academic course Masters Thesis, JKU.


This repository builds on the implementation of MAT repository https://github.com/PKU-MARL/Multi-Agent-Transformer.git 
We have included two new algorithms/architectures MAM and MARWKV build on Mamaba and RWKV. We have also added a new environment called Collaborative Warehouse for benchmarking purpose.


**For more details, one can go through the thesis report which will be uploaded in the same repository.**



## Installation

### Dependences
``` Bash
pip install -r requirements.txt
```

### Multi-agent MuJoCo
Following the instructios in https://github.com/openai/mujoco-py and https://github.com/schroederdewitt/multiagent_mujoco to setup a mujoco environment. In the end, remember to set the following environment variables:
``` Bash
LD_LIBRARY_PATH=${HOME}/.mujoco/mujoco200/bin;
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```

### StarCraft II & SMAC
Run the script
``` Bash
bash install_sc2.sh
```
Or you could install them manually to other path you like, just follow here: https://github.com/oxwhirl/smac.


### Bi-DexHands 
Please following the instructios in https://github.com/PKU-MARL/DexterousHands. 

### Collaborative Warehouse
Should run with the existing installation if some issue then can also follow installation from https://github.com/j-thapa/collaborative_warehouse.git

### MAMBA

To use selective scan used in Mamba which uses specific CUDA code, one have to installed Mamba following the official repository https://github.com/state-spaces/mamba.git



## How to run
When the environment is ready, one could run shells in the "scripts" folder with algo="mat" or algo="marwkv_v4" or algo="mamamba". For example:
``` Bash
./train_mujoco.sh  # run MAT/MARWKV on Multi-agent MuJoCo
```
If you would like to change the configs of experiments, you could modify sh files or look for config.py for more details.
