from gymnasium.spaces import Box

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch

import logging


#from torchsummary import summary


import torch.nn.functional as F

from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


class compressed_VN(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
      
        self.conv1 = nn.Conv2d(in_channels=obs_space['observations'].shape[0],  
                               out_channels=16,
                               kernel_size=3,
                               stride=2,
                               padding=1)  
        self.conv2 = nn.Conv2d(in_channels=16,
                               out_channels=32,
                               kernel_size=3,
                               stride=2,
                               padding=2)  
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization after conv1
        self.conv3 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               stride=1,
                               padding=1) 
        self.bn2 = nn.BatchNorm2d(64)  # Batch normalization after conv1
        self.conv4 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=1,
                               stride=1,
                               padding=1) 
        
        self.fc1 = nn.Linear(3328, 128) 
        self.fc2 = nn.Linear(128,36)
        self.fc3 = nn.Linear(36, num_outputs)
        self.fc_value_out = nn.Linear(36,1)

    @override(TorchModelV2)
    def forward(
        self,
        input_dict,
        state,
        seq_lens):

        obs = input_dict["obs_flat"]["observations"]

        obs = obs.float()

        self.obs = obs
        x = self.conv1(obs)


        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.bn1(x)
   

        
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = nn.ReLU()(x)


        
   

        x = nn.Flatten()(x)
      

        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        model_out= self.fc3(x)
        # ... continue with the rest of your forward pass ...

        
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]
    

        # Assuming action_mask is a PyTorch tensor
        log_action_mask = torch.log(action_mask)

        # Clamp the values of `log_action_mask` between -1e10 and FLOAT_MAX
        FLOAT_MAX = 3.4e38
        inf_mask = torch.clamp(log_action_mask, min=-1e10, max=FLOAT_MAX)

        return model_out + inf_mask, state
    
    



    @override(TorchModelV2)
    def value_function(self):

 
        x = self.conv1(self.obs)


        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.bn1(x)
   

        
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = nn.ReLU()(x)

        x = nn.Flatten()(x)
        

        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        v_out = self.fc_value_out(x).squeeze(1)

        return v_out



class TorchCentralizedCriticModel(TorchModelV2, nn.Module):
    """Multi-agent model that implements a centralized VF."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # Base of the model
        self.model = compressed_VN(obs_space, action_space, num_outputs, model_config, name)

        # Central VF maps (obs, opp_obs, opp_act) -> vf_pred
               # Convolutional layers

        num_agents = obs_space['num_agents'].shape[0]
        

        self.conv1 = nn.Conv2d(in_channels=obs_space['observations'].shape[0] * num_agents,  
                               out_channels=16,
                               kernel_size=3,
                               stride=2,
                               padding=1)  
        self.conv2 = nn.Conv2d(in_channels=16,
                               out_channels=32,
                               kernel_size=3,
                               stride=2,
                               padding=2)  
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization after conv1
        self.conv3 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               stride=1,
                               padding=1) 
        self.bn2 = nn.BatchNorm2d(64)  # Batch normalization after conv1
        self.conv4 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=1,
                               stride=1,
                               padding=1) 



        # Dense layers
        #72 (output of flttened conv stack) + action_dim of opponents
        self.fc1 = nn.Linear(3328 + num_agents - 1, 128) 
        self.fc2 = nn.Linear(128,36)
        self.fc3 = nn.Linear(36, 1)



    def central_value_function(self, observations, opponent_obs, opponent_actions):

   
        #stack agent and opponent observations
        x = torch.cat([observations.float(), opponent_obs.float()], dim=1)


        # Apply convolutional layers
        x = self.conv1(x)


        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.bn1(x)
   

        
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = nn.ReLU()(x)       

        x = nn.Flatten()(x)  # Flatten

        

        #add vector action choices of opponent in the flatten output of convolution block
        x = torch.cat([x, opponent_actions], dim=1)

        # Apply dense layers
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        value = self.fc3(x)

        return torch.reshape(value, [-1])
        



    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        model_out, _ = self.model(input_dict, state, seq_lens)
        return model_out, []

    @override(ModelV2)
    def value_function(self):
        return self.model.value_function()  # not used

