import math,time

import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
from torch.distributions import Categorical
from sm.algorithms.utils.util import check, init
from sm.algorithms.utils.rnn_act import discrete_autoregreesive_act
from sm.algorithms.utils.rnn_act import discrete_parallel_act
from sm.algorithms.utils.rnn_act import continuous_autoregreesive_act
from sm.algorithms.utils.rnn_act import continuous_parallel_act


PREV_X_TIME = 0
NUM_STATE = 1
DEN_STATE = 2
MAX_STATE = 3
PREV_X_CHANNEL = 4

def init_bfloat16(layer):
    if hasattr(layer, 'weight') and layer.weight is not None:
        layer.weight.data = layer.weight.data.to(torch.bfloat16)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data = layer.bias.data.to(torch.bfloat16)


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


# copied from nanoGPT
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

# copied from nanoGPT
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class ChannelMixing(nn.Module):
    def __init__(self, n_embd,layer_id):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.layer_id = layer_id
        
        n_embd = n_embd
        intermediate_size = (
             4 * n_embd
        )
        
        ## Learnable Matrix
        self.key_proj        = nn.Linear(n_embd,intermediate_size,bias=False)
        self.value_proj      = nn.Linear(intermediate_size,n_embd,bias=False)
        self.receptance_proj = nn.Linear(n_embd,n_embd,bias=False)
        
        ## Learnable Vector
        self.time_mix_key        = nn.Parameter(torch.empty(1, 1, n_embd))
        self.time_mix_receptance = nn.Parameter(torch.empty(1, 1, n_embd))


    def forward(self,x,state=None):
        # x = (Batch,Time,Channel)
        if state is not None:
            prev_x = state[self.layer_id,:,PREV_X_CHANNEL,:,:]
            state[self.layer_id,:,PREV_X_CHANNEL,:,:] = x
        else:
            prev_x = self.time_shift(x)
            
        ## R
        receptance = x.clone() * self.time_mix_receptance + prev_x.clone() * (1 - self.time_mix_receptance)
        receptance = self.receptance_proj(receptance)
        receptance = F.sigmoid(receptance)

        # K
        key = x.clone() * self.time_mix_key + prev_x.clone() * (1 - self.time_mix_key)
        key = self.key_proj(key)

        # V
        value = self.value_proj(torch.square(torch.relu(key)))

        ## output
        out = receptance.clone() * value.clone()
        return out, state
    

class TimeMixing(nn.Module):
    def __init__(self, n_embd,layer_id):
        super().__init__()
     
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.layer_id = layer_id
        
        n_embd = n_embd
        attn_sz = n_embd

        ## learnable matrix
        self.key_proj        = nn.Linear(n_embd, attn_sz, bias=False)
        self.value_proj      = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance_proj = nn.Linear(n_embd, attn_sz, bias=False)
        self.output_proj     = nn.Linear(attn_sz, n_embd, bias=False)

        ## learnable vector
        self.time_decay          = nn.Parameter(torch.empty(attn_sz))
        self.time_first          = nn.Parameter(torch.empty(attn_sz))
        self.time_mix_key        = nn.Parameter(torch.empty(1, 1, n_embd))
        self.time_mix_value      = nn.Parameter(torch.empty(1, 1, n_embd))
        self.time_mix_receptance = nn.Parameter(torch.empty(1, 1, n_embd))


    def forward(self,x,state=None):
        # x = (Batch,Time,Channel)
        if state is not None:
            prev_x = state[self.layer_id,:,PREV_X_TIME,:,:]
            state[self.layer_id,:,PREV_X_TIME,:,:] = x
        else:
            prev_x = self.time_shift(x)

        # K
        key = x.clone() * self.time_mix_key + prev_x.clone() * (1 - self.time_mix_key)
        key = self.key_proj(key)
        
        # V
        value = x.clone() * self.time_mix_value + prev_x.clone() * (1 - self.time_mix_value)
        value = self.value_proj(value)
        
        # R
        receptance = x.clone() * self.time_mix_receptance + prev_x.clone() * (1 - self.time_mix_receptance)
        receptance = self.receptance_proj(receptance)
        receptance = F.sigmoid(receptance)

        # WKV
        wkv, state  = self.wkv_function(key,value,state=state, use_customized_cuda_kernel=True)
        
        # RWKV
        if state is not None:

            rwkv = receptance * wkv.unsqueeze(1).expand(-1, state.shape[3], -1)
        else:
            rwkv = receptance * wkv
        rwkv = self.output_proj(rwkv)
        
        return rwkv, state


    def wkv_function(self,key,value,state=None, use_customized_cuda_kernel=False):

        ## essentially, this customized cuda kernel delivers a faster for loop across time steps
        ## only for training and evaluating loss and ppl


        if state is None and use_customized_cuda_kernel is True:
            B, T, C = key.size()
            return WKVKernel.apply(B, T, C, self.time_decay, self.time_first, key, value), None

        else:
            print("""
            
            ______________state was not none, no cuda used++++++++++++++++++++++++++++++++++++
            ______________state was not none, no cuda used++++++++++++++++++++++++++++++++++++
            ______________state was not none, no cuda used++++++++++++++++++++++++++++++++++++
            ______________state was not none, no cuda used++++++++++++++++++++++++++++++++++++
            ______________state was not none, no cuda used++++++++++++++++++++++++++++++++++++
            ______________state was not none, no cuda used++++++++++++++++++++++++++++++++++++
            

            
            """)
         

            
            ## raw wkv function (from Huggingface Implementation)
            ## only for generation (because using raw pytorch for loop to train the model would be super super slow)

            _, seq_length, _ = key.size()
            output = torch.zeros_like(key)

            debug_mode = False
            if state is None:
                ## only for debug purpose when use_customized_cuda_kernel=False and state is None
                debug_mode = True
                num_state = torch.zeros_like(key[:, 0], dtype=torch.float32)
                den_state = torch.zeros_like(key[:, 0], dtype=torch.float32)
                max_state = torch.zeros_like(key[:, 0], dtype=torch.float32) - 1e38
            else:
                num_state  = state[self.layer_id,:,NUM_STATE,:,:] [:, 0]
                den_state  = state[self.layer_id,:,DEN_STATE,:,:] [:, 0]
                max_state  = state[self.layer_id,:,MAX_STATE,:,:] [:, 0]

            time_decay = -torch.exp(self.time_decay)

            for current_index in range(seq_length):
                current_key = key[:, current_index].float()
                current_value = value[:, current_index]

                # wkv computation at time t
        
        
                max_for_output = torch.maximum(max_state, current_key + self.time_first)
                e1 = torch.exp(max_state - max_for_output)
                e2 = torch.exp(current_key + self.time_first - max_for_output)
                numerator = e1 * num_state + e2 * current_value
                denominator = e1 * den_state + e2
                output = (numerator / denominator).to(output.dtype)

                # Update state for next iteration
                max_for_state = torch.maximum(max_state + time_decay, current_key)
                e1 = torch.exp(max_state + time_decay - max_for_state)
                e2 = torch.exp(current_key - max_for_state)
                num_state = e1 * num_state.clone() + e2 * current_value
                den_state = e1 * den_state.clone() + e2
                max_state = max_for_state
            
            if debug_mode:
                return output, None

            else:

                state[self.layer_id,:,NUM_STATE,:,:] = num_state.clone().unsqueeze(1).expand(-1, state.shape[3], -1)
            
                state[self.layer_id,:,DEN_STATE,:,:] = den_state.clone().unsqueeze(1).expand(-1, state.shape[3], -1)
                state[self.layer_id,:,MAX_STATE,:,:] = max_state.clone().unsqueeze(1).expand(-1, state.shape[3], -1)

            return output, state


class Block(nn.Module):

    def __init__(self, n_embd,layer_id):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd)
        self.attn = TimeMixing(n_embd, layer_id)
        self.ln_2 = LayerNorm(n_embd)
        self.ffn = ChannelMixing(n_embd,layer_id)

        # # Initialize the layer in bfloat16
        # init_bfloat16(self.ln_1)


        # # Initialize the layer in bfloat16
        # init_bfloat16(self.ln2)


    def forward(self, x, state = None):
        # state: [batch_size, 5 , n_embd]
       
        
        # time mixing
        residual = x
        x,state = self.attn(self.ln_1(x),state=state)
        x = x + residual
        
        # channel mixing
        residual = x
        x, state = self.ffn(self.ln_2(x),state=state)
        x = x + residual

        return x, state
    

class RWKV(nn.Module):

    def __init__(self, n_layer, n_embd,lr_init=0.0008):
        super().__init__()

        self.n_layer = n_layer
        self.n_embd = n_embd

        self.rescale_every = 6


        self.rwkv =        self.rwkv = nn.ModuleDict(dict(
            h = nn.ModuleList([Block(self.n_embd,layer_id) for layer_id in range(self.n_layer)]),
            ln_f = LayerNorm(self.n_embd, bias= False),
        ))


        self.apply(self._init_weights)
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

        print("loading customized cuda kernel")
        dtype = 'bfloat16' #if torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
        self.load_cuda_kernel(dtype)


    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the token embeddings get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())

        return n_params

    def _init_weights(self, module):

        ## initialize Vector Parameters in TimeMixing
        if isinstance(module,TimeMixing):
            layer_id = module.layer_id
            n_layer = self.n_layer
            n_embd = self.n_embd
            attn_sz = n_embd
            
            with torch.no_grad():
                if n_layer>1:
                    ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
                else:
                    ratio_0_to_1 = 1

                ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
                ddd = torch.ones(1, 1, n_embd)
                for i in range(n_embd):
                    ddd[0, 0, i] = i / n_embd

                decay_speed = torch.ones(attn_sz)
                for h in range(attn_sz):
                    decay_speed[h] = -5 + 8 * (h / (attn_sz - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                module.time_decay = nn.Parameter(decay_speed)

                zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(attn_sz)]) * 0.5
                module.time_first = nn.Parameter(torch.ones(attn_sz) * math.log(0.3) + zigzag)
                module.time_mix_key = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
                module.time_mix_value = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
                module.time_mix_receptance = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))
        
        ## initialize Vector Parameters in ChannelMixing
        elif isinstance(module,ChannelMixing):
            layer_id = module.layer_id
            n_layer = self.n_layer
            n_embd = self.n_embd
            
            with torch.no_grad():  # fancy init of time_mix
                ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
                ddd = torch.ones(1, 1, n_embd)
                for i in range(n_embd):
                    ddd[0, 0, i] = i / n_embd
                module.time_mix_key = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
                module.time_mix_receptance = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
        
        ## initialize Linear Layer and Embedding Layer
        elif isinstance(module,(nn.Embedding,nn.Linear)):
            weight = module.weight
            shape = weight.shape
            gain = 1.0
            scale = 1.0
            
            ## get the current name of the parameters
            for _name,_parameters in self.named_parameters():
                if id(_parameters) == id(weight):
                    current_module_name = _name
            
            # print(current_module_name)

            ## Embedding
            if isinstance(module, nn.Embedding):
                gain = math.sqrt(max(shape[0], shape[1]))
                scale = -1 * self.lr_init

            ## Linear
            elif isinstance(module,nn.Linear):
                if shape[0] > shape[1]:
                    gain = math.sqrt(shape[0] / shape[1])
                
                ## initialize some matrix to be all ZEROS
                for name in [".attn.key_proj.", ".attn.receptance_proj.", ".attn.output_proj.", 
                             ".ffn.value_proj.", ".ffn.receptance_proj."]:
                    if name in current_module_name:
                        scale = 0
                
                if current_module_name == 'lm_head.weight':
                    scale = 0.5

            if scale == 0:
                nn.init.zeros_(weight)
            elif scale < 0:
                nn.init.uniform_(weight, a=scale, b=-scale)
            else:
                nn.init.orthogonal_(weight, gain=gain * scale)

    
    def load_cuda_kernel(self,dtype):
        
        from torch.utils.cpp_extension import load
        T_MAX = 45
        RWKV_FLOAT_MODE = dtype
        cuda_dir = "...algorithm/cuda"

        print("load from directory")
    
            
        wkv_cuda = load(name=f"wkv_{T_MAX}_bf16", sources=[f"{cuda_dir}/wkv_op_bf16.cpp", f"{cuda_dir}/wkv_cuda_bf16.cu"], verbose=True, extra_cuda_cflags=["-t 4", "-std=c++17", "-res-usage", "--maxrregcount 60", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-DTmax={T_MAX}"])
        print("CUDA loaded sucessfully")
        class WKV(torch.autograd.Function):
            @staticmethod
            def forward(ctx, B, T, C, w, u, k, v):
                        
                k = k.bfloat16()
                v = v.bfloat16()
                w = w.bfloat16()
                u = u.bfloat16()
                ctx.B = B
                ctx.T = T
                ctx.C = C
                assert T <= T_MAX
                assert B * C % min(C, 32) == 0
                w = -torch.exp(w.float().contiguous())
                u = u.contiguous().bfloat16()
                k = k.contiguous()
                v = v.contiguous()
                y = torch.empty((B, T, C), device=w.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
                wkv_cuda.forward(B, T, C, w, u, k, v, y)
                ctx.save_for_backward(w, u, k, v, y)
                return y
            @staticmethod
            def backward(ctx, gy):
                B = ctx.B
                T = ctx.T
                C = ctx.C
                gy = gy.bfloat16()
                assert T <= T_MAX
                assert B * C % min(C, 32) == 0
                w, u, k, v, y = ctx.saved_tensors
                gw = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
                gu = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
                gk = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
                gv = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
                wkv_cuda.backward(B, T, C, w, u, k, v, y, gy.contiguous(), gw, gu, gk, gv)
                gw = torch.sum(gw, dim=0)
                gu = torch.sum(gu, dim=0)
                return (None, None, None, gw, gu, gk, gv)
 


        global WKVKernel
        WKVKernel = WKV 
                
    def forward(self, x, state, return_state=True):
 



        for block_idx,block in enumerate(self.rwkv.h):
            x, state = block(x,state)
            if state is not None: ## in generation mode
                if (
                    self.rescale_every > 0 
                    and (block_idx + 1) % self.rescale_every == 0
                ):
                    x = x/2
        x = self.rwkv.ln_f(x)


        if return_state:
            return x, state
        else:
            return x 






class Encoder(nn.Module):

    def __init__(self, state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state):
        super(Encoder, self).__init__()

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_agent = n_agent
        self.encode_state = encode_state


        self.state_encoder = nn.Sequential(nn.LayerNorm(state_dim),
                                           init_(nn.Linear(state_dim, n_embd), activate=True), nn.GELU())
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())

        self.ln = nn.LayerNorm(n_embd)
        # self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
        #use RWKV architecture; no use of ctx_length
        self.rwkv =  RWKV(n_block, n_embd)
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, 1)))

    def forward(self, state, obs):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)

        obs_embeddings = self.obs_encoder(obs)
        x = obs_embeddings
     

        rep, state= self.rwkv(self.ln(x), state)

        v_loc = self.head(rep)


        return v_loc, rep, state



class Decoder(nn.Module):
    def __init__(self, obs_dim, action_dim, n_block, n_embd, n_head, n_agent,
                 action_type='Discrete', dec_actor=False, share_actor=False):
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd
        self.dec_actor = dec_actor
        self.share_actor = share_actor
        self.action_type = action_type
     

        if action_type == 'Discrete':
            self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim + 1, n_embd, bias=False), activate=True),
                                                nn.GELU())
        else:
            log_std = torch.ones(action_dim)
            # log_std = torch.zeros(action_dim)
            self.log_std = torch.nn.Parameter(log_std)
            # self.log_std = torch.nn.Parameter(torch.zeros(action_dim))
            self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim, n_embd), activate=True), nn.GELU())
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())
        self.ln = nn.LayerNorm(n_embd)
        # self.blocks = nn.Sequential(*[DecodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
        self.rwkv = RWKV(n_block, n_embd)
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, action_dim)))

    def zero_std(self, device):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    # state, action, and return
    def forward(self, action, obs_rep, state):
        # action: (batch, n_agent, action_dim), one-hot/logits?
        # obs_rep: (batch, n_agent, n_embd)



        action_embeddings = self.action_encoder(action.clone())
        x = action_embeddings

        #last obs should represent obs_seq more aptly ?
        x = x + obs_rep


        
    
        x, state = self.rwkv(x, state)
        logit = self.head(x)

        return logit, state 





class MultiAgentRWKV(nn.Module):
    def __init__(self, state_dim, obs_dim, action_dim, n_agent,
                 n_block, n_embd, n_head, encode_state=False, device=torch.device("cpu"),
                 action_type='Discrete', dec_actor=False, share_actor=False):
        super(MultiAgentRWKV, self).__init__()

        self.n_agent = n_agent
        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.device = device
        self.n_layer = n_block
        self.n_embd = n_embd

        # state unused
        state_dim = 37

        self.encoder = Encoder(state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state)
        self.decoder = Decoder(obs_dim, action_dim, n_block, n_embd, n_head, n_agent,
                               self.action_type, dec_actor=dec_actor, share_actor=share_actor)
        self.to(device)

    def zero_std(self):
        if self.action_type != 'Discrete':
            self.decoder.zero_std(self.device)


    def init_state(self,batch_size):

        n_state = len([PREV_X_TIME,NUM_STATE,DEN_STATE,MAX_STATE,PREV_X_CHANNEL])
        state = torch.zeros(
            (self.n_layer,batch_size,n_state,self.n_agent,self.n_embd),
            dtype=torch.float32, device=self.device,
        )
        state[:,:,MAX_STATE,:,:] -= 1e30
        
        return state

    
    
    def forward(self, state, obs, action, available_actions=None):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        # action: (batch, n_agent, 1)
        # available_actions: (batch, n_agent, act_dim)

        # state unused
        ori_shape = np.shape(state)
        batch_size = np.shape(state)[0]

        
        #state = self.init_state(batch_size)
        state = None
        

        # state = check(state).to(**self.tpdv)
        # state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        deterministic = False

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

       
        v_loc, obs_rep, state= self.encoder(state, obs)
        if self.action_type == 'Discrete':

            action = action.long()
            # , entropy = discrete_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
            #                                             self.n_agent, self.action_dim, self.tpdv, available_actions)

            #rnn formulation of discrete_parallel_act

            action_log, entropy = discrete_parallel_act(self.decoder, obs_rep, obs, action, batch_size, self.n_agent, self.action_dim, self.tpdv, available_actions)




                                            


        else:
            action_log, entropy = continuous_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                          self.n_agent, self.action_dim, self.tpdv)

        return action_log, v_loc, entropy

    def get_actions(self, state, obs, available_actions=None, deterministic=False, use_state =False):
        # state unused
        ori_shape = np.shape(obs)
        #state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)
        batch_size = np.shape(state)[0]

        #use this to initislize state if it is in generation mode else use state None

        if use_state:

            state = self.init_state(batch_size)
            state = check(state).to(**self.tpdv)
        else:
            state = None

        
        

        
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(obs)[0]

        v_loc, obs_rep, state = self.encoder(state, obs)
        if self.action_type == "Discrete":

            output_action, output_action_log = discrete_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                           self.n_agent, self.action_dim, self.tpdv,
                                                                           available_actions, deterministic, state)
        else:
            output_action, output_action_log = continuous_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                             self.n_agent, self.action_dim, self.tpdv,
                                                                             deterministic, state)




        return output_action, output_action_log, v_loc

    def get_values(self, state, obs, available_actions=None):
        # state unused
        ori_shape = np.shape(state)
        # state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)
        batch_size = np.shape(state)[0]

        state = None

        # state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        v_tot, obs_rep, state= self.encoder(state, obs)
        return v_tot



