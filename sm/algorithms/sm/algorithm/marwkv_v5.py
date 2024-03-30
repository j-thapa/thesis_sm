import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torch.distributions import Categorical
from sm.algorithms.utils.util import check, init
from sm.algorithms.utils.transformer_act import discrete_autoregreesive_act
from sm.algorithms.utils.transformer_act import discrete_parallel_act
from sm.algorithms.utils.transformer_act import continuous_autoregreesive_act
from sm.algorithms.utils.transformer_act import continuous_parallel_act


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)

class ChannelMix(nn.Module):
    def __init__(self, layer_id, n_layer, n_embed):
        super().__init__()
        self.layer_id = layer_id
        
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - layer_id/n_layer
            x = torch.ones(1,1, n_embed)
            for i in range(n_embed):
                x[0, 0, i] = i/n_embed
            
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        
        hidden_size = 4*n_embed
        self.key = nn.Linear(n_embed, hidden_size, bias=False)
        self.receptance = nn.Linear(n_embed, n_embed, bias=False)
        
        self.value = nn.Linear(hidden_size, n_embed, bias=False)
        
    def forward(self, x, enc_rep=None):
        xx = self.time_shift(x)

        if enc_rep is None:
            enc_rep = x

        #my own doing instead of using x in key use encoder embedding
        xk = enc_rep * self.time_mix_k + (1-self.time_mix_k) * enc_rep

       
        xr = x * self.time_mix_r + (1-self.time_mix_r) * xx
        
        k = self.key(xk)
        k = torch.square(torch.relu(k))
        
        kv = self.value(k)
        
        
        
        rkv = torch.sigmoid(self.receptance(xr)) * kv
        
        return rkv


class RWKV_TimeMix_RWKV5(nn.Module):
    def __init__(self,layer_id, n_layer, n_embd):
        super().__init__()
    
        self.layer_id = layer_id
        self.dim_att = n_embd
        self.n_head = 1
        self.head_size = n_embd 


        with torch.no_grad():
            ratio_0_to_1 = layer_id / 1  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_mix_g = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            # fancy time_decay
            decay_speed = torch.ones(self.dim_att)
            for n in range(self.dim_att):
                decay_speed[n] = -6 + 5 * (n / (self.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(self.n_head, self.head_size))
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            tmp = torch.zeros(self.dim_att)
            for n in range(self.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (self.dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(n_embd, self.dim_att, bias=False)
        self.key = nn.Linear(n_embd, self.dim_att, bias=False)

        self.value = nn.Linear(n_embd, self.dim_att, bias=False)
        self.output = nn.Linear(self.dim_att, n_embd, bias=False)
        self.gate = nn.Linear(n_embd, self.dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, self.dim_att)

  
    def jit_func(self, x):
        B, T, C = x.size()

        xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        xg = x * self.time_mix_g + xx * (1 - self.time_mix_g)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        return r, k, v, g

  
    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)
        
        x = self.ln_x(x / 8).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head

        r, k, v, g = self.jit_func(x)

       

        return self.jit_func_2(x, g)


class RWKV_encode_block(nn.Module):
    def __init__(self, layer_id, n_layer, n_embd):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        # if self.layer_id == 0:
        #     self.ln0 = nn.LayerNorm(n_embd)

        # if self.layer_id == 0 :
        #     self.ffnPre = ChannelMix(0, n_layer, n_embd)

        # else:
           
            
        self.att = RWKV_TimeMix_RWKV5(layer_id, n_layer, n_embd)

        self.ffn = ChannelMix(layer_id, n_layer, n_embd)

    def forward(self, x):
        # # if self.layer_id == 0:
        #     x = self.ln0(x)        
        # if self.layer_id == 0 :
        #     x = x + self.ffnPre(self.ln1(x))  # better in some cases
        # else:
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class RWKV_decode_block(nn.Module):
    def __init__(self, layer_id, n_layer, n_embd):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        # if self.layer_id == 0:
        #     self.ln0 = nn.LayerNorm(n_embd)

        # if self.layer_id == 0 :
        #     self.ffnPre = ChannelMix(0, n_layer, n_embd)
        # else:
            #self.att = TimeMix(layer_id, n_layer, n_embd)
            
        self.att = RWKV_TimeMix_RWKV5(layer_id, n_layer, n_embd)

        self.ffn = ChannelMix(layer_id, n_layer, n_embd)

    def forward(self, x, enc_rep):
        # if self.layer_id == 0:
        #     x = self.ln0(x)        
        # if self.layer_id == 0 :
        #     x = x + self.ffnPre(self.ln1(x))  # better in some cases
        # else:
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x), None)
        return x

# class RWKV(nn.Module):
#     def __init__(self, n_layer, vocab_size,  n_embd, ctx_len):
#         super().__init__()
#         self.step = 0
#         self.ctx_len = ctx_len

#        # self.emb = nn.Embedding(vocab_size, n_embd)

#         self.blocks = nn.Sequential(*[Block(i, n_layer, n_embd)
#                                     for i in range(n_layer)])


    
#     def forward(self, x, targets=None):
#             #idx = idx.to(self.emb.weight.device)

    

#             # print("size of idx", idx.size())
            
#             # B, T = idx.size()
#             #no need to compare the maximum context length in this use case
#             # assert T <= self.ctx_len, "Cannot forward, because len(input) > model ctx_len."

#            # x = self.emb(idx)

#             x = self.blocks(x)
            
#             x = self.ln_out(x)

#             x = self.head(x)


#             # loss = None
#             # if targets is not None:
#             #     loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.to(x.device).view(-1))
#             #x = torch.mean(x, dim=0, keepdim=True)

#             return x
        
#     #no use of this function in this use case
#     # def generate(self, idx, max_new_tokes):
#     #     for _ in range(max_new_tokes):
#     #         idx_cond = idx[:, -block_size:]
#     #         logits, loss = self(idx_cond)
#     #         logits = logits[:, -1, :]
#     #         probs = F.softmax(logits, dim = -1)
#     #         idx_next = torch.multinomial(probs, num_samples = 1)
#     #         idx = torch.cat((idx, idx_next), dim = 1)
#     #     return idx



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
        self.rwkv =  nn.Sequential(*[RWKV_encode_block(i, n_block, n_embd)
                                    for i in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, 1)))

    def forward(self, state, obs):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        
        obs_embeddings = self.obs_encoder(obs)
        x = obs_embeddings
     

        rep = self.rwkv(self.ln(x))

        v_loc = self.head(rep)


        return v_loc, rep


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
        self.rwkv = nn.Sequential(*[RWKV_decode_block(i, n_block, n_embd)
                                    for i in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, action_dim)))

    def zero_std(self, device):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    # state, action, and return
    def forward(self, action, obs_rep, obs):
        # action: (batch, n_agent, action_dim), one-hot/logits?
        # obs_rep: (batch, n_agent, n_embd)
        action_embeddings = self.action_encoder(action)
        x = action_embeddings + obs_rep
        
        for block in self.rwkv:
            x = block(x, obs_rep)
        logit = self.head(x)

        return logit


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

        # state unused
        state_dim = 37

        self.encoder = Encoder(state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state)
        self.decoder = Decoder(obs_dim, action_dim, n_block, n_embd, n_head, n_agent,
                               self.action_type, dec_actor=dec_actor, share_actor=share_actor)
        self.to(device)

    def zero_std(self):
        if self.action_type != 'Discrete':
            self.decoder.zero_std(self.device)

    def forward(self, state, obs, action, available_actions=None):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        # action: (batch, n_agent, 1)
        # available_actions: (batch, n_agent, act_dim)

        # state unused
        ori_shape = np.shape(state)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(state)[0]
        v_loc, obs_rep = self.encoder(state, obs)
        if self.action_type == 'Discrete':
            action = action.long()
            action_log, entropy = discrete_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                        self.n_agent, self.action_dim, self.tpdv, available_actions)
        else:
            action_log, entropy = continuous_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                          self.n_agent, self.action_dim, self.tpdv)

        return action_log, v_loc, entropy

    def get_actions(self, state, obs, available_actions=None, deterministic=False):
        # state unused
        ori_shape = np.shape(obs)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(obs)[0]
        v_loc, obs_rep = self.encoder(state, obs)
        if self.action_type == "Discrete":
            output_action, output_action_log = discrete_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                           self.n_agent, self.action_dim, self.tpdv,
                                                                           available_actions, deterministic)
        else:
            output_action, output_action_log = continuous_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                             self.n_agent, self.action_dim, self.tpdv,
                                                                             deterministic)

        return output_action, output_action_log, v_loc

    def get_values(self, state, obs, available_actions=None):
        # state unused
        ori_shape = np.shape(state)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        v_tot, obs_rep = self.encoder(state, obs)
        return v_tot