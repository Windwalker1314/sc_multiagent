import numpy as np
import torch.nn as nn
import torch
import math
from network.transformers.obs_w import OBS_W

class DTRANS(nn.Module):
    def __init__(self, input_shape,args) -> None:
        super(DTRANS, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.n_agents = args.n_agents
        self.obs_w = OBS_W(input_shape, args)

        self.state_w =nn.Sequential(nn.Linear(args.state_shape, args.hypernet_emb),
                                     nn.ReLU(),
                                     nn.Linear(args.hypernet_emb, self.n_agents))
        self.state_b = nn.Sequential(nn.Linear(args.state_shape, args.hypernet_emb),
                                     nn.ReLU(),
                                     nn.Linear(args.hypernet_emb, 1))
    
    def forward(self, z_values, states, obs, rnd_q):
        b, t, n, nq = z_values.shape
        assert(obs.shape == (b,t,n,self.input_shape))
        assert(n==self.n_agents)
        assert(states.shape == (b,t,self.args.state_shape))
        assert(rnd_q.shape == (b, t, 1, nq))
        q_vals = z_values.mean(dim=3) # b, t, n
        z_shape = z_values-q_vals.unsqueeze(3).expand(-1,-1,-1,nq) # b,t,n,nq

        # Z_shape
        w = torch.abs(self.state_w(states)).unsqueeze(3).expand(-1,-1,-1,nq) # b, t, n, nq
        z_shape *= w
        Z_shape = z_shape.sum(dim=2,keepdim=True) # b, t, 1, nq
        state_b = self.state_b(states).unsqueeze(3).expand(-1,-1,-1,nq)*torch.cos(math.pi * rnd_q) # b, t, 1 -> b, t, 1, nq
        assert(state_b.shape==(b,t,1,nq))
        Z_shape = Z_shape + state_b # b, t, 1, nq

        # Z_mean
        Z_mean = self.obs_w(q_vals, states, obs).reshape(b,t,1,1)
        Z_mean = Z_mean.expand(-1,-1,-1,nq) # b, t, 1, nq
        assert(Z_mean.shape==(b,t,1,nq))

        Z_mean = Z_mean.reshape(b, t, 1, nq)
        Z_shape = Z_shape.reshape(b, t, 1, nq)
        return Z_mean + Z_shape

