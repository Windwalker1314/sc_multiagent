import numpy as np
import torch.nn as nn
import torch
from network.transformers.obs_w import OBS_W

class DTRANS(nn.Module):
    def __init__(self, input_shape,args) -> None:
        super(DTRANS, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.n_agents = args.n_agents
        self.obs_w = OBS_W(input_shape, args)
    
    def forward(self, z_values, states, obs):
        b, t, n, nq = z_values.shape
        assert(obs.shape == (b,t,n,self.input_shape))
        Z_total = z_values.sum(dim=2,keepdim=True) # b, t, 1, nq

        q_vals = z_values.mean(dim=3) # b, t, n
        q_total = q_vals.sum(dim=2,keepdim=True).unsqueeze(3).expand(-1,-1,-1,nq) # b, t, 1, nq
        Q_trans = self.obs_w(q_vals, states, obs).reshape(b,t,1,1)
        Q_trans = Q_trans.expand(-1,-1,-1,nq) # b, t, 1, nq
        assert(Q_trans.shape==(b,t,1,nq))

        Z_total = Z_total.reshape(b, t, 1, nq)
        q_total = q_total.reshape(b, t, 1, nq)
        return Z_total - q_total + Q_trans

