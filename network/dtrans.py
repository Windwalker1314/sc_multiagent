import torch.nn as nn
from network.transformers.obs_w import OBS_W
import torch
import numpy as np

class DTRANS(nn.Module):
    def __init__(self, input_shape,args) -> None:
        super(DTRANS, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.n_agents = args.n_agents
        hypernet_emb = args.hypernet_emb
        self.state_dim = int(np.prod(args.state_shape))
        self.obs_w = OBS_W(input_shape, args)
        self.b_mean = nn.Sequential(nn.Linear(self.state_dim, hypernet_emb),
                                    nn.ReLU(),
                                    nn.Linear(hypernet_emb, 1))
        
        """self.w_shape = nn.Sequential(nn.Linear(self.state_dim, hypernet_emb),
                                    nn.ReLU(),
                                    nn.Linear(hypernet_emb, input_shape))"""
    
    def forward(self, z_values, states, obs):
        b, t, n, nq = z_values.shape
        assert(n==self.n_agents)
        assert(states.shape == (b,t,self.args.state_shape))

        q_vals = z_values.mean(dim=3) # b, t, n
        Q_total = q_vals.sum(dim=2,keepdim=True).unsqueeze(3).expand(-1,-1,-1,nq) # b, t, 1, nq
        Z_total = z_values.sum(dim=2,keepdim=True) # b, t, 1, nq
        Z_shape = Z_total-Q_total

        w_mean = self.obs_w(states, obs).reshape(b,t,n)
        b_mean = self.b_mean(states).view(b,t,1).repeat(1, 1, self.n_agents)
        Z_mean = w_mean*q_vals + b_mean
        Z_mean = Z_mean.unsqueeze(2).expand(-1,-1,-1,nq) # b, t, 1, nq
        assert(Z_mean.shape==(b,t,1,nq))
        assert(Z_shape.shape==(b,t,1,nq))

        return Z_mean + Z_shape

