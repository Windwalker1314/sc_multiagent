import torch.nn as nn
from network.transformers.obs_w import OBS_W
import torch.nn.functional as f
import torch
import math
import numpy as np

class DTRANS(nn.Module):
    def __init__(self, input_shape,args) -> None:
        super(DTRANS, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.n_agents = args.n_agents
        self.qe = args.quantile_emb_dim 
        self.hypernet_emb = args.hypernet_emb
        self.state_dim = int(np.prod(args.state_shape))
        self.obs_w = OBS_W(args)
        """self.b_mean = nn.Sequential(nn.Linear(self.state_dim, self.hypernet_emb),
                                    nn.ReLU(),
                                    nn.Linear(self.hypernet_emb, 1))"""

        self.psi = nn.Sequential(nn.Linear(self.state_dim, self.hypernet_emb),
                                 nn.ReLU(),
                                 nn.Linear(self.hypernet_emb, self.hypernet_emb))
        self.phi = nn.Linear(self.qe, self.hypernet_emb)
        self.g = nn.Linear(self.hypernet_emb, 1)
        
        self.w_shape = nn.Sequential(nn.Linear(self.state_dim, self.hypernet_emb),
                                    nn.ReLU(),
                                    nn.Linear(self.hypernet_emb, self.n_agents))
    
    """def forward(self, z_values, states, obs,rnd_q):
        b, t, n, nq = z_values.shape
        assert(n==self.n_agents)
        assert(states.shape == (b,t,self.args.state_shape))

        q_vals = z_values.mean(dim=3) # b, t, n
        w_shape = torch.abs(self.w_shape(states)).reshape(b,t,n,1).expand(-1,-1,-1,nq)
        z_shape = z_values-q_vals.view(b,t,n,1).expand(-1,-1,-1,nq) # b, t, n, nq
        z_shape *= w_shape
        Z_shape = z_shape.sum(dim=2,keepdim=True)

        tau = rnd_q.view(b*t*nq, 1).expand(-1, self.qe)  # b*t*nq, qe
        i = torch.arange(0,self.qe).view(1,-1).expand(b*t*nq, self.qe)
        if self.args.cuda:
            i = i.cuda()
        phi = f.relu(self.phi(torch.cos(math.pi * i * tau))) # b*t*nq, emb
        assert phi.shape == (b*t*nq, self.hypernet_emb)
        psi = self.psi(states)
        psi = psi.reshape(b,t,1,self.hypernet_emb).expand(b,t,nq,self.hypernet_emb).reshape(b*t*nq,self.hypernet_emb)
        # psi: b*t*nq, emb
        Z_state = self.g(psi * phi).reshape(b, t, 1, nq)
        Z_state = Z_state - Z_state.mean(dim=3,keepdim=True).expand(-1,-1,-1,nq)
        Z_shape += Z_state

        w_mean = self.obs_w(states, obs).reshape(b,t,n)
        b_mean = self.b_mean(states).view(b,t,1).repeat(1, 1, self.n_agents)
        Z_mean = w_mean*q_vals + b_mean # b,t,n
        Z_mean = Z_mean.sum(dim=2,keepdim=True).unsqueeze(3).expand(-1,-1,-1,nq) # b, t, 1, nq
        
        assert(Z_mean.shape==(b,t,1,nq))
        assert(Z_shape.shape==(b,t,1,nq))

        return Z_mean + Z_shape"""
    def forward(self, z_values, states, obs,rnd_q):
        b, t, n, nq = z_values.shape
        assert(n==self.n_agents)
        assert(states.shape == (b,t,self.args.state_shape))

        w_shape = self.obs_w(states, obs).reshape(b,t,n,1).expand(-1,-1,-1,nq)
        #w_shape = f.softmax(self.w_shape(states),dim=2).unsqueeze(3).expand(-1,-1,-1,nq) * n
        z_values *= w_shape

        tau = rnd_q.view(b*t*nq, 1).expand(-1, self.qe)  # b*t*nq, qe
        i = torch.arange(0,self.qe).view(1,-1).expand(b*t*nq, self.qe)
        if self.args.cuda:
            i = i.cuda()
        phi = f.relu(self.phi(torch.cos(math.pi * i * tau))) # b*t*nq, emb
        assert phi.shape == (b*t*nq, self.hypernet_emb)
        psi = self.psi(states)
        psi = psi.reshape(b,t,1,self.hypernet_emb).expand(b,t,nq,self.hypernet_emb).reshape(b*t*nq,self.hypernet_emb)
        # psi: b*t*nq, emb
        Z_state = self.g(psi * phi).reshape(b, t, 1, nq)
        Z_total = z_values.sum(dim=2, keepdim=True) + Z_state
        return Z_total
