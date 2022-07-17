import torch.nn as nn
from network.transformers.obs_w import OBS_W
import torch

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
        assert(n==self.n_agents)
        assert(states.shape == (b,t,self.args.state_shape))

        w = self.obs_w(states,obs).reshape(b,t,n,1).expand(-1,-1,-1,nq) #b,t,n,nq
        z_values = z_values*w

        return torch.sum(z_values, dim=2, keepdim=True)

