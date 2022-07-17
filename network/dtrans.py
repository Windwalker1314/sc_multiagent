import torch.nn as nn
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
        assert(n==self.n_agents)
        assert(states.shape == (b,t,self.args.state_shape))

        q_vals = z_values.mean(dim=3) # b, t, n
        Q_total = q_vals.sum(dim=2,keepdim=True).unsqueeze(3).expand(-1,-1,-1,nq) # b, t, 1, nq
        Z_total = z_values.sum(dim=2,keepdim=True) # b, t, 1, nq
        Z_shape = Z_total-Q_total

        Z_mean = self.obs_w(q_vals, states, obs).reshape(b,t,1,1)
        Z_mean = Z_mean.expand(-1,-1,-1,nq) # b, t, 1, nq
        assert(Z_mean.shape==(b,t,1,nq))
        assert(Z_shape.shape==(b,t,1,nq))

        return Z_mean + Z_shape

