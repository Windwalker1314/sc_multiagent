import torch.nn as nn
import torch.nn.functional as f
import torch
import math

class IQNRNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args) -> None:
        super(IQNRNN,self).__init__()
        self.args = args
        self.qe = args.quantile_emb_dim # quantile embedding dimension
        self.nq = args.n_quantiles   # n quantiles
        self.ntq = args.n_target_quantiles  # n target quantiles
        self.naq = args.n_approx_quantiles  # n approx quantiles
        self.rhd = args.rnn_hidden_dim # rnn hidden dimension
        self.a = args.n_actions  # number of actions

        self.fc_obs = nn.Linear(input_shape, self.rhd)
        self.rnn = nn.GRUCell(self.rhd, self.rhd)
        self.phi = nn.Linear(self.qe, self.rhd)
        self.g = nn.Linear(self.rhd, self.a)
    
    def forward(self, obs, hidden_state, forward_type=None):
        x = f.relu(self.fc_obs(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        psi = self.rnn(x, h_in)
        if forward_type == "approx":
            nq = self.naq
        elif forward_type == "policy":
            nq = self.nq
        elif forward_type == "target":
            nq = self.ntq
        else:
            raise ValueError("Unknown forward type")
        if psi.shape[0] >= self.args.n_agents:
            b = psi.shape[0]//self.args.n_agents  # b x n_agents
            n = self.args.n_agents
        else:
            b = psi.shape[0]
            n = 1
        psi2 = psi.reshape(b*n,1,self.rhd).expand(-1,nq, -1).reshape(-1,self.rhd) # b*n*nq, rnn
        rnd_q = torch.rand(b * nq)
        if self.args.cuda:
            rnd_q = rnd_q.cuda()
        tau = rnd_q.view(b, 1, nq).expand(-1, n, -1)
        tau = tau.unsqueeze(3).expand(-1,-1,-1,self.qe).reshape(-1, self.qe)
        i = torch.arange(1,self.qe+1).view(1,-1).expand(b*n*nq, self.qe)
        if self.args.cuda:
            i = i.cuda()
        phi = f.relu(self.phi(torch.cos(math.pi * i * tau))) # b*n*nq, rnn
        Z_vals = self.g(psi2 * phi)
        Z_vals = Z_vals.view(b*n, nq, self.a).permute(0, 2, 1)
        rnd_q = rnd_q.view(b, nq)
        assert(Z_vals.shape==(b*n, self.a, nq))
        assert(psi.shape==(b*n,self.rhd))
        assert(rnd_q.shape==(b, nq))
        return Z_vals, psi, rnd_q

