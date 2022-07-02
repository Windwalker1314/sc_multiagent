import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

class SI_Weight(nn.Module):
    def __init__(self, args) -> None:
        super(SI_Weight, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.action_dim = args.n_agents*self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim

        self.n_head = args.n_head
        
        self.keys_nn = nn.ModuleList()
        self.agents_nn = nn.ModuleList()
        self.actions_nn = nn.ModuleList()

        adv_hypernet_emb = args.hypernet_emb
        for i in range(self.n_head):
            self.keys_nn.append(nn.Linear(self.state_dim, 1))
            self.agents_nn.append(nn.Linear(self.state_dim, self.n_agents))
            self.actions_nn.append(nn.Linear(self.state_action_dim, self.n_agents))
    
    def forward(self,states,actions):
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        states_actions = torch.cat([states,actions], dim=1)
        all_k = [k_nn(states) for k_nn in self.keys_nn]
        all_a = [a_nn(states) for a_nn in self.agents_nn]
        all_sa = [sa_nn(states_actions) for sa_nn in self.actions_nn]

        atten_w = []
        for k,a,sa in zip(all_k, all_a, all_sa):
            x_key = torch.abs(k).repeat(1, self.n_agents) + 1e-10
            x_agents = torch.sigmoid(a)
            x_action = torch.sigmoid(sa)
            w = x_key * x_agents * x_action
            atten_w.append(w)
        atten_w = torch.stack(atten_w, dim=1)
        atten_w = atten_w.view(-1, self.n_head, self.n_agents)
        atten_w = torch.sum(atten_w, dim=1)

        return atten_w