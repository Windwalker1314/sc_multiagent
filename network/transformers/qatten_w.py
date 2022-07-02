import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

class Qatten_w(nn.Module):
    def __init__(self, args) -> None:
        super(Qatten_w, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions =args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.unit_dim = args.unit_dim
        self.sa_dim = self.state_dim + self.n_agents * self.n_actions
        self.n_head = args.n_head
        
        self.emb_dim = args.attention_dim
        self.atten_reg_coef = args.atten_reg_coef

        self.keys_nn = nn.ModuleList()
        self.querys_nn = nn.ModuleList()

        hypernet_emb = args.hypernet_emb
        for i in range(self.n_head):
            query_nn = nn.Sequential(nn.Linear(self.state_dim, hypernet_emb),
                                     nn.ReLU(),
                                     nn.Linear(hypernet_emb, self.emb_dim, bias=False))
            self.querys_nn.append(query_nn)
            self.keys_nn.append(nn.Linear(self.unit_dim,self.emb_dim,bias=False))
        self.hyper_w_head = nn.Sequential(nn.Linear(self.state_dim,hypernet_emb),
                                          nn.ReLU(),
                                          nn.Linear(hypernet_emb, self.n_head))
        self.b = nn.Sequential(nn.Linear(self.state_dim,self.emb_dim),
                               nn.ReLU(),
                               nn.Linear(self.emb_dim, 1))

    def forward(self, q_vals, states):
        states = states.reshape(-1, self.state_dim)
        unit_s = states[:, :self.unit_dim*self.n_agents]
        unit_s = unit_s.reshape(-1,self.n_agents, self.unit_dim)
        unit_s = unit_s.permute(1, 0, 2)  # n, b, emb

        q_vals = q_vals.view(-1, 1, self.n_agents)

        all_querys = [q_nn(states) for q_nn in self.querys_nn]
        all_keys = [[k_nn(u_s) for u_s in unit_s] for k_nn in self.keys_nn]

        all_atten_logits = []
        all_atten_w = []
        for k,q in zip(all_keys, all_querys):
            # (b, 1, emb) * (b, emb, n)
            atten_logit = torch.matmul(q.view(-1,1,self.emb_dim),
                                       torch.stack(k).permute(1,2,0))
            scaled_atten_logit = atten_logit / np.sqrt(self.emb_dim)

            atten_w = f.softmax(scaled_atten_logit, dim=2)
            all_atten_logits.append(atten_logit)
            all_atten_w.append(atten_w)
        all_atten_w = torch.stack(all_atten_w, dim=1)
        all_atten_w = all_atten_w.view(-1, self.n_head, self.n_agents)

        b = self.b(states).view(-1, 1)
        
        w_head = torch.abs(self.hyper_w_head(states))
        w_head = w_head.view(-1, self.n_head, 1).repeat(1,1,self.n_agents)
        all_atten_w *= w_head

        all_atten_w = torch.sum(all_atten_w, dim=1)

        return all_atten_w, b