import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import torch

class OBS_W(nn.Module):
    def __init__(self, args) -> None:
        super(OBS_W, self).__init__()
        self.args = args
        self.obs_shape = args.obs_shape
        self.n_head = args.n_head
        self.emb_dim = args.attention_dim
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        hypernet_emb = args.hypernet_emb

        self.key_embs = nn.ModuleList()
        self.query_embs = nn.ModuleList()

        for i in range(self.n_head):
            query_emb = nn.Sequential(nn.Linear(self.state_dim, hypernet_emb),
                                        nn.ReLU(),
                                        nn.Linear(hypernet_emb, self.emb_dim, bias=False))
            key_emb = nn.Linear(self.obs_shape, self.emb_dim, bias=False)
            self.key_embs.append(key_emb)
            self.query_embs.append(query_emb)
        self.w_head = nn.Sequential(nn.Linear(self.state_dim, hypernet_emb),
                                    nn.ReLU(),
                                    nn.Linear(hypernet_emb, self.n_head))
        self.b_head = nn.Sequential(nn.Linear(self.state_dim, hypernet_emb),
                                    nn.ReLU(),
                                    nn.Linear(hypernet_emb, 1))

    def forward(self, states, obs):
        b,t,n,o = obs.shape
        bs = b*t
        obs = obs.reshape(bs, n, o)
        obs = obs.permute(1, 0, 2)  # n, bs, o
        states = states.reshape(-1, self.state_dim) #bs, s

        all_head_querys = [q(states) for q in self.query_embs]
        all_head_keys = [[k(unit_obs) for unit_obs in obs] for k in self.key_embs]

        all_head_w = []
        for k, q in zip(all_head_keys, all_head_querys):
            # q: bs, e
            # k: n, bs, e
            # (batch_size, 1, embed_dim) * (batch_size, embed_dim, agent_num)
            w = torch.matmul(q.view(-1, 1, self.emb_dim),
                             torch.stack(k).permute(1,2,0))
            w = w/np.sqrt(self.emb_dim)
            w = f.softmax(w, dim=2) # b, 1, n
            all_head_w.append(w)
        all_head_w = torch.stack(all_head_w, dim=1) 
        all_head_w = all_head_w.view(-1, self.n_head, self.n_agents) #  bs, h, n

        w_head = torch.abs(self.w_head(states))
        w_head = w_head.view(-1, self.n_head, 1).repeat(1, 1, self.n_agents)
        #b_head = self.b_head(states).view(-1, 1).repeat(1, self.n_agents)

        all_head_w *= w_head
        obs_w = torch.sum(all_head_w, dim = 1) # bs, n

        return obs_w