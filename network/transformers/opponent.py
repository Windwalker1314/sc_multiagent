import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

class OpponnetModelling(nn.Module):
    def __init__(self,input_shape,args) -> None:
        super(OpponnetModelling,self).__init__()
        self.args= args
        # enemy
        self.enemy_f_start = args.move_feats_dim
        self.n_enemy, self.enemy_dim = args.enemy_feats_dim
        self.enemy_f_size = self.n_enemy * self.enemy_dim
        # ally
        self.ally_f_start = self.enemy_f_size + self.enemy_f_start
        self.n_ally, self.ally_dim = args.ally_feats_dim
        self.ally_f_size = self.n_ally * self.ally_dim
        # self
        self.other_f_start = self.ally_f_size + self.ally_f_start
        self.m_f_size = input_shape - self.enemy_f_size - self.ally_f_size
        self.n_agents = self.n_ally+self.n_enemy+1
        # transformers
        
        self.n_head = args.n_head
        self.emb_dim = args.attention_dim
        
        self.querys_nn = nn.ModuleList()
        self.keys_m_nn = nn.ModuleList()
        self.keys_ally_nn = nn.ModuleList()
        self.keys_enemy_nn = nn.ModuleList()

        self.vs_m = nn.ModuleList()
        self.vs_enemy = nn.ModuleList()
        self.vs_ally =nn.ModuleList()

        hypernet_emb = args.hypernet_emb
        for i in range(self.n_head):
            query_nn = nn.Sequential(nn.Linear(self.m_f_size, hypernet_emb),
                                     nn.ReLU(),
                                     nn.Linear(hypernet_emb, self.emb_dim, bias=False))
            key_m_nn = nn.Sequential(nn.Linear(self.m_f_size, hypernet_emb),
                                     nn.ReLU(),
                                     nn.Linear(hypernet_emb, self.emb_dim, bias=False))
            key_ally_nn = nn.Linear(self.ally_dim, self.emb_dim, bias=False)
            key_enemy_nn = nn.Linear(self.enemy_dim, self.emb_dim, bias=False)

            v_m = nn.Sequential(nn.Linear(self.m_f_size, hypernet_emb),
                                     nn.ReLU(),
                                     nn.Linear(hypernet_emb, self.emb_dim, bias=False))
            v_enemy = nn.Linear(self.enemy_dim, self.emb_dim, bias=False)
            v_ally = nn.Linear(self.ally_dim, self.emb_dim, bias=False)
            self.querys_nn.append(query_nn)
            self.keys_m_nn.append(key_m_nn)
            self.keys_ally_nn.append(key_ally_nn)
            self.keys_enemy_nn.append(key_enemy_nn)
            self.vs_m.append(v_m)
            self.vs_enemy.append(v_enemy)
            self.vs_ally.append(v_ally)
        self.b = nn.Sequential(nn.Linear(input_shape, self.emb_dim),
                               nn.ReLU(),
                               nn.Linear(self.emb_dim, 1))
        self.hyper_w_head = nn.Sequential(nn.Linear(input_shape,hypernet_emb),
                                          nn.ReLU(),
                                          nn.Linear(hypernet_emb, self.n_head))
        self.final_layer = nn.Linear(self.emb_dim* self.n_agents, args.rnn_hidden_dim)

    def forward(self, obs):
        bs, o = obs.shape
        m_move = obs[:, :self.args.move_feats_dim]
        m_other = obs[:, self.other_f_start:]
        m = torch.cat([m_move,m_other],dim=1)
        
        enemies = obs[:, self.enemy_f_start:self.ally_f_start].view(bs, self.n_enemy, self.enemy_dim)
        allies = obs[:, self.ally_f_start:self.other_f_start].view(bs, self.n_ally, self.ally_dim )

        all_querys = [q_nn(m) for q_nn in self.querys_nn]
        all_keys = []
        for km, ke, ka in zip(self.keys_m_nn, self.keys_enemy_nn, self.keys_ally_nn):
            keys = [km(m)] + [ke(enemies[:,i,:]) for i in range(self.n_enemy)] + [ka(allies[:,i,:]) for i in range(self.n_ally)]
            all_keys.append(keys)
        all_values = []
        for vm,ve,va in zip(self.vs_m,self.vs_enemy, self.vs_ally):
            v = [vm(m)] + [ve(enemies[:,i,:]) for i in range(self.n_enemy)] + [va(allies[:,i,:]) for i in range(self.n_ally)]
            v = torch.stack(v).permute(1,2,0) # b emb n
            all_values.append(v)
        # (b, 1, emb) * (b, emb, n)
        all_atten_scores = []
        for q, k, v in zip(all_querys, all_keys, all_values):
            atten_logit = torch.matmul(q.view(-1,1,self.emb_dim),
                                       torch.stack(k).permute(1,2,0))
            scaled_atten_logit = atten_logit / np.sqrt(self.emb_dim)

            atten_w = f.softmax(scaled_atten_logit, dim=2) # b,1,n * b,emb,n
            atten_v = atten_w.expand(-1,self.emb_dim, -1) * v

            all_atten_scores.append(atten_v)
        all_atten_scores = torch.stack(all_atten_scores, dim=1) # b h emb n
        all_atten_scores = all_atten_scores.view(-1, self.n_head, self.emb_dim*self.n_agents)

        b = self.b(obs).view(-1, 1)
        w_head = torch.abs(self.hyper_w_head(obs))
        w_head = w_head.view(-1, self.n_head, 1).repeat(1,1,self.emb_dim*self.n_agents)

        all_atten_scores *= w_head
        all_atten_scores = torch.sum(all_atten_scores, dim=1) + b

        return self.final_layer(all_atten_scores)