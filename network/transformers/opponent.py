import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

class OpponnetModelling(nn.Module):
    def __init__(self,input_shape,output_shape, args) -> None:
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
        
        self.n_head = args.n_head_om
        self.emb_dim = args.attention_dim
        hypernet_emb = args.hypernet_emb
        
        self.emb_m = nn.Sequential(nn.Linear(self.m_f_size, hypernet_emb),
                                   nn.ReLU(),
                                   nn.Linear(hypernet_emb, self.emb_dim))
        self.emb_e = nn.Linear(self.enemy_dim, self.emb_dim)
        self.emb_a = nn.Linear(self.ally_dim, self.emb_dim)
        self.emb_p = nn.Embedding(self.n_agents, self.emb_dim)

        self.w_q = nn.Linear(self.emb_dim, self.emb_dim)
        self.w_k = nn.Linear(self.emb_dim, self.emb_dim)
        self.w_v = nn.Linear(self.emb_dim, self.emb_dim)

        self.attn = nn.MultiheadAttention(self.emb_dim, self.n_head)
        self.final_layer = nn.Linear(self.emb_dim, output_shape)

    def forward(self, obs):
        bs, o = obs.shape
        m_move = obs[:, :self.args.move_feats_dim]
        m_other = obs[:, self.other_f_start:]
        m = torch.cat([m_move,m_other],dim=1)
        
        enemies = obs[:, self.enemy_f_start:self.ally_f_start].view(bs, self.n_enemy, self.enemy_dim)
        allies = obs[:, self.ally_f_start:self.other_f_start].view(bs, self.n_ally, self.ally_dim )

        positions = torch.LongTensor([0]+[1]*self.n_ally+[2]*self.n_enemy).repeat(bs,1)
        if self.args.cuda:
            positions = positions.cuda()
        enemy_embs = f.relu(self.emb_e(enemies))
        ally_embs = f.relu(self.emb_a(allies))
        m_embs = f.relu(self.emb_m(m)).unsqueeze(1)
        p_emb = f.relu(self.emb_p(positions))
        x = torch.cat([m_embs,ally_embs,enemy_embs], dim=1) + p_emb

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        o, w = self.attn(q,k,v)
        all_atten_scores = o[:,0,:]

        return self.final_layer(all_atten_scores)