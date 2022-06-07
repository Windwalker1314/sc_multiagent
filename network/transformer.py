import torch.nn as nn
import torch.nn.functional as f
import torch

class Transformer(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(Transformer, self).__init__()
        self.args = args
        self.enemy_f_start = args.move_feats_dim
        self.n_enemy, self.enemy_dim = args.enemy_feats_dim
        self.enemy_f_size = self.n_enemy * self.enemy_dim

        self.ally_f_start = self.enemy_f_size + self.enemy_f_start
        self.n_ally, self.ally_dim = args.ally_feats_dim
        self.ally_f_size = self.n_ally * self.ally_dim

        self.other_f_start = self.ally_f_size + self.ally_f_start
        self.m_f_size = input_shape - self.enemy_f_size - self.ally_f_size

        self.n_agents = self.n_ally+self.n_enemy+1

        self.emb_dim = 32
        self.emb_m = nn.Linear(self.m_f_size, self.emb_dim)
        self.emb_e = nn.Linear(self.enemy_dim, self.emb_dim)
        self.emb_a = nn.Linear(self.ally_dim, self.emb_dim)

        self.emb_p = nn.Embedding(self.n_agents, self.emb_dim)  # n_agents emb_dim

        self.qkv_dim = 32
        self.w_q = nn.Linear(self.emb_dim, self.qkv_dim)
        self.w_k = nn.Linear(self.emb_dim, self.qkv_dim)
        self.w_v = nn.Linear(self.emb_dim, self.qkv_dim)

        self.attn = nn.MultiheadAttention(self.qkv_dim, 4)

        self.fc1 = nn.Linear(self.qkv_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs, hidden_state):
        b_size = obs.shape[0]
        
        m_move = obs[:, :self.args.move_feats_dim]
        m_other = obs[:, self.other_f_start:]
        m = torch.cat([m_move,m_other],dim=1)
        
        enemies = obs[:, self.enemy_f_start:self.ally_f_start].view(b_size, self.n_enemy, self.enemy_dim)
        allies = obs[:, self.ally_f_start:self.other_f_start].view(b_size, self.n_ally, self.ally_dim )
        positions = torch.LongTensor(torch.arange(0, self.n_agents).repeat(b_size,1))

        enemy_embs = f.relu(self.emb_e(enemies))
        ally_embs = f.relu(self.emb_a(allies))
        m_embs = f.relu(self.emb_m(m)).unsqueeze(1)
        p_emb = f.relu(self.emb_p(positions))
        x = torch.cat([m_embs,ally_embs,enemy_embs], dim=1) + p_emb

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        o, w = self.attn(q,k,v)
        o = o[:,0,:]
        x = self.fc1(o)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h