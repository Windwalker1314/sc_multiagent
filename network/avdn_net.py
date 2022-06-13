import torch.nn as nn
import torch
import torch.nn.functional as f


class AVDNNet(nn.Module):
    def __init__(self, input_shape, args):
        super(AVDNNet, self).__init__()
        self.encoding = nn.Linear(input_shape, args.rnn_hidden_dim)  # 对所有agent的obs解码
        self.q = nn.Linear(args.rnn_hidden_dim, args.attention_dim, bias=False)
        self.k = nn.Linear(args.rnn_hidden_dim, args.attention_dim, bias=False)
        self.v = nn.Linear(args.rnn_hidden_dim, args.attention_dim)
        self.attn = nn.MultiheadAttention(args.attention_dim, 2)
        self.fc1 = nn.Linear(args.attention_dim, input_shape)

    def forward(self, q_values):
        b, t, n, nq = q_values.shape
        x = q_values.reshape(b*t, n, nq)
        obs_emb = f.relu(self.encoding(x))

        q = self.q(obs_emb)
        k = self.k(obs_emb)
        v = self.v(obs_emb)

        o, w = self.attn(q,k,v)
        o = o.sum(dim=1)  # b*t, nq
        out = f.relu(self.fc1(o))
        out = out.reshape(b, t, 1, nq)
        return out