from os import stat
from tkinter import W
import numpy as np
from requests import head
import torch.nn as nn
import torch
import torch.nn.functional as f


class DQATTEN(nn.Module):
    def __init__(self, input_shape, args):
        super(DQATTEN, self).__init__()
        self.args = args

        self.querys_nn = nn.ModuleList()
        self.keys_nn = nn.ModuleList()
        self.n_head = 4
        self.qkv_emb_dim = 32

        for i in range(self.n_head):
            wq = nn.Sequential(nn.Linear(self.args.state_shape, self.qkv_emb_dim, bias=False),
                               nn.ReLU())
            self.querys_nn.append(wq)
            wk = nn.Linear(self.args.obs_shape, self.qkv_emb_dim, bias=False)
            self.keys_nn.append(wk)
        self.state_b = nn.Sequential(nn.Linear(self.args.state_shape, self.qkv_emb_dim),
                               nn.ReLU(),
                               nn.Linear(self.qkv_emb_dim, 1))

        self.fc1 = nn.Linear(args.attention_dim, input_shape)

    def forward(self, z_values, states, obs):
        b, t, n, nq = z_values.shape
        q_vals = z_values.mean(dim=3) # b, t, n

        w, b = self.get_weights(q_vals, states, obs)
        w = w.reshape(-1, t, n)
        w = w.unsqueeze(3).expand(-1,-1,-1,nq) + 1e-10
        b = b.reshape(-1, t, 1)
        b = b.unsqueeze(3).expand(-1,-1,-1,nq)
        Z_weighted = w*z_values + b
        Z_total = Z_weighted.sum(dim=2, keepdim = True)

        return Z_total 

    
    def get_weights(self, q_values, states, obs):
        b1, t1, s = states.shape
        b, t, n, o = obs.shape
        assert (q_values.shape==(b,t,n))
        assert (b1==b and t1==t)
        states = states.reshape(-1, s)
        u = obs.reshape(-1, n, o)
        u = u.permute(1,0,2) # n, bt, o
        assert(states.shape == (b*t, s))
        assert(u.shape == (n, b*t, o))
        q_values = q_values.view(-1,1,n) # bt, 1, n

        all_querys_out = [wq(states) for wq in self.querys_nn]
        all_keys_out = [[wk(ui) for ui in u] for wk in self.keys_nn]

        atten_weights = []
        for k, q, in zip(all_keys_out, all_querys_out):
            # (bt, 1, qkv) * (bt, qkv, n)
            kT = torch.stack(k).permute(1,2,0)
            assert(kT.shape==(b*t, self.qkv_emb_dim, n))
            atten_out = torch.matmul(q.view(-1, 1, self.qkv_emb_dim),kT)  # bt, 1, n
            assert(atten_out.shape==(b*t,1,n))
            scaled_atten_out = atten_out/np.sqrt(self.qkv_emb_dim)

            scaled_atten_out[q_values<=-999999] = -999999
            
            atten_w = f.softmax(scaled_atten_out, dim =2) # bt, 1, n
            atten_weights.append(atten_w)
            assert(atten_w.shape==(b*t, 1, n))
        atten_weights = torch.stack(atten_weights,dim=1).squeeze(2)  # bt, h, n
        assert(atten_weights.shape == (b*t, self.n_head, n))
        head_atten = torch.sum(atten_weights, dim = 1) # bt, n
        v = self.state_b(states).view(-1,1) # bt, 1
        assert (head_atten.shape==(b*t, n))
        return head_atten, v