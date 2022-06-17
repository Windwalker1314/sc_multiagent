import torch.nn as nn
import torch
import torch.nn.functional as f


class DQMIX(nn.Module):
    def __init__(self, input_shape, args):
        super(DQMIX, self).__init__()
        self.args = args

        self.state_w1 = nn.Linear(args.state_shape, args.n_agents * args.qmix_hidden_dim)
        self.state_w2 = nn.Linear(args.state_shape, args.qmix_hidden_dim * 1)

        self.state_b1 = nn.Linear(args.state_shape, args.qmix_hidden_dim)
        self.state_b2 =nn.Sequential(nn.Linear(args.state_shape, args.qmix_hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(args.qmix_hidden_dim, 1))

        self.fc1 = nn.Linear(args.attention_dim, input_shape)

    def forward(self, z_values, states):
        b, t, n, nq = z_values.shape
        Z_total = z_values.sum(dim=2,keepdim=True) # b, t, 1, nq

        q_vals = z_values.mean(dim=3) # b, t, n
        q_total = q_vals.sum(dim=2,keepdim=True).unsqueeze(3).expand(-1,-1,-1,nq) # b, t, 1, nq
        Q_mix = self.forward_qmix(q_vals, states).expand(-1,-1,-1,nq) # b, t, 1, nq

        Z_total = Z_total.reshape(b, t, 1, nq)
        q_total = q_total.reshape(b, t, 1, nq)
        return Z_total - q_total + Q_mix

        # q_mixture : b, t, 1, nq  (sum n)
        # q_vals_expected : b, t, n, 1 (mean nq)
        # q_vals_sum : b, t, 1, 1  (Q_total)
        # q_joint_expected: b, t, 1, 1 (Q mix, from q_vals_expected)
        # q_mixture - q_vals_sum + q_joint_expected (Ztotal - Q_total + Q_mix) = Z mix
        # Z_att_total - E(Z_att_total) + Q_att_mix
    
    def forward_qmix(self, q_values, states):
        n = q_values.shape[-1]
        b, t, s = states.shape
        q_values = q_values.view(-1, 1, n) # bt, 1, n
        states = states.reshape(-1, s)  # bt, s
        w1 = torch.abs(self.state_w1(states))
        b1 = self.state_b1(states)
        w1 = w1.view(-1, n, self.args.qmix_hidden_dim)
        b1 = b1.view(-1, 1, self.args.qmix_hidden_dim)

        x = f.elu(torch.bmm(q_values, w1) +b1)

        w2 = torch.abs(self.state_w2(states))
        w2 = w2.view(-1, self.args.qmix_hidden_dim, 1)
        b2 = self.state_b2(states).view(-1, 1, 1)

        out = torch.bmm(x, w2) +b2

        qmix = out.view(b,t, 1,1)
        return qmix