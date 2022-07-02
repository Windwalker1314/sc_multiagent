import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as f
from network.transformers.qatten_w import Qatten_w
from network.transformers.si_w import SI_Weight

class DQATTEN(nn.Module):
    def __init__(self, input_shape,args):
        super(DQATTEN, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.action_dim = args.n_agents * self.n_actions

        self.atten_w = Qatten_w(args)
        self.si_w = SI_Weight(args)

    def forward(self, z_values, states, actions = None, max_q_i = None, is_v=False):
        b, t, n, nq = z_values.shape
        Z_total = z_values.sum(dim=2,keepdim=True) # b, t, 1, nq

        q_vals = z_values.mean(dim=3) # b, t, n
        q_total = q_vals.sum(dim=2,keepdim=True).unsqueeze(3).expand(-1,-1,-1,nq) # b, t, 1, nq
        Q_atten = self.forward_qatten(q_vals, states, actions,max_q_i,is_v)
        Q_atten = Q_atten.reshape(b, t, 1, 1).expand(-1,-1,-1,nq) # b, t, 1, nq
        Z_total = Z_total.reshape(b, t, 1, nq)
        q_total = q_total.reshape(b, t, 1, nq)
        if is_v:
            return Z_total - q_total + Q_atten
        else:
            return Q_atten
    
    def forward_qatten(self,agent_qs, states, actions = None, max_q_i = None, is_v=False):
        w, b = self.atten_w(agent_qs, states)
        w = w.view(-1, self.n_agents) +1e-10
        b = b.view(-1, 1).repeat(1, self.n_agents)
        b/= self.n_agents
        agent_qs = agent_qs.view(-1, self.n_agents)
        agent_qs = w * agent_qs + b
        if not is_v:
            max_q_i = max_q_i.view(-1, self.n_agents)
            max_q_i = w * max_q_i + b
        y = self.calc(agent_qs,states,actions=actions,max_q_i=max_q_i, is_v=is_v)
        return y
    
    def calc(self,agent_qs,states, actions=None,max_q_i=None, is_v=False):
        if is_v:
            return self.calc_v(agent_qs)
        else:
            return self.calc_adv(agent_qs, states,actions,max_q_i)

    def calc_v(self,agent_qs):
        return torch.sum(agent_qs, dim=-1)
    
    def calc_adv(self,agent_qs, states,actions, max_q_i):
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)
        max_q_i = max_q_i.view(-1, self.n_agents)

        adv_q = (agent_qs - max_q_i).view(-1, self.n_agents).detach()

        adv_w_final = self.si_w(states, actions)
        adv_w_final = adv_w_final.view(-1, self.n_agents)

        adv_tot = torch.sum(adv_q * adv_w_final, dim=1)
        return adv_tot
