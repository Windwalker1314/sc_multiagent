import torch
import os
from network.iqnrnn import IQNRNN
from network.vdn_net import VDNNet
from network.avdn_net import AVDNNet
from network.transformer import Transformer
import torch.nn.functional as f

class DDN:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        input_shape = self.obs_shape

        self.nq = args.n_quantiles   # n quantiles
        self.ntq = args.n_target_quantiles  # n target quantiles
        self.naq = args.n_approx_quantiles  # n approx quantiles
        self.rhd = args.rnn_hidden_dim # rnn hidden dimension

        # 根据参数决定RNN的输入维度
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents

        self.eval_rnn = IQNRNN(input_shape, args)  # 每个agent选动作的网络
        self.target_rnn = IQNRNN(input_shape, args)
        # 神经网络
        if args.alg == 'ddn':
            self.eval_vdn_net = VDNNet()  # 把agentsQ值加起来的网络
            self.target_vdn_net = VDNNet()
        elif args.alg == 'dan':
            self.eval_vdn_net = AVDNNet(self.nq, args)
            self.target_vdn_net = AVDNNet(self.ntq, args)
        else:
            raise Exception("No such algorithm")
        
        self.args = args
        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_vdn_net.cuda()
            self.target_vdn_net.cuda()

        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map
        # 如果存在模型则加载模型
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
                path_rnn = self.model_dir + '/rnn_net_params.pkl'
                path_vdn = self.model_dir + '/vdn_net_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.eval_vdn_net.load_state_dict(torch.load(path_vdn, map_location=map_location))
                print('Successfully load the model: {} and {}'.format(path_rnn, path_vdn))
            else:
                raise Exception("No model!")

        # 让target_net和eval_net的网络参数相同
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_vdn_net.load_state_dict(self.eval_vdn_net.state_dict())

        self.eval_parameters = list(self.eval_vdn_net.parameters()) + list(self.eval_rnn.parameters())
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)


        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden、target_hidden
        self.eval_hidden = None
        self.target_hidden = None
        print('Init alg', self.args.alg)

    def learn(self, batch, max_episode_len, train_step, epsilon=None):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        '''
        在learn的时候，抽取到的数据是四维的，四个维度分别为 1——第几个episode 2——episode中第几个transition
        3——第几个agent的数据 4——具体obs维度。因为在选动作时不仅需要输入当前的inputs，还要给神经网络输入hidden_state，
        hidden_state和之前的经验相关，因此就不能随机抽取经验进行学习。所以这里一次抽取多个episode，然后一次给神经网络
        传入每个episode的同一个位置的transition
        '''
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():  # 把batch里的数据转化成tensor
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        # TODO pymarl中取得经验没有取最后一条，找出原因
        s, s_next, u, r, avail_u, avail_u_next, terminated = batch['s'], batch['s_next'], \
                                                             batch['u'], batch['r'],  batch['avail_u'], \
                                                             batch['avail_u_next'], batch['terminated']

        mask = 1 - batch["padded"].float()  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习
        if self.args.cuda:
            s = s.cuda()
            s_next = s_next.cuda()
            u = u.cuda()
            r = r.cuda()
            mask = mask.cuda()
            avail_u_next = avail_u_next.cuda()
            terminated = terminated.cuda()
        # 得到每个agent对应的Q值，维度为(episode个数, max_episode_len， n_agents，n_actions)
        Z_evals, Z_targets, rnd_qs, rnd_tqs = self.get_Z_values(batch, max_episode_len)
        # Z : (b, t, n, a, nq)
        # rnd_qs : (b, t, n, nq)
        action_for_zs = u.unsqueeze(4).expand(-1,-1,-1,-1,self.nq)
        chosen_action_Zs = torch.gather(Z_evals, dim=3, index = action_for_zs).squeeze(3)
       
        avail_actions = avail_u.unsqueeze(4).expand(-1,-1,-1,-1, self.nq)
        target_avail_actions = avail_u_next.unsqueeze(4).expand(-1, -1, -1, -1, self.ntq)
        Z_targets[target_avail_actions==0] = -9999999

        target_max_actions = Z_targets.mean(dim=4).max(dim=3,keepdim=True)[1]  # (b, t, n, 1) argmax E(Z)
        target_max_actions = target_max_actions.unsqueeze(4).expand(-1,-1,-1,-1, self.ntq) # (b,t,n,1,ntq) 
        target_max_Zs = torch.gather(Z_targets, dim=3, index=target_max_actions).squeeze(3) #(b,t,n,ntq)  

        # Mixer
        if self.args.alg == 'ddn':
            chosen_action_Z = self.eval_vdn_net(chosen_action_Zs)  # chosen_action_zs: (b, t, n, nq) -> (b,t,nq)
            target_max_Z = self.target_vdn_net(target_max_Zs)
        elif self.args.alg == 'dan':
            chosen_action_Z = self.eval_vdn_net(chosen_action_Zs, s)  # chosen_action_zs: (b, t, n, nq) -> (b,t,nq)
            target_max_Z = self.target_vdn_net(target_max_Zs, s_next)    # b, t, ntq
        
        targets = r.unsqueeze(3) + (self.args.gamma * (1-terminated)).unsqueeze(3) * target_max_Z
        # targets (b, t, 1, ntq)
        targets = targets.unsqueeze(3).expand(-1,-1,-1,self.nq,-1) # (b,t,1,nq,ntq)
        chosen_action_Z = chosen_action_Z.unsqueeze(4).expand(-1,-1,-1,-1,self.ntq) 

        delta = targets - chosen_action_Z
        tau = rnd_qs.unsqueeze(3).unsqueeze(2).expand(-1,-1,-1,-1,self.ntq)  #(b,t,nq,ntq)
        abs_weight = torch.abs(tau-delta.le(0.).float())
        y = torch.zeros(delta.shape)
        if self.args.cuda:
            y = y.cuda()
        loss = f.smooth_l1_loss(delta, y, reduction="none") # (b,t,1,nq,ntq)
        loss = (abs_weight * loss).mean(dim=4).sum(dim=3)
        assert(loss.shape==mask.shape)
        loss = loss*mask
        loss = loss.sum() / mask.sum()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()
        # Max over target Z-values

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_vdn_net.load_state_dict(self.eval_vdn_net.state_dict())

    def _get_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)

        # 给obs添加上一个动作、agent编号
        if self.args.last_action:
            if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        if self.args.reuse_network:
            # 因为当前的obs三维的数据，每一维分别代表(episode，agent，obs维度)，直接在dim_1上添加对应的向量
            # 即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
            # agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # 要把obs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成episode_num*n_agents条数据
        # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next

    def get_Z_values(self, batch, max_episode_len):
        b = batch['o'].shape[0]
        Z_evals = []
        Z_targets = []
        rnd_qs = []
        rnd_tqs = []
        for i in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, i)
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()
            Z_eval, self.eval_hidden, rnd_q = self.eval_rnn(inputs, self.eval_hidden, "policy")
            Z_target, self.target_hidden, rnd_tq = self.target_rnn(inputs_next,self.target_hidden, "target")
            Z_eval = Z_eval.view(b, self.n_agents, self.n_actions, self.nq)
            Z_target = Z_target.view(b, self.n_agents, self.n_actions, self.ntq)
            rnd_q = rnd_q.view(b, self.nq)
            rnd_tq = rnd_tq.view(b, self.ntq)
            Z_evals.append(Z_eval)
            Z_targets.append(Z_target)
            rnd_qs.append(rnd_q)
            rnd_tqs.append(rnd_tq)
        Z_evals = torch.stack(Z_evals, dim=1)
        Z_targets = torch.stack(Z_targets, dim=1)
        rnd_qs = torch.stack(rnd_qs, dim=1)
        rnd_tqs = torch.stack(rnd_tqs, dim=1)
        return Z_evals, Z_targets, rnd_qs, rnd_tqs

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_vdn_net.state_dict(), self.model_dir + '/' + num + '_vdn_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_net_params.pkl')
