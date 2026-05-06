import torch

class SABRERolloutStorage:
    def __init__(self, 
                 storage_num_steps, 
                 num_processes, 
                 observation_space, 
                 action_space, 
                 hidden_dim, 
                 lstm_num_layers, 
                 num_frame_stack, 
                 device):
        self.step = 0
        self.storage_num_steps = storage_num_steps
        self.num_processes = num_processes
        self.num_frame_stack = num_frame_stack
        self.device = device

        # 1. 动态存储观测 (基于 observation_space)
        # 预期的 obs 已经是堆叠后的，形状为 [B, L, ...]
        self.obs = {}
        for key, space in observation_space.spaces.items():
            # 形状计算：[Steps + 1, num_processes, num_frame_stack, *feature_dims]
            # 注意：如果环境返回的已经是堆叠好的，space.shape 已经是 (L, ...)
            full_shape = (storage_num_steps + 1, num_processes, *space.shape)
            self.obs[key] = torch.zeros(*full_shape).to(device)

        # 2. 动态存储动作 (基于 action_space)
        self.actions = {}
        for key, space in action_space.spaces.items():
            # 动作形状：[Steps, num_processes, 1] (Discrete)
            self.actions[key] = torch.zeros(storage_num_steps, num_processes, 1).long().to(device)

        # 3. 存储隐藏状态 (h, c)
        self.recurrent_hidden_states = torch.zeros(storage_num_steps + 1, lstm_num_layers, num_processes, hidden_dim).to(device)
        self.recurrent_cell_states = torch.zeros(storage_num_steps + 1, lstm_num_layers, num_processes, hidden_dim).to(device)

        # 4. 存储掩码
        # recurrent_masks 用于输入 LSTM ([Steps+1, B, 1])
        self.recurrent_masks = torch.ones(storage_num_steps + 1, num_processes, 1).to(device)
        # 用于优势计算：1.0 表示未结束，0.0 表示 Episode 结束
        self.gae_masks = torch.ones(storage_num_steps + 1, num_processes, 1).to(device)

        # 5. 其他常规数据
        self.rewards = torch.zeros(storage_num_steps, num_processes, 1).to(device)
        self.value_preds = torch.zeros(storage_num_steps + 1, num_processes, 1).to(device)
        self.returns = torch.zeros(storage_num_steps + 1, num_processes, 1).to(device)
        self.action_log_probs = torch.zeros(storage_num_steps, num_processes, 1).to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs, value_preds, rewards, dones):
        """
        obs: OrderedDict [B, L, ...]
        actions: Dict
        """
        # 1. 更新观测 (动态遍历观测字典)
        for k in self.obs.keys():
            self.obs[k][self.step + 1].copy_(obs[k])

        # 2. 更新动作 (动态遍历动作字典)
        for k in self.actions.keys():
            self.actions[k][self.step].copy_(actions[k])

        # 3. 更新隐藏状态
        h, c = recurrent_hidden_states
        self.recurrent_hidden_states[self.step + 1].copy_(h)
        self.recurrent_cell_states[self.step + 1].copy_(c)

        # 4. 处理单步和堆叠掩码
        current_mask = torch.FloatTensor(1.0 - dones).view(-1, 1).to(self.device)
        
        self.recurrent_masks[self.step + 1].copy_(current_mask)
        
        # 5. GAE 掩码
        self.gae_masks[self.step + 1].copy_(current_mask)

        # 6. 其他信号
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)

        self.step = (self.step + 1) % self.storage_num_steps
    
    def after_update(self):
        """
        在 PPO 更新后执行，将最后一个时间步的数据拷贝到第 0 位，
        以便下一轮 rollout 采样能够接续状态。
        """
        # 1. 拷贝观测值 (obs)
        for k in self.obs.keys():
            self.obs[k][0].copy_(self.obs[k][-1])

        # 2. 拷贝隐藏状态 (h, c)
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.recurrent_cell_states[0].copy_(self.recurrent_cell_states[-1])

        # 3. 拷贝掩码 (masks)
        self.recurrent_masks[0].copy_(self.recurrent_masks[-1])
        self.gae_masks[0].copy_(self.gae_masks[-1])
        
        # 4. 重置步数计数器
        self.step = 0

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.num_processes
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)

        for start_ind in range(0, num_processes, num_envs_per_batch):
            indices = perm[start_ind:start_ind + num_envs_per_batch]
            T, N = self.storage_num_steps, len(indices)

            # 准备观测块 (保留原始特征维度)
            obs_batch = {}
            for k, v in self.obs.items():
                # v[:-1, indices] 形状: [T, N, L, ...]
                # view 为 [T*N, L, ...]
                obs_batch[k] = v[:-1, indices].view(T * N, *v.shape[2:])
            
            recurrent_mask_batch = self.recurrent_masks[:-1, indices].view(T * N, -1)
            
            h_init = self.recurrent_hidden_states[0, :, indices]
            c_init = self.recurrent_cell_states[0, :, indices]
            recurrent_hidden_states_batch = (h_init, c_init)

            # 动作块展平 [T*N, 1]
            actions_batch = {k: v[:, indices].view(T * N, -1) for k, v in self.actions.items()}
            
            # 其他常规数据展平 [T*N, 1]
            value_preds_batch = self.value_preds[:-1, indices].view(T * N, -1)
            return_batch = self.returns[:-1, indices].view(T * N, -1)
            old_action_log_probs_batch = self.action_log_probs[:, indices].view(T * N, -1)
            adv_targ = advantages[:, indices].view(T * N, -1)

            yield (obs_batch, 
                   recurrent_hidden_states_batch, 
                   recurrent_mask_batch, 
                   actions_batch, 
                   value_preds_batch, 
                   return_batch, 
                   old_action_log_probs_batch, 
                   adv_targ)