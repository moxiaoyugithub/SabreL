import torch
import torch.nn as nn
import torch.optim as optim

class PPO():
    def __init__(self, 
                 agent, 
                 ppo_clip_param, 
                 ppo_epoch, 
                 ppo_num_mini_batch, 
                 ppo_value_loss_coef, 
                 ppo_entropy_coef,
                 ppo_adam_learning_rate=None, 
                 ppo_adam_epsilon=None, 
                 ppo_max_grad_norm=0.5, 
                 gamma=0.99, 
                 gae_lambda=0.95,
                 use_clipped_value_loss=True):
        self.agent = agent

        self.ppo_clip_param = ppo_clip_param
        self.ppo_epoch = ppo_epoch
        self.ppo_num_mini_batch = ppo_num_mini_batch

        self.ppo_value_loss_coef = ppo_value_loss_coef
        self.ppo_entropy_coef = ppo_entropy_coef

        self.ppo_max_grad_norm = ppo_max_grad_norm
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(agent.parameters(), lr=ppo_adam_learning_rate, eps=ppo_adam_epsilon)

    def compute_gae_advantages(self, rollouts):
        """
        核心：实时计算 GAE 优势值和 Returns
        """
        with torch.no_grad():
            # 获取最后一个时间步的 Value
            next_value = self.agent.get_value(
                function_PalmTree_embedding=rollouts.obs["function_PalmTree_embedding"][-1],
                function_PalmTree_mask=rollouts.obs["function_PalmTree_mask"][-1],
                function_LLM_embedding=rollouts.obs.get("function_LLM_embedding")[-1] if "function_LLM_embedding" in rollouts.obs else None,
                recurrent_hidden_states_in=(rollouts.recurrent_hidden_states[-1], rollouts.recurrent_cell_states[-1]),
                recurrent_mask_in=rollouts.recurrent_masks[-1]
            )

            gae = 0
            # 逆向回溯计算回报和优势
            for step in reversed(range(rollouts.storage_num_steps)):
                # delta = r_t + gamma * V(s_t+1) * mask_t+1 - V(s_t)
                delta = (rollouts.rewards[step] + 
                         self.gamma * next_value * rollouts.gae_masks[step + 1] - 
                         rollouts.value_preds[step])
                
                # gae = delta + gamma * lambda * mask_t+1 * gae
                gae = delta + self.gamma * self.gae_lambda * rollouts.gae_masks[step + 1] * gae
                
                rollouts.returns[step] = gae + rollouts.value_preds[step]
                next_value = rollouts.value_preds[step]
            
            # 这能让 Value Loss 的目标值回归到均值为 0、标准差为 1 的区间
            # 从而让 MSE 损失的量级保持在 [0, 1] 附近
            #ret_mean = rollouts.returns[:-1].mean()
            #ret_std = rollouts.returns[:-1].std()
            #rollouts.returns[:-1] = (rollouts.returns[:-1] - ret_mean) / (ret_std + 1e-8)

            # 计算标准化优势值
            advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            return advantages

    def update(self, rollouts):
        # 1. 每次更新前重新计算优势和回报
        advantages = self.compute_gae_advantages(rollouts)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        # 2. 训练循环
        for e in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator(advantages, self.ppo_num_mini_batch)

            for sample in data_generator:
                (obs_batch, 
                 recurrent_hidden_states_batch, 
                 recurrent_masks_batch, 
                 actions_batch, 
                 value_preds_batch, 
                 return_batch, 
                 old_action_log_probs_batch, 
                 adv_targ) = sample

                # 评估动作
                new_log_probs, dist_entropy, values = self.agent.evaluate_actions(
                    actions=actions_batch,
                    function_PalmTree_embedding=obs_batch["function_PalmTree_embedding"],
                    function_PalmTree_mask=obs_batch["function_PalmTree_mask"],
                    function_LLM_embedding=obs_batch.get("function_LLM_embedding"),
                    recurrent_hidden_states_in=recurrent_hidden_states_batch,
                    recurrent_mask_in=recurrent_masks_batch,
                    junk_repeat_ratio=obs_batch["junk_repeat_ratio"],
                    available_actions_mask=obs_batch["available_actions_mask"]
                )

                # 策略 Loss
                ratio = torch.exp(new_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_param, 1.0 + self.ppo_clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                # 价值 Loss (Clipped)
                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.ppo_clip_param, self.ppo_clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                # 反向传播
                self.optimizer.zero_grad()
                total_loss = (action_loss + 
                              value_loss * self.ppo_value_loss_coef - 
                              dist_entropy * self.ppo_entropy_coef)
                total_loss.backward()
                
                # 裁剪梯度，确保二进制分析中深层 LSTM/Attention 的稳定性
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.ppo_max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.ppo_num_mini_batch
        return value_loss_epoch / num_updates, action_loss_epoch / num_updates, dist_entropy_epoch / num_updates