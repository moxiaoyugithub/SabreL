import torch
import torch.nn as nn

from arch.decider.film import TacticalFusion

class Core(nn.Module):
    def __init__(self, function_PalmTree_embedding_dim, function_LLM_embedding_dim, hidden_dim, lstm_num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm_num_layers = lstm_num_layers
        
        # 1. 结构化特征预投影 (PalmTree 路)
        # 将 128 映射到内部计算维度，方便后续与 LLM context 进行门控操作
        self.pt_project = nn.Sequential(
            nn.Linear(function_PalmTree_embedding_dim, hidden_dim),
            nn.ReLU()
        )

        # 2. 门控融合层 (GLU 适配)
        if function_LLM_embedding_dim:
            self.with_LLM_embedding = True
            self.llm_project = nn.Linear(function_LLM_embedding_dim, hidden_dim)
            self.tactical_fusion = TacticalFusion(hidden_dim, 256)
        else:
            self.with_LLM_embedding = False

        # 3. 最终投影与归一化
        self.project_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 4. LSTM
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_layers=lstm_num_layers, 
            batch_first=True, 
            dropout=dropout if lstm_num_layers > 1 else 0
        )
    
    def forward(self, 
                function_PalmTree_embedding, 
                function_PalmTree_mask, 
                function_LLM_embedding=None, 
                recurrent_hidden_states_in=None, 
                recurrent_mask_in=None):
        
        # B: Batch size, L: Sequence length (unroll steps)
        B, L, max_blocks, max_instructions, pt_dim = function_PalmTree_embedding.shape
        
        # 折叠 B*L 维度以便并行处理
        pt_embed = function_PalmTree_embedding.view(B * L, max_blocks, max_instructions, pt_dim)
        pt_mask = function_PalmTree_mask.view(B * L, max_blocks, max_instructions)
        
        # --- 空间池化 ---
        valid_inst_counts = pt_mask.sum(dim=-1, keepdim=True)
        block_sums = torch.einsum('bijk,bij->bik', pt_embed, pt_mask.float())
        blocks_embedding = torch.where(valid_inst_counts > 0, block_sums / (valid_inst_counts + 1e-8), torch.zeros_like(block_sums))

        block_masks = (pt_mask.sum(dim=-1) > 0).float()
        valid_block_counts = block_masks.sum(dim=-1, keepdim=True)
        fn_sums = torch.einsum('bik,bi->bk', blocks_embedding, block_masks)
        fn_pt_embedding = torch.where(valid_block_counts > 0, fn_sums / (valid_block_counts + 1e-8), torch.zeros_like(fn_sums))

        # --- GLU 门控融合 ---
        # 映射 PalmTree 特征到目标空间
        x_pt = self.pt_project(fn_pt_embedding)
        
        if self.with_LLM_embedding:
            # LLM 特征作为 context 指导 GLU 的门控开关
            llm_embed = function_LLM_embedding.view(B * L, -1)
            x_llm = self.llm_project(llm_embed)
            x_fused = self.tactical_fusion(x_pt, x_llm)
        else:
            x_fused = x_pt
        
        # 归一化与 Dropout
        x = self.project_out(x_fused)

        # 还原帧维度并进入 LSTM
        x_input_rnn = x.view(B, L, -1)
        last_output, recurrent_hidden_states_out = self._forward_rnn(
            x_input_rnn, 
            recurrent_hidden_states_in, 
            recurrent_mask_in
        )

        return last_output, recurrent_hidden_states_out

    def _forward_rnn(self, x, hxs, masks):
        h_n, c_n = hxs
        B = h_n.size(1) # 环境数
        
        if x.size(0) == B:
            # --- 采样模式 ---
            # x: [B, L, D], masks: [B, 1]
            m = masks.view(1, -1, 1) # [1, B, 1]
            
            # 这里的 x 已经自带了 L 的长度
            # LSTM 会连续计算 L 次状态转移，捕捉这 L 帧内的代码修改趋势
            x_out, (h_n, c_n) = self.lstm(x, (h_n * m, c_n * m))
            
            # 返回最后一帧的输出作为决策特征
            return x_out[:, -1, :], (h_n, c_n)
            
        else:
            # --- 训练模式 ---
            T = int(x.size(0) / B)
            L = x.size(1) 
            
            # 还原为 [T, B, L, D] -> 转置为 [B, T, L, D]
            # 为了喂给 LSTM，需要合并 T 和 L
            x = x.view(T, B, L, -1).transpose(0, 1) 
            x = x.reshape(B, T * L, -1) # [B, T*L, D]
            
            masks = masks.view(T, B)

            # 寻找重置点
            has_zeros = ((masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu())
            if has_zeros.dim() == 0:
                has_zeros = [has_zeros.item() + 1] if has_zeros.numel() > 0 else []
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            segments = [0] + has_zeros + [T]
            outputs = []

            for i in range(len(segments) - 1):
                start_step = segments[i]
                end_step = segments[i + 1]
                
                # 对应的序列索引要乘以 L
                start_idx = start_step * L
                end_idx = end_step * L
                
                m = masks[start_step].view(1, -1, 1)

                rnn_output, (h_n, c_n) = self.lstm(
                    x[:, start_idx:end_idx], 
                    (h_n * m, c_n * m)
                )
                
                # 因为 Head 只需要每一步(Step)结束时的特征
                # 从训练序列中每隔 L 抽一帧出来
                step_outputs = rnn_output.view(B, -1, L, self.hidden_dim)
                outputs.append(step_outputs[:, :, -1, :]) # 取每一步的最后一帧

            # 还原 PPO 展平顺序 [T*B, D]
            lstm_output = torch.cat(outputs, dim=1) # [B, T, D]
            lstm_output = lstm_output.transpose(0, 1).contiguous()
            
            return lstm_output.view(T * B, -1), (h_n, c_n)