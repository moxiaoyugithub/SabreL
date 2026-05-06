import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

class InstructionPointerHead(nn.Module):
    """
    指令位置选择头
    """
    def __init__(self, autoregressive_embedding_dim, function_PalmTree_embedding_dim, 
                 action_type_num, max_instructions, hidden_dim=256, temperature=1.0):
        super().__init__()
        self.param_name = "selected_instruction"

        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.max_instructions = max_instructions
        
        # 指令编码
        self.instruction_proj = nn.Linear(function_PalmTree_embedding_dim, hidden_dim)
        
        # 自回归嵌入投
        self.project_in = nn.Linear(autoregressive_embedding_dim, hidden_dim)

        # 动作类型嵌入
        self.action_embed = nn.Linear(action_type_num, hidden_dim)
        
        # 基本块处理
        self.block_proj = nn.Linear(function_PalmTree_embedding_dim, hidden_dim)
        
        # 多头注意力
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # 自回归嵌入更新投影
        self.project_out = nn.Linear(max_instructions + 1, autoregressive_embedding_dim)
    
    def forward(self, 
                obf_params, 
                autoregressive_embedding,   # [B, autoregressive_embedding_dim]
                function_PalmTree_embedding,# [B, L, max_blocks, max_instructions, function_PalmTree_embedding_dim]无效位置填充0
                function_PalmTree_mask,     # [B, L, max_blocks, max_instructions], 1表示有效，0表示填充
                selected_basic_block_idx,   # [B, 1]
                action_type_one_hot         # [B, action_type_num]
                ):
        return self.act(
                        obf_params, 
                        autoregressive_embedding,   # [B, autoregressive_embedding_dim]
                        function_PalmTree_embedding,# [B, L, max_blocks, max_instructions, function_PalmTree_embedding_dim]无效位置填充0
                        function_PalmTree_mask,     # [B, L, max_blocks, max_instructions], 1表示有效，0表示填充
                        selected_basic_block_idx,   # [B, 1]
                        action_type_one_hot         # [B, action_type_num]
                        )
    
    def _get_dist(self, 
                  obf_params, 
                  autoregressive_embedding,   # [B, autoregressive_embedding_dim]
                  function_PalmTree_embedding,# [B, L, max_blocks, max_instructions, function_PalmTree_embedding_dim]无效位置填充0
                  function_PalmTree_mask,     # [B, L, max_blocks, max_instructions], 1表示有效，0表示填充
                  selected_basic_block_idx,   # [B, 1]
                  action_type_one_hot         # [B, action_type_num]
                ):
        device = autoregressive_embedding.device
        
        # 从帧堆叠的输入中，取出当前帧
        function_PalmTree_embedding = function_PalmTree_embedding[:, -1, :, :, :]
        function_PalmTree_mask = function_PalmTree_mask[:, -1, :, :]
        
        # 1. 获取选中的基本块（key）
        B, num_blocks, max_instructions, function_PalmTree_embedding_dim = function_PalmTree_embedding.shape
        selected_basic_block_idx_expanded = selected_basic_block_idx.view(B, 1, 1, 1)
        selected_basic_block_idx_expanded = selected_basic_block_idx_expanded.expand(
            -1, -1, self.max_instructions, function_PalmTree_embedding_dim
        )
        instructions_embedding = torch.gather(
            function_PalmTree_embedding, 1, selected_basic_block_idx_expanded
        ).squeeze(1)  # [B, max_instructions, function_PalmTree_embedding_dim]
        
        # 2. 获取选中基本块的有效掩码
        mask_idx_expanded = selected_basic_block_idx.view(B, 1, 1)
        mask_idx_expanded = mask_idx_expanded.expand(
            -1, -1, self.max_instructions
        )
        instructions_mask = torch.gather(
            function_PalmTree_mask, 1, mask_idx_expanded
        ).squeeze(1)  # [B, max_instructions]
        
        # 3. 计算实际的有效指令数
        valid_instruction_counts = instructions_mask.sum(dim=1)  # [B]
        
        # 4. 平均池化求出基本块嵌入（只对有效位置进行池化）
        # 将掩码扩展维度以匹配嵌入维度
        instructions_mask_expanded = instructions_mask.unsqueeze(-1).expand(-1, -1, function_PalmTree_embedding_dim)  # [B, max_instructions, function_PalmTree_embedding_dim]
        selected_block_embedding = torch.where(
            valid_instruction_counts.unsqueeze(-1) > 0,
            torch.sum(instructions_embedding * instructions_mask_expanded, dim=1)/valid_instruction_counts.unsqueeze(-1),
            torch.zeros_like(instructions_embedding[:, 0])       # 0除时返回0向量
        )  # [B, function_PalmTree_embedding_dim]
        
        # 5. 投影自回归嵌入
        z = F.relu(self.project_in(autoregressive_embedding))  # [B, hidden_dim]

        # 6. 嵌入动作类型信息
        action_type_embed = F.relu(self.action_embed(action_type_one_hot))  # [B, hidden_dim]

        # 7. 处理基本块嵌入
        selected_block_project = F.relu(self.block_proj(selected_block_embedding))  # [B, hidden_dim]
        
        # 8. 计算查询向量
        combined_query = z + selected_block_project + action_type_embed
        instructions_query = combined_query.unsqueeze(1)  # [B, 1, hidden_dim]

        # 9. 将选中的基本块特征投影到hidden_dim（key）
        instructions_project = F.relu(self.instruction_proj(instructions_embedding))  # [B, max_instructions, hidden_dim]
        
        # 10. 创建注意力掩码
        # 注意力掩码的形状需要是 [B, max_instructions]（对于batch_first=True）
        # 我们需要将有效位置设为False（不mask），无效位置设为True（mask）
        attention_mask = ~instructions_mask.bool()  # [B, max_instructions]，True表示填充位置（需要mask）
        
        # 11. 多头注意力（使用掩码过滤填充位置）
        attention_output, attention_weights = self.attention(
            instructions_query, 
            instructions_project, 
            instructions_project,
            key_padding_mask=attention_mask  # 使用key_padding_mask来mask无效位置
        )
        
        selected_instruction_logits = attention_weights.squeeze(1)  # [B, max_instructions]
        selected_instruction_logits = torch.log(selected_instruction_logits + 1e-9) # pytorch的attention内部应用了softmax，需要使用log将其映射回原始预测空间
        
        # 再填充确保无效位置不会有概率
        selected_instruction_logits = selected_instruction_logits.masked_fill(
            ~instructions_mask.bool(), float('-1e9'))
        
        # logits填充到模型预设的 max_instructions
        # pad_len = self.max_instructions - selected_instruction_logits.size(-1)
        
        # if pad_len > 0:
        #     # F.pad 的参数是从最后一维开始的 (左填充, 右填充)
        #     # 我们在右侧填充极小值，确保这些位置在 softmax 后概率为 0
        #     selected_instruction_logits = F.pad(selected_instruction_logits, (0, pad_len), value=float('-1e9'))

        # 12. 扩展并填充NO_OP位置
        B, _ = selected_instruction_logits.shape

        # 获取批量中哪些样本需要指令位置参数
        need_instruction_mask = obf_params[self.param_name]  # [B] 布尔张量

        # 为所有样本创建基础形状
        no_op_logits = torch.zeros((B, 1), device=device)
        other_logits = torch.full((B, self.max_instructions), float('-1e9'), device=device)

        # 创建最终的logits张量
        expanded_logits = torch.cat([no_op_logits, other_logits], dim=-1)  # [B, max_instructions + 1]

        # 对于需要指令位置参数的样本，更新logits
        if need_instruction_mask.any():
            need_idx = torch.where(need_instruction_mask)[0]    # 获取需要指令位置的样本索引
            expanded_logits[need_idx, 0] = float('-1e9')        # 这些样本的NO_OP位置应该设为极小的负数
            expanded_logits[need_idx, 1:] = selected_instruction_logits[need_idx]   # 这些样本的指令位置使用真实的logits
        # 对于不需要指令位置参数的样本，NO_OP位置保持0，其他位置保持-1e9
        selected_instruction_logits = expanded_logits  # [B, max_instructions + 1]

        return selected_instruction_logits, distributions.Categorical(logits=selected_instruction_logits / self.temperature)
    
    def evaluate_actions(self, 
                         selected_instruction_idx, 
                         obf_params, 
                         autoregressive_embedding,   # [B, autoregressive_embedding_dim]
                         function_PalmTree_embedding,# [B, L, max_blocks, max_instructions, function_PalmTree_embedding_dim]无效位置填充0
                         function_PalmTree_mask,     # [B, L, max_blocks, max_instructions], 1表示有效，0表示填充
                         selected_basic_block_idx,   # [B, 1]
                         action_type_one_hot         # [B, action_type_num]
            ):
        selected_instruction_logits, dist = self._get_dist(
                       obf_params, 
                       autoregressive_embedding,   # [B, autoregressive_embedding_dim]
                       function_PalmTree_embedding,# [B, L, max_blocks, max_instructions, function_PalmTree_embedding_dim]无效位置填充0
                       function_PalmTree_mask,     # [B, L, max_blocks, max_instructions], 1表示有效，0表示填充
                       selected_basic_block_idx,   # [B, 1]
                       action_type_one_hot         # [B, action_type_num]
        )
        
        selected_instruction_log_prob = dist.log_prob(selected_instruction_idx.squeeze(-1)).unsqueeze(-1)
        selected_instruction_entropy = dist.entropy().mean()
        
        # 维持梯度链条
        t = self.project_out(selected_instruction_logits)   # [B, autoregressive_embedding_dim]
        updated_autoregressive_embedding = autoregressive_embedding + t
        
        return selected_instruction_log_prob, selected_instruction_entropy, updated_autoregressive_embedding
    
    def act(self, 
            obf_params, 
            autoregressive_embedding,   # [B, autoregressive_embedding_dim]
            function_PalmTree_embedding,# [B, L, max_blocks, max_instructions, function_PalmTree_embedding_dim]无效位置填充0
            function_PalmTree_mask,     # [B, L, max_blocks, max_instructions], 1表示有效，0表示填充
            selected_basic_block_idx,   # [B, 1]
            action_type_one_hot         # [B, action_type_num]
            ):
        selected_instruction_logits, dist = self._get_dist(
                       obf_params, 
                       autoregressive_embedding,   # [B, autoregressive_embedding_dim]
                       function_PalmTree_embedding,# [B, L, max_blocks, max_instructions, function_PalmTree_embedding_dim]无效位置填充0
                       function_PalmTree_mask,     # [B, L, max_blocks, max_instructions], 1表示有效，0表示填充
                       selected_basic_block_idx,   # [B, 1]
                       action_type_one_hot         # [B, action_type_num]
        )

        # 13. 采样获取选中的指令idx
        selected_instruction_idx = dist.sample().unsqueeze(-1)  # [B, 1]
        selected_instruction_log_prob = dist.log_prob(selected_instruction_idx.squeeze(-1)).unsqueeze(-1)
        selected_instruction_entropy = dist.entropy().mean()
        
        # 14. 更新自回归嵌入（加入选择结果预测信息）
        t = self.project_out(selected_instruction_logits)   # [B, autoregressive_embedding_dim]
        updated_autoregressive_embedding = autoregressive_embedding + t
        
        return selected_instruction_logits, selected_instruction_idx, selected_instruction_log_prob, selected_instruction_entropy, updated_autoregressive_embedding