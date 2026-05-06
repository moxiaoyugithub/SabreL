import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

from arch.decider.glu import GLU

class JunkCodeHead(nn.Module):
    """
    垃圾代码选择头
    """
    def __init__(self, autoregressive_embedding_dim, function_PalmTree_embedding_dim, junk_num, hidden_dim=256):
        super().__init__()
        self.param_name = "junk"

        self.junk_num = junk_num
        
        # 自回归嵌入投影
        self.project_in = nn.Linear(autoregressive_embedding_dim, hidden_dim)
        
        # GLU层
        self.glu = GLU(input_size=hidden_dim, context_size=function_PalmTree_embedding_dim+junk_num, output_size=junk_num)

    def forward(self, 
                obf_params, 
                autoregressive_embedding,   # [B, autoregressive_embedding_dim]
                function_PalmTree_embedding,# [B, L, max_blocks, max_instructions, function_PalmTree_embedding_dim]无效位置填充0
                function_PalmTree_mask,     # [B, L, max_blocks, max_instructions], 1表示有效，0表示填充
                selected_basic_block_idx,   # [B, 1]
                junk_repeat_ratio           # [B, L, junk_num]重复率
                ):
        return self.act(
                  obf_params, 
                  autoregressive_embedding,   # [B, autoregressive_embedding_dim]
                  function_PalmTree_embedding,# [B, L, max_blocks, max_instructions, function_PalmTree_embedding_dim]无效位置填充0
                  function_PalmTree_mask,     # [B, L, max_blocks, max_instructions], 1表示有效，0表示填充
                  selected_basic_block_idx,   # [B, 1]
                  junk_repeat_ratio           # [B, L, junk_num]重复率
                )
    
    def _get_dist(self, 
                  obf_params, 
                  autoregressive_embedding,   # [B, autoregressive_embedding_dim]
                  function_PalmTree_embedding,# [B, L, max_blocks, max_instructions, function_PalmTree_embedding_dim]无效位置填充0
                  function_PalmTree_mask,     # [B, L, max_blocks, max_instructions], 1表示有效，0表示填充
                  selected_basic_block_idx,   # [B, 1]
                  junk_repeat_ratio           # [B, L, junk_num]重复率
                ):
        device = autoregressive_embedding.device
        
        # 从帧堆叠的输入中，取出当前帧
        function_PalmTree_embedding = function_PalmTree_embedding[:, -1, :, :, :]
        function_PalmTree_mask = function_PalmTree_mask[:, -1, :, :]
        junk_repeat_ratio = junk_repeat_ratio[:, -1, :]
        
        # 1. 获取选中的基本块（key）
        B, num_blocks, max_instructions, function_PalmTree_embedding_dim = function_PalmTree_embedding.shape
        selected_basic_block_idx_expanded = selected_basic_block_idx.view(B, 1, 1, 1)
        selected_basic_block_idx_expanded = selected_basic_block_idx_expanded.expand(
            -1, -1, max_instructions, function_PalmTree_embedding_dim
        )
        instructions_embedding = torch.gather(
            function_PalmTree_embedding, 1, selected_basic_block_idx_expanded
        ).squeeze(1)  # [B, max_instructions, function_PalmTree_embedding_dim]
        
        # 2. 获取选中基本块的有效掩码
        mask_idx_expanded = selected_basic_block_idx.view(B, 1, 1)
        mask_idx_expanded = mask_idx_expanded.expand(
            -1, -1, max_instructions
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
        z = F.relu(self.project_in(autoregressive_embedding))

        # 6. 结合基本块嵌入和重复比信息作为门控条件
        combined_condition = torch.cat([selected_block_embedding, junk_repeat_ratio], dim=1)

        # 7. 使用结合门控条件，输入自回归嵌入预测垃圾代码类型
        junk_logits = self.glu(z, combined_condition)  # [B x junk_num]

        # 8. 扩展并填充NO_OP位置
        B, _ = junk_logits.shape

        # 获取批量中哪些样本需要垃圾代码类型参数
        need_junk_mask = obf_params[self.param_name]  # [B] 布尔张量

        # 为所有样本创建基础形状
        no_op_logits = torch.zeros((B, 1), device=device)
        other_logits = torch.full((B, self.junk_num), float('-1e9'), device=device)

        # 创建最终的logits张量
        expanded_logits = torch.cat([no_op_logits, other_logits], dim=-1)  # [B, junk_num + 1]

        # 对于需要垃圾代码类型参数的样本，更新logits
        if need_junk_mask.any():
            need_idx = torch.where(need_junk_mask)[0]    # 获取需要垃圾代码类型的样本索引
            expanded_logits[need_idx, 0] = float('-1e9')        # 这些样本的NO_OP位置应该设为极小的负数
            expanded_logits[need_idx, 1:] = junk_logits[need_idx]   # 这些样本的垃圾代码类型使用真实的logits
        # 对于不需要垃圾代码类型参数的样本，NO_OP位置保持0，其他位置保持-1e9
        junk_logits = expanded_logits  # [B, junk_num + 1]
        
        return junk_logits, distributions.Categorical(logits=junk_logits)

    def evaluate_actions(self, 
                         junk_idx, 
                         obf_params, 
                         autoregressive_embedding,   # [B, autoregressive_embedding_dim]
                         function_PalmTree_embedding,# [B, L, max_blocks, max_instructions, function_PalmTree_embedding_dim]无效位置填充0
                         function_PalmTree_mask,     # [B, L, max_blocks, max_instructions], 1表示有效，0表示填充
                         selected_basic_block_idx,   # [B, 1]
                         junk_repeat_ratio           # [B, L, junk_num]重复率
                         ):
        junk_logits, dist = self._get_dist(
                  obf_params, 
                  autoregressive_embedding,   # [B, autoregressive_embedding_dim]
                  function_PalmTree_embedding,# [B, L, max_blocks, max_instructions, function_PalmTree_embedding_dim]无效位置填充0
                  function_PalmTree_mask,     # [B, L, max_blocks, max_instructions], 1表示有效，0表示填充
                  selected_basic_block_idx,   # [B, 1]
                  junk_repeat_ratio           # [B, L, junk_num]重复率
                )

        # 9. 采样垃圾代码类型
        junk_log_prob = dist.log_prob(junk_idx.squeeze(-1)).unsqueeze(-1)
        junk_entropy = dist.entropy().mean()
        
        return junk_log_prob, junk_entropy
        
    def act(self, 
            obf_params, 
            autoregressive_embedding,   # [B, autoregressive_embedding_dim]
            function_PalmTree_embedding,# [B, L, max_blocks, max_instructions, function_PalmTree_embedding_dim]无效位置填充0
            function_PalmTree_mask,     # [B, L, max_blocks, max_instructions], 1表示有效，0表示填充
            selected_basic_block_idx,   # [B, 1]
            junk_repeat_ratio           # [B, L, junk_num]重复率
            ):
        junk_logits, dist = self._get_dist(
                  obf_params, 
                  autoregressive_embedding,   # [B, autoregressive_embedding_dim]
                  function_PalmTree_embedding,# [B, L, max_blocks, max_instructions, function_PalmTree_embedding_dim]无效位置填充0
                  function_PalmTree_mask,     # [B, L, max_blocks, max_instructions], 1表示有效，0表示填充
                  selected_basic_block_idx,   # [B, 1]
                  junk_repeat_ratio           # [B, L, junk_num]重复率
                )

        # 9. 采样垃圾代码类型
        junk_idx = dist.sample().unsqueeze(-1)  # [B x 1]
        junk_log_prob = dist.log_prob(junk_idx.squeeze(-1)).unsqueeze(-1)
        junk_entropy = dist.entropy().mean()
        
        return junk_logits, junk_idx, junk_log_prob, junk_entropy