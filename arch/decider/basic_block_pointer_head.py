import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

class BasicBlockPointerHead(nn.Module):
    """
    基本块位置选择头
    """
    def __init__(self, autoregressive_embedding_dim, function_PalmTree_embedding_dim, action_type_num, 
                 max_blocks=128, hidden_dim=256, temperature=1.0):
        super().__init__()
        self.param_name = "selected_basic_block"

        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.max_blocks = max_blocks
        
        # 自回归嵌入投影
        self.project_in = nn.Linear(autoregressive_embedding_dim, hidden_dim)

        # 动作类型嵌入
        self.action_embed = nn.Linear(action_type_num, hidden_dim)

        # 基本块处理
        self.blocks_proj = nn.Linear(function_PalmTree_embedding_dim, hidden_dim)
        
        # 多头注意力
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # 自回归嵌入更新投影
        self.project_out = nn.Linear(max_blocks, autoregressive_embedding_dim)
    
    def forward(self, 
                # obf_params, 
                autoregressive_embedding,   # [B, autoregressive_embedding_dim]
                function_PalmTree_embedding,# [B, L, max_blocks, max_instructions, function_PalmTree_embedding_dim]无效位置填充0
                function_PalmTree_mask,     # [B, L, max_blocks, max_instructions], 1表示有效，0表示填充
                action_type_one_hot         # [B, action_type_num]
                ):
        return self.act(# obf_params, 
                        autoregressive_embedding,   # [B, autoregressive_embedding_dim]
                        function_PalmTree_embedding,# [B, L, max_blocks, max_instructions, function_PalmTree_embedding_dim]无效位置填充0
                        function_PalmTree_mask,     # [B, L, max_blocks, max_instructions], 1表示有效，0表示填充
                        action_type_one_hot         # [B, action_type_num]
                        )
    
    def _get_dist(self, 
                  # obf_params, 
                  autoregressive_embedding,   # [B, autoregressive_embedding_dim]
                  function_PalmTree_embedding,# [B, L, max_blocks, max_instructions, function_PalmTree_embedding_dim]无效位置填充0
                  function_PalmTree_mask,     # [B, L, max_blocks, max_instructions], 1表示有效，0表示填充
                  action_type_one_hot         # [B, action_type_num]
                ):
        device = autoregressive_embedding.device
        # 1. 对每个block平均池化得到block级别的embedding合集（key）
        # 从帧堆叠的输入中，取出当前帧
        function_PalmTree_embedding = function_PalmTree_embedding[:, -1, :, :, :]
        function_PalmTree_mask = function_PalmTree_mask[:, -1, :, :]
        # 计算每个block的有效指令数
        valid_instruction_counts = function_PalmTree_mask.sum(dim=-1, keepdim=True)  # [B, max_blocks, 1]
        
        # 使用einsum计算加权和
        # 公式: block_sums[b,i,d] = Σ_j mask[b,i,j] * embedding[b,i,j,d]
        block_sums = torch.einsum(
            'bijk,bij->bik', 
            function_PalmTree_embedding, 
            function_PalmTree_mask.float()
        )  # [B, max_blocks, function_PalmTree_embedding_dim]
        
        # 计算平均值
        blocks_embedding = torch.where(
            valid_instruction_counts > 0,
            block_sums / (valid_instruction_counts),
            torch.zeros_like(block_sums)
        )  # [B, max_blocks, function_PalmTree_embedding_dim]

        # 2. 计算标记有效基本块的掩码
        block_masks = (function_PalmTree_mask.sum(dim=-1) > 0).float()  # [B, max_blocks], 1表示有效，0表示无效

        # 3. 投影自回归嵌入
        z = F.relu(self.project_in(autoregressive_embedding))  # [B, hidden_dim]

        # 4. 嵌入动作类型信息
        action_embed = F.relu(self.action_embed(action_type_one_hot))  # [B, hidden_dim]

        # 5. 处理所有基本块的嵌入（key）
        blocks_project = F.relu(self.blocks_proj(blocks_embedding))  # [B, max_blocks, hidden_dim]

        # 6. 计算查询向量
        combined_query = z + action_embed  # [B, hidden_dim]
        blocks_query = combined_query.unsqueeze(1)  # [B, 1, hidden_dim]

        # 7. 创建注意力掩码
        # 注意力掩码的形状需要是 [B, max_blocks]（对于batch_first=True）
        # 我们需要将有效位置设为False（不mask），无效位置设为True（mask）
        attention_mask = ~block_masks.bool()  # [B, max_blocks]，True表示填充位置（需要mask）

        # 8. 多头注意力（使用掩码过滤填充位置）
        attention_output, attention_weights = self.attention(
            blocks_query, 
            blocks_project, 
            blocks_project,
            key_padding_mask=attention_mask  # 使用key_padding_mask来mask无效位置
        )

        selected_basic_block_logits = attention_weights.squeeze(1)  # [B, max_blocks]
        selected_basic_block_logits = torch.log(selected_basic_block_logits + 1e-9) # pytorch的attention内部应用了softmax，需要使用log将其映射回原始预测空间
        
        # 再填充确保无效位置不会有概率
        selected_basic_block_logits = selected_basic_block_logits.masked_fill(
            ~block_masks.bool(), float('-1e9'))
        
        return selected_basic_block_logits, distributions.Categorical(logits=selected_basic_block_logits / self.temperature)

    def evaluate_actions(self, 
                         selected_basic_block_idx, 
                         # obf_params, 
                         autoregressive_embedding,   # [B, autoregressive_embedding_dim]
                         function_PalmTree_embedding,# [B, L, max_blocks, max_instructions, function_PalmTree_embedding_dim]无效位置填充0
                         function_PalmTree_mask,     # [B, L, max_blocks, max_instructions], 1表示有效，0表示填充
                         action_type_one_hot         # [B, action_type_num]
                         ):
        selected_basic_block_logits, dist = self._get_dist(
            # obf_params, 
            autoregressive_embedding,   # [B, autoregressive_embedding_dim]
            function_PalmTree_embedding,# [B, L, max_blocks, max_instructions, function_PalmTree_embedding_dim]无效位置填充0
            function_PalmTree_mask,     # [B, L, max_blocks, max_instructions], 1表示有效，0表示填充
            action_type_one_hot         # [B, action_type_num]
        )
        selected_basic_block_log_prob = dist.log_prob(selected_basic_block_idx.squeeze(-1)).unsqueeze(-1)
        selected_basic_block_entropy = dist.entropy().mean()
        
        # 10. 更新自回归嵌入（加入选择结果预测信息）
        t = self.project_out(selected_basic_block_logits)   # [B, autoregressive_embedding_dim]
        updated_autoregressive_embedding = autoregressive_embedding + t

        return selected_basic_block_log_prob, selected_basic_block_entropy, updated_autoregressive_embedding
        
    def act(self, 
            # obf_params, 
            autoregressive_embedding,   # [B, autoregressive_embedding_dim]
            function_PalmTree_embedding,# [B, L, max_blocks, max_instructions, function_PalmTree_embedding_dim]无效位置填充0
            function_PalmTree_mask,     # [B, L, max_blocks, max_instructions], 1表示有效，0表示填充
            action_type_one_hot         # [B, action_type_num]
            ):
        selected_basic_block_logits, dist = self._get_dist(
            # obf_params, 
            autoregressive_embedding,   # [B, autoregressive_embedding_dim]
            function_PalmTree_embedding,# [B, L, max_blocks, max_instructions, function_PalmTree_embedding_dim]无效位置填充0
            function_PalmTree_mask,     # [B, L, max_blocks, max_instructions], 1表示有效，0表示填充
            action_type_one_hot         # [B, action_type_num]
        )
        
        # logits填充到模型预设的 max_blocks
        # pad_len = self.max_blocks - selected_basic_block_logits.size(-1)
        
        # if pad_len > 0:
        #     # F.pad 的参数是从最后一维开始的 (左填充, 右填充)
        #     # 我们在右侧填充极小值，确保这些位置在 softmax 后概率为 0
        #     selected_basic_block_logits = F.pad(selected_basic_block_logits, (0, pad_len), value=float('-1e9'))

        # 9. 采样获取选中的基本块idx
        selected_basic_block_idx = dist.sample().unsqueeze(-1)
        selected_basic_block_log_prob = dist.log_prob(selected_basic_block_idx.squeeze(-1)).unsqueeze(-1)
        selected_basic_block_entropy = dist.entropy().mean()

        # 10. 更新自回归嵌入（加入选择结果预测信息）
        t = self.project_out(selected_basic_block_logits)   # [B, autoregressive_embedding_dim]
        updated_autoregressive_embedding = autoregressive_embedding + t

        return selected_basic_block_logits, selected_basic_block_idx, selected_basic_block_log_prob, selected_basic_block_entropy, updated_autoregressive_embedding