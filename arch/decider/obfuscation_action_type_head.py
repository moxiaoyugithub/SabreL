import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

class ObfuscationActionTypeHead(nn.Module):
    """
    混淆动作类型头
    """
    def __init__(self, autoregressive_embedding_dim, hidden_dim, action_type_num, dropout=0.1):
        super().__init__()
        self.action_type_num = action_type_num
        self.hidden_dim = hidden_dim
        
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_type_num)
        )
        
        self.project = nn.Linear(action_type_num, autoregressive_embedding_dim)

        # 0: {"selected_basic_block": True, "selected_instruction": True, "predicate": False, "junk":False},                 # 基本块拆分
        # 1: {"selected_basic_block": True, "selected_instruction": False, "predicate": True, "junk":False},                 # 不透明谓词
        # 2: {"selected_basic_block": True, "selected_instruction": True, "predicate": False, "junk": True},                 # 垃圾代码
        
        # 混淆类型与所需参数的映射关系
        # self.register_buffer('selected_basic_block_map', torch.tensor([True, True, True], dtype=torch.bool))
        self.register_buffer('selected_instruction_map', torch.tensor([True, False, True], dtype=torch.bool))
        self.register_buffer('predicate_map', torch.tensor([False, True, False], dtype=torch.bool))
        self.register_buffer('junk_map', torch.tensor([False, False, True], dtype=torch.bool))
    
    def forward(self, 
                lstm_output, 
                available_actions_mask  # [B, action_type_num]，1表示可用，0表示不可用
                ):
        return self.act(lstm_output, 
                        available_actions_mask  # [B, action_type_num]，1表示可用，0表示不可用
                        )
    
    def _get_dist(self, lstm_output, available_actions_mask):
        device = lstm_output[0].device
        
        # 1. 计算混淆类型
        action_type_logits = self.output_fc(lstm_output)
        
        # 2. 根据可用动作掩码调整logits
        # 注意：此处使用 .masked_fill 会比加法更稳定，效果一样
        available_actions_mask = available_actions_mask[:, -1, :]
        action_type_logits = action_type_logits.masked_fill(~available_actions_mask.bool(), float('-1e9'))
        
        # 返回 logits 用于计算自回归嵌入，返回 dist 用于采样/算概率
        return action_type_logits, distributions.Categorical(logits=action_type_logits)
    
    def evaluate_actions(
                self, 
                action_type, 
                lstm_output, 
                available_actions_mask  # [B, action_type_num]，1表示可用，0表示不可用
                ):
        action_type_logits, dist = self._get_dist(lstm_output, 
                                                  available_actions_mask)
        
        # 3. 采样混淆类型
        action_type_log_prob = dist.log_prob(action_type.squeeze(-1)).unsqueeze(-1)
        action_type_entropy = dist.entropy().mean()
        action_type = dist.sample().unsqueeze(-1)
        action_type_one_hot = F.one_hot(action_type.squeeze(-1), self.action_type_num).float()

        # 4. 计算自回归嵌入
        autoregressive_embedding = self.project(action_type_logits)

        # 5. 计算混淆所需参数
        # 使用索引从映射张量中获取批量参数
        # selected_basic_block_mask = self.selected_basic_block_map[action_type.squeeze(-1)]  # [B]
        selected_instruction_mask = self.selected_instruction_map[action_type.squeeze(-1)]  # [B]
        predicate_mask = self.predicate_map[action_type.squeeze(-1)]  # [B]
        junk_mask = self.junk_map[action_type.squeeze(-1)]  # [B]
        obf_params = {
            # "selected_basic_block": selected_basic_block_mask,
            "selected_instruction": selected_instruction_mask,
            "predicate": predicate_mask,
            "junk": junk_mask
        }
        
        return action_type_log_prob, action_type_entropy, action_type_one_hot, obf_params, autoregressive_embedding
    
    def act(self, 
            lstm_output, 
            available_actions_mask  # [B, action_type_num]，1表示可用，0表示不可用
            ):
        action_type_logits, dist = self._get_dist(lstm_output, 
                                                  available_actions_mask)
        
        # 3. 采样混淆类型
        action_type = dist.sample().unsqueeze(-1)
        action_type_one_hot = F.one_hot(action_type.squeeze(-1), self.action_type_num).float()
        action_type_log_prob = dist.log_prob(action_type.squeeze(-1)).unsqueeze(-1)
        action_type_entropy = dist.entropy().mean()

        # 4. 计算自回归嵌入
        autoregressive_embedding = self.project(action_type_logits)

        # 5. 计算混淆所需参数
        # 使用索引从映射张量中获取批量参数
        # selected_basic_block_mask = self.selected_basic_block_map[action_type.squeeze(-1)]  # [B]
        selected_instruction_mask = self.selected_instruction_map[action_type.squeeze(-1)]  # [B]
        predicate_mask = self.predicate_map[action_type.squeeze(-1)]  # [B]
        junk_mask = self.junk_map[action_type.squeeze(-1)]  # [B]
        obf_params = {
            # "selected_basic_block": selected_basic_block_mask,
            "selected_instruction": selected_instruction_mask,
            "predicate": predicate_mask,
            "junk": junk_mask
        }
        
        return action_type_logits, action_type, action_type_log_prob, action_type_entropy, action_type_one_hot, obf_params, autoregressive_embedding