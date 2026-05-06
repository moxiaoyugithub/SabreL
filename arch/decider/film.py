import torch
import torch.nn as nn

class TacticalFusion(nn.Module):
    """
    最前沿的 FiLM (Feature-wise Linear Modulation) 融合层
    """
    def __init__(self, state_dim, strategy_dim):
        super().__init__()
        # 生成缩放系数 gamma 和 偏移系数 beta
        self.modulation = nn.Sequential(
            nn.Linear(strategy_dim, state_dim * 2),
            nn.LayerNorm(state_dim * 2)
        )
        self.gate = nn.Sequential(
            nn.Linear(strategy_dim, state_dim),
            nn.Sigmoid()
        )

    def forward(self, x_pt, x_strategy):
        # x_pt: [B*L, hidden_dim]
        # x_strategy: [B*L, strategy_dim]
        
        # 1. 计算调制参数
        mod = self.modulation(x_strategy)
        gamma, beta = torch.chunk(mod, 2, dim=-1)
        
        # 2. 线性调制 (FiLM)
        # 核心逻辑：让 LLM 的战术建议直接改写代码特征的重要程度
        x_modulated = (1 + gamma) * x_pt + beta
        
        # 3. 门控残差连接
        # 防止 LLM 给出错误战术时彻底破坏原始特征
        g = self.gate(x_strategy)
        return x_pt + g * torch.relu(x_modulated)