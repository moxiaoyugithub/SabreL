import torch.nn as nn

class Conv1dResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels)
        )
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        # x shape: [Batch, Channels, 1]
        residual = x
        out = self.conv(x)
        out += residual
        return self.activation(out)

class Base(nn.Module):
    def __init__(self, hidden_dim, res_block_num=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 将 [Batch, hidden_dim] 转换为 [Batch, hidden_dim, 1] 进行卷积
        res_blocks = []
        for _ in range(res_block_num):
            res_blocks.append(Conv1dResidualBlock(self.hidden_dim))
        
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # 最后的回归头
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim // 2, 1)
        )

    def forward(self, lstm_output):
        # 转换维度以适应 Conv1d
        # lstm_output: [Total_Batch, hidden_dim] -> [Total_Batch, hidden_dim, 1]
        x = lstm_output.unsqueeze(-1)
        x = self.res_blocks(x)
        value = self.value_head(x)
        return value