import torch

# 查看可用 GPU 数量
device_count = torch.cuda.device_count()
print(f"检测到 GPU 数量: {device_count}")

for i in range(device_count):
    # 获取显卡名称
    device_name = torch.cuda.get_device_name(i)
    # 获取显存总量 (单位转换为 GB)
    total_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"编号 {i}: {device_name} | 总显存: {total_memory:.2f} GB")