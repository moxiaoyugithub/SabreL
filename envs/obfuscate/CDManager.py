import numpy as np

class ObfuscationActionCDManager:
    def __init__(self, action_type_num, cd_config=None):
        """
        cd_config: 字典，定义每个 action_type 的冷却回合数
        例如: {0: 1, 1: 5, 2: 1} -> 0(Split)冷却1步, 1(Opaque)冷却5步, 2(Junk)冷却1步
        """
        self.action_type_num = action_type_num
        # 默认配置：如果未提供，默认所有动作 CD 为 0 (即可连续使用)
        self.cd_config = cd_config if cd_config else {i: 0 for i in range(action_type_num)}
        
        # 记录每个动作最后一次被使用的 step 索引
        # 初始化为很小的值，确保第一步所有动作都可用
        self.last_used_step = {i: -999 for i in range(action_type_num)}

    def update_usage(self, action_type, step_i):
        """当 Agent 执行了某个动作后，更新其最后使用记录"""
        self.last_used_step[action_type] = step_i

    def get_available_mask(self, step_i):
        """
        根据当前步数计算动作掩码
        1 表示可用，0 表示处于 CD 中
        """
        mask = np.ones(self.action_type_num, dtype=np.int8)
        for act in range(self.action_type_num):
            wait_time = step_i - self.last_used_step[act]
            if wait_time < self.cd_config.get(act, 0):
                mask[act] = 0
        return mask