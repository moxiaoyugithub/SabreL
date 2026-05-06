import numpy as np

class Obfuscation_step_evaluator:
    def __init__(
            self, 
            obf_max_inst_growth, 
            obf_success_similarity, # 这是一个阈值，比如 0.2
            score_attack_settlement=2.0, 
            ratio_growth_score=1.0
            ):
        self.obf_max_inst_growth = obf_max_inst_growth
        self.obf_success_similarity = obf_success_similarity
        self.score_attack_settlement = score_attack_settlement
        self.ratio_growth_score = ratio_growth_score

        # 权重系数
        self.score_reduced_sim = 2.0
        self.score_min_sim = 5.0
    
    def calculate_reward_and_done(self, accumulated_inst_growth, step_inst_growth, step_sim_dict, beyond_sim_dict, min_sim_dict):
        model_names = list(step_sim_dict.keys())
        # 处理step_sim_dict中可能的缺失值
        for m in model_names:
            if step_sim_dict[m] == None:
                step_sim_dict[m] = beyond_sim_dict[m]
        
        # 1. 基础代价保持不变 (预算控制)
        cost_penalty = - (step_inst_growth / self.obf_max_inst_growth)
        
        # 2. 多维进步奖 (取消动态权重，改用静态平均或求和)
        # 这样 Critic 网络能学到一个稳定的预期
        improvement_reward = 0
        model_count = len(step_sim_dict)
        for m in step_sim_dict:
            delta = beyond_sim_dict[m] - step_sim_dict[m]
            # 统一缩放，防止模型越多奖励越大
            improvement_reward += (delta * self.score_reduced_sim) / model_count

        # 3. 核心改进：最弱环节奖励 (Min-Max 优化)
        # 我们只看那个相似度最高的模型有没有降，这才是混淆的真正瓶颈
        worst_sim_now = max(step_sim_dict.values())
        worst_sim_before = max(beyond_sim_dict.values())
        bottleneck_reward = (worst_sim_before - worst_sim_now) * self.score_reduced_sim

        # 4. 破纪录奖 (对每个模型独立的 min_sim 突破给一个小额奖励)
        record_reward = 0
        for m in step_sim_dict:
            if step_sim_dict[m] < min_sim_dict[m]:
                # 只要打破任何一个模型的纪录，就给一个固定的微小正反馈
                record_reward += 0.05 / model_count 

        # 5. 结算逻辑
        done = False
        settlement_reward = 0
        if worst_sim_now <= self.obf_success_similarity:
            done = True
            remaining_ratio = max(0, 1.0 - (accumulated_inst_growth / self.obf_max_inst_growth))
            settlement_reward = self.score_attack_settlement + remaining_ratio * self.ratio_growth_score
        elif accumulated_inst_growth >= self.obf_max_inst_growth:
            done = True

        step_reward = cost_penalty + improvement_reward + bottleneck_reward + record_reward + settlement_reward
        return step_reward, done