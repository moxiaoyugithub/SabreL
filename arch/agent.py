import torch
import torch.nn as nn

from arch.center.core_net import Core

from arch.decider.obfuscation_action_type_head import ObfuscationActionTypeHead
from arch.decider.basic_block_pointer_head import BasicBlockPointerHead
from arch.decider.instruction_pointer_head import InstructionPointerHead
from arch.decider.opaque_predicate_head import OpaquePredicateHead
from arch.decider.junk_code_head import JunkCodeHead

from arch.critic.value_net import Base

from logs.logger import logger

class FunctionObfuscationAgent_AI(nn.Module):
    """
    二进制函数混淆智能体（对AI分析器）
    """
    def __init__(self, 
                 autoregressive_embedding_dim=128, 
                 function_PalmTree_embedding_dim=128, 
                 function_LLM_embedding_dim=256, 
                 hidden_dim=256, 
                 max_blocks=128,           # 最大基本块数量
                 max_instructions=128,     # 最大指令数量
                 predicate_num=8,
                 junk_num=12, 
                 debug=False
                 ):
        super().__init__()
        
        self.action_type_num = 3  # 三种混淆动作类型：0-基本块拆分，1-不透明谓词，2-垃圾代码

        self.predicate_num = predicate_num
        self.junk_num = junk_num
        self.max_blocks = max_blocks
        self.max_instructions = max_instructions
        
        self.debug = debug
        
        if self.debug:
            logger.info(f'[FunctionObfuscationAgent]: ========== Init ==========', file_only=True)

        # 核心网络
        self.core = Core(function_PalmTree_embedding_dim, function_LLM_embedding_dim, hidden_dim)
        
        # 价值网络
        self.base = Base(hidden_dim)

        # 各个头部
        self.obfuscation_action_type_head = ObfuscationActionTypeHead(autoregressive_embedding_dim, hidden_dim, self.action_type_num)
        self.basic_block_head = BasicBlockPointerHead(autoregressive_embedding_dim, function_PalmTree_embedding_dim, self.action_type_num, max_blocks, hidden_dim)
        self.instruction_head = InstructionPointerHead(autoregressive_embedding_dim, function_PalmTree_embedding_dim, self.action_type_num, max_instructions, hidden_dim)
        self.opaque_predicate_head = OpaquePredicateHead(autoregressive_embedding_dim, function_PalmTree_embedding_dim, predicate_num, hidden_dim)
        self.junk_code_head = JunkCodeHead(autoregressive_embedding_dim, function_PalmTree_embedding_dim, junk_num, hidden_dim)

        self.init_parameters()
        
        self.total_params_num = self.count_parameters()
        
        if self.debug:
            logger.info(f'[FunctionObfuscationAgent]: ==========================', file_only=True)

    def init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def set_temperature(self, temperature):
        self.basic_block_head.temperature = temperature
        self.instruction_head.temperature = temperature
    
    def count_parameters(self):
        """
        统计并记录模型各个子模块的参数量
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(f'[FunctionObfuscationAgent]: ========== Parameter Statistics ==========', file_only=True)
        
        # 遍历子模块统计
        for name, child in self.named_children():
            child_params = sum(p.numel() for p in child.parameters())
            ratio = (child_params / total_params) * 100 if total_params > 0 else 0
            logger.info(f'[FunctionObfuscationAgent]: Module: {name:<25} | Params: {child_params:<12,} | Ratio: {ratio:>6.2f}%', file_only=True)
            
        logger.info(f'[FunctionObfuscationAgent]: ------------------------------------------', file_only=True)
        logger.info(f'[FunctionObfuscationAgent]: Total Parameters: {total_params:,}', file_only=True)
        logger.info(f'[FunctionObfuscationAgent]: Trainable Parameters: {trainable_params:,}', file_only=True)
        logger.info(f'[FunctionObfuscationAgent]: ===========================================', file_only=True)
        
        return total_params
    
    def forward(self, 
                function_PalmTree_embedding, 
                function_PalmTree_mask, 
                function_LLM_embedding=None, 
                recurrent_hidden_states_in=None, 
                recurrent_mask_in=None, 
                junk_repeat_ratio=None, 
                available_actions_mask=None, 
                return_logits=True):
        
        return self.act(
                function_PalmTree_embedding, 
                function_PalmTree_mask, 
                function_LLM_embedding, 
                recurrent_hidden_states_in, 
                recurrent_mask_in, 
                junk_repeat_ratio, 
                available_actions_mask, 
                return_logits)
    
    # state -> |core| -> |base| -> value
    def get_value(self, 
                  function_PalmTree_embedding, 
                  function_PalmTree_mask, 
                  function_LLM_embedding=None, 
                  recurrent_hidden_states_in=None, 
                  recurrent_mask_in=None
                  ):
        lstm_output, recurrent_hidden_states_out = self.core(function_PalmTree_embedding, 
                                                             function_PalmTree_mask, 
                                                             function_LLM_embedding, 
                                                             recurrent_hidden_states_in, 
                                                             recurrent_mask_in)
        values = self.base(lstm_output)
        return values
    
    def evaluate_actions(self, 
                         actions, 
                         function_PalmTree_embedding, 
                         function_PalmTree_mask, 
                         function_LLM_embedding=None, 
                         recurrent_hidden_states_in=None, 
                         recurrent_mask_in=None, 
                         junk_repeat_ratio=None, 
                         available_actions_mask=None
                         ):
        """
        专门用于更新阶段：给定状态和当时采取的动作，重新计算概率和熵
        """
        # 过 Core 网络
        lstm_output, recurrent_hidden_states_out = self.core(
            function_PalmTree_embedding, 
            function_PalmTree_mask, 
            function_LLM_embedding, 
            recurrent_hidden_states_in, 
            recurrent_mask_in
        )
        
        # 计算价值
        values = self.base(lstm_output)

        action_type_log_prob, action_type_entropy, action_type_one_hot, obf_params, autoregressive_embedding = self.obfuscation_action_type_head.evaluate_actions(
            actions['action_type'], 
            lstm_output, 
            available_actions_mask=available_actions_mask
            )
        selected_basic_block_log_prob, selected_basic_block_entropy, updated_autoregressive_embedding = self.basic_block_head.evaluate_actions(
            actions['selected_basic_block'], 
            # obf_params, 
            autoregressive_embedding,           # [B, autoregressive_embedding_dim]
            function_PalmTree_embedding,        # [B, L, max_blocks, max_instructions, function_PalmTree_embedding_dim]无效位置填充0
            function_PalmTree_mask,             # [B, L, max_blocks, max_instructions], 1表示有效，0表示填充
            action_type_one_hot                 # [B, action_type_num]
        )
        selected_instruction_log_prob, selected_instruction_entropy, updated_autoregressive_embedding = self.instruction_head.evaluate_actions(
            actions['selected_instruction'], 
            obf_params, 
            updated_autoregressive_embedding,   # [B, autoregressive_embedding_dim]
            function_PalmTree_embedding,        # [B, L, max_blocks, max_instructions, function_PalmTree_embedding_dim]无效位置填充0
            function_PalmTree_mask,             # [B, L, max_blocks, max_instructions], 1表示有效，0表示填充
            actions['selected_basic_block'],    # [B, 1]
            action_type_one_hot                 # [B, action_type_num]
        )
        predicate_log_prob, predicate_entropy = self.opaque_predicate_head.evaluate_actions(
            actions['predicate'], 
            obf_params, 
            updated_autoregressive_embedding,   # [B, autoregressive_embedding_dim]
            function_PalmTree_embedding,        # [B, L, max_blocks, max_instructions, function_PalmTree_embedding_dim]无效位置填充0
            function_PalmTree_mask,             # [B, L, max_blocks, max_instructions], 1表示有效，0表示填充
            actions['selected_basic_block']     # [B, 1]
        )
        junk_log_prob, junk_entropy = self.junk_code_head.evaluate_actions(
            actions['junk'], 
            obf_params, 
            updated_autoregressive_embedding,   # [B, autoregressive_embedding_dim]
            function_PalmTree_embedding,        # [B, L, max_blocks, max_instructions, function_PalmTree_embedding_dim]无效位置填充0
            function_PalmTree_mask,             # [B, L, max_blocks, max_instructions], 1表示有效，0表示填充
            actions['selected_basic_block'],    # [B, 1]
            junk_repeat_ratio                   # [B, L, junk_num]重复率
        )

        # 4. 汇总
        # 总 Log Prob 是各分支之和 (假设各分支在给定状态下条件独立)
        # 在自回归中，这对应 log P(a1, a2, a3...) = log P(a1) + log P(a2|a1) + ...
        total_log_probs = action_type_log_prob + selected_basic_block_log_prob + selected_instruction_log_prob + predicate_log_prob + junk_log_prob
        
        # 总熵是各分支熵的均值（或和），用于鼓励探索
        entropys = torch.stack([action_type_entropy, selected_basic_block_entropy, selected_instruction_entropy, predicate_entropy, junk_entropy])
        avg_entropy = entropys.mean()
        
        return total_log_probs, avg_entropy, values
    
    def act(self, 
            function_PalmTree_embedding, 
            function_PalmTree_mask, 
            function_LLM_embedding=None, 
            recurrent_hidden_states_in=None, 
            recurrent_mask_in=None, 
            junk_repeat_ratio=None, 
            available_actions_mask=None, 
            return_logits=True):
        action_logits = {}
        actions = {}

        # 过网络顺序：
		# |core|                        |critic|------>value
    	#   |                              ^
        #   v                              |
        # lstm_output, hidden_state        |                          状态信息处理
        #        |           |-------------|
        #======================================================================
        #        v
        # |obfuscation_action_type_head|（不包含NO_OP位）
        #               |                                             分层决策
        #               v
        #   autoregressive_embedding, action_type, obf_params_mask--|
        #               |<---------------|                          |
        #               v                                           |
        # |basic_block_head|<---------------------------------------|（不包含NO_OP位）
        #               |                                           |
        #               v                                           |
        #   autoregressive_embedding（更新）, basic_block            |
        #               |<---------------------|                    |
        #               v                                           |
        # |instruction_head|<---------------------------------------|（包含NO_OP位）
        #               |                                           |
        #               v                                           |
        #         autoregressive_embedding（更新）, instruction      |
        #           |__________________|____________________________|
        #           v                  v                            |
        # |opaque_predicate_head|   |junk_code_head|<---------------|（包含NO_OP位）
        #          |                   |
        #          v                   v
        #   opaque_predicate         junk_code

        # 1. 状态信息处理
        if self.debug:
            logger.info(f'[FunctionObfuscationAgent]: ========== Agent running ==========', file_only=True)
        
        #if self.debug:
        #    logger.info(f'[FunctionObfuscationAgent]: Core input - \n\tfunction_PalmTree_embedding {function_PalmTree_embedding} \n\tfunction_PalmTree_mask {function_PalmTree_mask} \n\tfunction_LLM_embedding {function_LLM_embedding}')
        
        lstm_output, recurrent_hidden_states_out = self.core(
            function_PalmTree_embedding, 
            function_PalmTree_mask, 
            function_LLM_embedding, 
            recurrent_hidden_states_in, 
            recurrent_mask_in
        )
        
        # 计算价值
        values = self.base(lstm_output)
        
        #if self.debug:
        #    logger.info(f'[FunctionObfuscationAgent]: Core output - \n\tlstm_output {lstm_output} \n\thidden_state {hidden_state}')
        
        # 2. 动作类型选择
        #if self.debug:
        #    logger.info(f'[FunctionObfuscationAgent]: Action type head input - \n\tlstm_output {lstm_output} \n\tavailable_actions_mask {available_actions_mask}')
        
        action_type_logits, action_type, action_type_log_prob, action_type_entropy, action_type_one_hot, obf_params, autoregressive_embedding = self.obfuscation_action_type_head(
            lstm_output, 
            available_actions_mask=available_actions_mask
            )
        
        #if self.debug:
        #    logger.info(f'[FunctionObfuscationAgent]: Action type head output - \n\taction_type_logits {action_type_logits} \n\taction_type {action_type} \n\taction_type_one_hot {action_type_one_hot} \n\tobf_params {obf_params} \n\tautoregressive_embedding {autoregressive_embedding}')
        
        action_logits["action_type_logits"] = action_type_logits
        actions["action_type"] = action_type
        
        # 3. 基本块位置选择（所有动作都需要）
        #if self.debug:
        #    logger.info(f'[FunctionObfuscationAgent]: Basic block head input - \n\tautoregressive_embedding {autoregressive_embedding} \n\tfunction_PalmTree_embedding {function_PalmTree_embedding} \n\tfunction_PalmTree_mask {function_PalmTree_mask} \n\taction_type_one_hot {action_type_one_hot}')
        
        selected_basic_block_logits, selected_basic_block_idx, selected_basic_block_log_prob, selected_basic_block_entropy, updated_autoregressive_embedding = self.basic_block_head(
            # obf_params, 
            autoregressive_embedding, 
            function_PalmTree_embedding, 
            function_PalmTree_mask, 
            action_type_one_hot
            )

        #if self.debug:
        #    logger.info(f'[FunctionObfuscationAgent]: Basic block head output - \n\tselected_basic_block_logits {selected_basic_block_logits} \n\tselected_basic_block {selected_basic_block} \n\tautoregressive_embedding {autoregressive_embedding}')
        
        action_logits["selected_basic_block_logits"] = selected_basic_block_logits
        actions["selected_basic_block"] = selected_basic_block_idx
        
        # 4. 指令位置选择
        #if self.debug:
        #    logger.info(f'[FunctionObfuscationAgent]: Instruction head input - \n\tobf_params {obf_params} \n\tautoregressive_embedding {autoregressive_embedding} \n\tfunction_PalmTree_embedding {function_PalmTree_embedding} \n\tfunction_PalmTree_mask {function_PalmTree_mask} \n\tselected_basic_block {selected_basic_block} \n\taction_type_one_hot {action_type_one_hot}')
        
        selected_instruction_logits, selected_instruction_idx, selected_instruction_log_prob, selected_instruction_entropy, updated_autoregressive_embedding = self.instruction_head(
            obf_params, 
            updated_autoregressive_embedding, 
            function_PalmTree_embedding, 
            function_PalmTree_mask, 
            selected_basic_block_idx, 
            action_type_one_hot
        )  # [B x max_instructions]
        
        #if self.debug:
        #    logger.info(f'[FunctionObfuscationAgent]: Instruction head output - \n\tselected_instruction_logits {selected_instruction_logits} \n\tselected_instruction {selected_instruction} \n\tautoregressive_embedding {autoregressive_embedding}')
        
        action_logits["selected_instruction_logits"] = selected_instruction_logits
        actions["selected_instruction"] = selected_instruction_idx

        # 5. 不透明谓词类型
        #if self.debug:
        #    logger.info(f'[FunctionObfuscationAgent]: Opaque predicate head input - \n\tobf_params {obf_params} \n\tautoregressive_embedding {autoregressive_embedding} \n\tfunction_PalmTree_embedding {function_PalmTree_embedding} \n\tfunction_PalmTree_mask {function_PalmTree_mask} \n\tselected_basic_block {selected_basic_block}')
        
        predicate_logits, predicate_idx, predicate_log_prob, predicate_entropy = self.opaque_predicate_head(
            obf_params, 
            updated_autoregressive_embedding, 
            function_PalmTree_embedding, 
            function_PalmTree_mask,
            selected_basic_block_idx
            )  # [B x predicate_num]
        
        #if self.debug:
        #    logger.info(f'[FunctionObfuscationAgent]: Opaque predicate head output - \n\tpredicate_logits {predicate_logits} \n\tpredicate {predicate}')
        
        action_logits["predicate_logits"] = predicate_logits
        actions["predicate"] = predicate_idx
        
        # 6. 垃圾代码类型
        #if self.debug:
        #    logger.info(f'[FunctionObfuscationAgent]: Junk code head input - \n\tobf_params {obf_params} \n\tautoregressive_embedding {autoregressive_embedding} \n\tfunction_PalmTree_embedding {function_PalmTree_embedding} \n\tfunction_PalmTree_mask {function_PalmTree_mask} \n\tselected_basic_block {selected_basic_block} \n\tjunk_repeat_ratio {junk_repeat_ratio}')
        
        junk_logits, junk_idx, junk_log_prob, junk_entropy = self.junk_code_head(
            obf_params, 
            updated_autoregressive_embedding, 
            function_PalmTree_embedding, 
            function_PalmTree_mask,
            selected_basic_block_idx, 
            junk_repeat_ratio
            )  # [B x junk_num]
        
        #if self.debug:
        #    logger.info(f'[FunctionObfuscationAgent]: Opaque predicate head output - \n\tjunk_logits {junk_logits} \n\tjunk {junk}')
        
        action_logits["junk_logits"] = junk_logits
        actions["junk"] = junk_idx
        
        if self.debug:
            logger.info(f'[FunctionObfuscationAgent]: ==========================', file_only=True)
        
        # 总 Log Prob 是各分支之和 (假设各分支在给定状态下条件独立)
        # 在自回归中，这对应 log P(a1, a2, a3...) = log P(a1) + log P(a2|a1) + ...
        total_log_probs = action_type_log_prob + selected_basic_block_log_prob + selected_instruction_log_prob + predicate_log_prob + junk_log_prob
        
        # 总熵是各分支熵的均值（或和），用于鼓励探索
        entropys = torch.stack([action_type_entropy, selected_basic_block_entropy, selected_instruction_entropy, predicate_entropy, junk_entropy])
        avg_entropy = entropys.mean()
        
        if return_logits:
            return recurrent_hidden_states_out, action_logits, actions, total_log_probs, avg_entropy, values
        else:
            return recurrent_hidden_states_out, actions, total_log_probs, avg_entropy, values