from copy import deepcopy

from envs.binary_process_editor import BPE_utils

from envs.obfuscate.opaque_predicate import OpaquePredicateInserter, ShadowOpaquePredicateInserter
from envs.obfuscate.junk_code import JunkCodeInserter, ShadowJunkCodeInserter
from envs.obfuscate.basic_block_splite import BasicBlockSpliter, ShadowBasicBlockSpliter

from logs.logger import logger

# 自定义函数修改器，使用函数修改器修改CFG和inst的基础功能定义的高级修改器
# 在一个回合内对某个函数做连续的修改，在回合结束reset时切换函数
class MixObfuscator:
    def __init__(self, 
                 rewritten_binary_directory, 
                 rewritten_gtirb_directory_w, 
                 rewritten_gtirb_directory_r, 
                 junk_blocks, 
                 rank, 
                 debug, 
                 draw_cfg
                 ):
        self.rank = rank
        self.debug = debug
        self.draw_cfg = draw_cfg
        
        logger.setup_rank(self.rank)
        
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: ========== obfuscator.init ==========', file_only=True)

        self.rewritten_binary_directory = rewritten_binary_directory
        self.rewritten_gtirb_directory_w = rewritten_gtirb_directory_w
        self.rewritten_gtirb_directory_r = rewritten_gtirb_directory_r

        # 定义一组混淆器
        # 不透明谓词插入器
        self.opaque_predicate_inserter = OpaquePredicateInserter()
        # 基本块分割器
        self.basic_block_spliter = BasicBlockSpliter()
        # 垃圾指令插入器
        # self.selected_blocks = selected_blocks      # 垃圾指令块
        self.junk_code_inserter = JunkCodeInserter(junk_blocks)

        # 由于位置选择直接来自指针网络，输出的idx严格与结构对应，不再需要硬编码位置表

        # 动作类型对应表
        self.obfuscate_function_map = {
            0: self.action_splite_basic_block,
            1: self.action_opaque_predicate,
            2: self.action_insert_junk_code
        }
        # ([参数列表], 是否需要step_i)
        self.obfuscate_param_map = {
            0: (['selected_basic_block', 'selected_instruction'], True),
            1: (['selected_basic_block', 'predicate'], True),
            2: (['selected_basic_block', 'selected_instruction', 'junk'], False),
        }

        # 操作记录
        self.operation_records = []
        
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: ==========================', file_only=True)
    
    # 混淆动作执行
    def action_execute(self, action, step_i):
        # 将带有NO_OP位动作还原到真实位置
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: ========== obfuscator.action_execute ==========', file_only=True)
        
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: Got raw action - {action}', file_only=True)
        
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: Remapping action...', file_only=True)
        
        action_remapped = deepcopy(action)
        
        action_type = action_remapped['action_type']
        if action['selected_instruction']:
            action_remapped['selected_instruction'] -= 1
            
        if action['predicate']:
            action_remapped['predicate'] -= 1
            
        if action['junk']:
            action_remapped['junk'] -= 1
        
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: Remapped action - {action_remapped}', file_only=True)
        
        # 解析混淆
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: Parsing obfuscate...', file_only=True)
        
        # 解析混淆函数
        obfuscate = self.obfuscate_function_map[action_type]
        # 解析参数名
        obfuscate_param_name, use_step_i = self.obfuscate_param_map[action_type]
        # 解析参数
        obfuscate_param = []
        for p in obfuscate_param_name:
            obfuscate_param.append(action_remapped[p])
        if use_step_i:
            obfuscate_param += [step_i]
            
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: Parsed obfuscate - Using obfuscate {obfuscate} with parameters {obfuscate_param}', file_only=True)
        
        inst_growth, vcp_growth = obfuscate(*obfuscate_param)

        # 重写并重加载        
        self.dump_and_reload_funtion(step_i)
        
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: ==========================', file_only=True)
        
        return inst_growth, vcp_growth, self.function, action_type

    # 基本块拆分
    def action_splite_basic_block(self, action_component_basic_block_index, action_component_instruction_index, step_i):
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: ========== obfuscator.action_splite_basic_block ==========', file_only=True)
            logger.info(f'[MixObfuscator Rank {self.rank}]: Splite basic block {action_component_basic_block_index} at instruction {action_component_instruction_index}', file_only=True)

        # 执行混淆
        inst_growth, vcp_growth = self.basic_block_spliter.basic_block_splitting(self.function, action_component_basic_block_index, action_component_instruction_index, self.decoder, step_i)
        
        if self.draw_cfg:
            self.function.CFG(self.decoder).draw('logs/log_files/CFG_'+ str(self.rank) + '_block_splite' + '.png', self.decoder)
        
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: ==========================', file_only=True)

        # 操作记录
        self.operation_records.append('Splite basic block {} at instruction {}'.format(action_component_basic_block_index, action_component_instruction_index))
        return inst_growth, vcp_growth
    
    # 不透明谓词插入
    def action_opaque_predicate(self, action_component_basic_block_idx, action_component_selected_opaque_predicate, step_i):
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: ========== obfuscator.action_opaque_predicate ==========', file_only=True)
            logger.info(f'[MixObfuscator Rank {self.rank}]: Insert opaque predicate {action_component_selected_opaque_predicate} at basic block {action_component_basic_block_idx}', file_only=True)
        
        # 解析动作
        obf_opaque_predicate = self.opaque_predicate_inserter.opaque_predicate_function_dict[action_component_selected_opaque_predicate]

        # 执行混淆
        inst_growth, vcp_growth = obf_opaque_predicate(self.function, action_component_basic_block_idx, self.decoder, step_i)
        
        if self.draw_cfg:
            self.function.CFG(self.decoder).draw('logs/log_files/CFG_'+ str(self.rank) + '_opaque_predicate' + '.png', self.decoder)
        
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: ==========================', file_only=True)

        # 操作记录
        self.operation_records.append(('Insert opaque predicate {} at basic block {}'.format(action_component_selected_opaque_predicate, action_component_basic_block_idx)))
        return inst_growth, vcp_growth

    # 垃圾指令插入
    def action_insert_junk_code(self, action_component_basic_block_idx, action_component_instruction_idx, action_component_selected_junk_code):
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: ========== obfuscator.action_insert_junk_code ==========', file_only=True)
            logger.info(f'[MixObfuscator Rank {self.rank}]: Insert junk code {action_component_selected_junk_code} at basic block {action_component_basic_block_idx} instruction {action_component_instruction_idx}', file_only=True)
        # 执行混淆
        inst_growth, vcp_growth = self.junk_code_inserter.insert_junk_code(self.function, action_component_basic_block_idx, action_component_instruction_idx, action_component_selected_junk_code, self.decoder)

        if self.draw_cfg:
            self.function.CFG(self.decoder).draw('logs/log_files/CFG_'+ str(self.rank) + '_junk_code' + '.png', self.decoder)

        # 操作记录
        self.operation_records.append('Insert junk code {} at basic block {} instruction {}'.format(action_component_selected_junk_code, action_component_basic_block_idx, action_component_instruction_idx))
        return inst_growth, vcp_growth
    
    # 每一个step，向临时文件夹写后重新读被修改的二进制函数，以更新一些标签状态
    def dump_and_reload_funtion(self, step_i):
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: ========== obfuscator.dump_and_reload_funtion ==========', file_only=True)
            logger.info(f'[MixObfuscator Rank {self.rank}]: Dumping function address {self.function_address} to {self.rewritten_binary_directory}{self.binary_name}_{str(self.rank)}...', file_only=True)

        # 重写到临时文件夹
        BPE_utils.binary_rewrite(self.rewritten_binary_directory, self.rewritten_gtirb_directory_w, self.function.cfr, self.binary_name + '_' + str(self.rank))
        
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: Dumped.', file_only=True)

        # 重新读取
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: Reloading function address {self.function_address} from {self.rewritten_binary_directory}{self.binary_name}_{str(self.rank)}...', file_only=True)
        
        cfr = BPE_utils.binary_read(self.rewritten_binary_directory, self.rewritten_gtirb_directory_r, self.binary_name + '_' + str(self.rank))
        
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: Reloaded.', file_only=True)
        
        # 重新使用地址定位函数
        try:
            # 优先使用名称查找
            if self.debug:
                logger.info(f'[MixObfuscator Rank {self.rank}]: Finding function by name {self.function_name}...', file_only=True)
            self.function = cfr.find_function_by_name(self.function_name, step_i)
        except Exception:
            # 地址查找失败，尝试通过地址查找
            try:
                if self.debug:
                    logger.info(f'[MixObfuscator Rank {self.rank}]: Finding function by address {self.function_address}...', file_only=True)
                self.function = cfr.find_function_by_address(self.function_address)
            except Exception:
                # 地址也找不到，此时才抛出异常
                raise RuntimeError(f"Cannot find function: {self.function_name} at {hex(self.function_address)}")
        
        # 更新函数定位信息
        self.function_address = self.function.get_entry_adress()
        self.function_name = self.function.name
        
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: Found function {self.function_address}.', file_only=True)
            logger.info(f'[MixObfuscator Rank {self.rank}]: ==========================', file_only=True)
    
    # 在每次reset时切换被混淆函数
    def reset(self, function, function_address, function_name, binary_name):
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: ========== obfuscator.reset ==========', file_only=True)
        
        self.function = function
        self.function_address = function_address
        self.function_name = function_name
        self.decoder = function.cfr.get_decoder()
        
        self.binary_name = binary_name
        
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: Reseted function {self.function_address} name {self.function_name}.', file_only=True)

        if self.draw_cfg:
            self.function.CFG(self.decoder).draw('logs/log_files/CFG_'+ str(self.rank) + '_orignal' + '.png', self.decoder)
        
    def show_operation_record(self):
        print('Operation record:')
        for step, operation_record in enumerate(self.operation_records):
            print(f'Step {step}: {operation_record}')

# ============================================ 影子模式 ===================================================
class ShadowMixObfuscator:
    def __init__(self, 
                 junk_blocks, 
                 rank, 
                 debug, 
                 draw_cfg
                 ):
        self.rank = rank
        self.debug = debug
        self.draw_cfg = draw_cfg
        
        logger.setup_rank(self.rank)
        
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: ========== obfuscator.init ==========', file_only=True)

        # 定义一组混淆器
        # 不透明谓词插入器
        self.opaque_predicate_inserter = ShadowOpaquePredicateInserter()
        # 基本块分割器
        self.basic_block_spliter = ShadowBasicBlockSpliter()
        # 垃圾指令插入器
        # self.selected_blocks = selected_blocks      # 垃圾指令块
        self.junk_code_inserter = ShadowJunkCodeInserter(junk_blocks)

        # 由于位置选择直接来自指针网络，输出的idx严格与结构对应，不再需要硬编码位置表

        # 动作类型对应表
        self.obfuscate_function_map = {
            0: self.action_splite_basic_block,
            1: self.action_opaque_predicate,
            2: self.action_insert_junk_code
        }
        # ([参数列表], 是否需要step_i)
        self.obfuscate_param_map = {
            0: ['selected_basic_block', 'selected_instruction'],
            1: ['selected_basic_block', 'predicate'],
            2: ['selected_basic_block', 'selected_instruction', 'junk'],
        }

        # 操作记录
        self.operation_records = []
        
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: ==========================', file_only=True)
    
    # 混淆动作执行
    def action_execute(self, action):
        # 将带有NO_OP位动作还原到真实位置
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: ========== obfuscator.action_execute ==========', file_only=True)
        
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: Got raw action - {action}', file_only=True)
        
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: Remapping action...', file_only=True)
        
        action_remapped = deepcopy(action)
        
        action_type = action_remapped['action_type']
        if action['selected_instruction']:
            action_remapped['selected_instruction'] -= 1
            
        if action['predicate']:
            action_remapped['predicate'] -= 1
            
        if action['junk']:
            action_remapped['junk'] -= 1
        
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: Remapped action - {action_remapped}', file_only=True)
        
        # 解析混淆
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: Parsing obfuscate...', file_only=True)
        
        # 解析混淆函数
        obfuscate = self.obfuscate_function_map[action_type]
        # 解析参数名
        obfuscate_param_name = self.obfuscate_param_map[action_type]
        # 解析参数
        obfuscate_param = []
        for p in obfuscate_param_name:
            obfuscate_param.append(action_remapped[p])
            
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: Parsed obfuscate - Using obfuscate {obfuscate} with parameters {obfuscate_param}', file_only=True)
        
        if self.draw_cfg:
            self.function.CFG().draw('logs/log_files/CFG_'+ str(self.rank) + '_orignal' + '.png')
        
        inst_growth, vcp_growth = obfuscate(*obfuscate_param)
        
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: ==========================', file_only=True)
        
        return inst_growth, vcp_growth, self.function, action_type

    # 基本块拆分
    def action_splite_basic_block(self, action_component_basic_block_index, action_component_instruction_index):
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: ========== obfuscator.action_splite_basic_block ==========', file_only=True)
            logger.info(f'[MixObfuscator Rank {self.rank}]: Splite basic block {action_component_basic_block_index} at instruction {action_component_instruction_index}', file_only=True)

        # 执行混淆
        inst_growth, vcp_growth = self.basic_block_spliter.basic_block_splitting(self.function, action_component_basic_block_index, action_component_instruction_index)
        
        if self.draw_cfg:
            self.function.CFG().draw('logs/log_files/CFG_'+ str(self.rank) + '_block_splite' + '.png')
        
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: ==========================', file_only=True)

        # 操作记录
        self.operation_records.append('Splite basic block {} at instruction {}'.format(action_component_basic_block_index, action_component_instruction_index))
        return inst_growth, vcp_growth
        
    # 不透明谓词插入
    def action_opaque_predicate(self, action_component_basic_block_idx, action_component_selected_opaque_predicate):
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: ========== obfuscator.action_opaque_predicate ==========', file_only=True)
            logger.info(f'[MixObfuscator Rank {self.rank}]: Insert opaque predicate {action_component_selected_opaque_predicate} at basic block {action_component_basic_block_idx}', file_only=True)
        
        # 解析动作
        obf_opaque_predicate = self.opaque_predicate_inserter.opaque_predicate_function_dict[action_component_selected_opaque_predicate]

        # 执行混淆
        inst_growth, vcp_growth = obf_opaque_predicate(self.function, action_component_basic_block_idx)
        
        if self.draw_cfg:
            self.function.CFG().draw('logs/log_files/CFG_'+ str(self.rank) + '_opaque_predicate' + '.png')
        
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: ==========================', file_only=True)

        # 操作记录
        self.operation_records.append(('Insert opaque predicate {} at basic block {}'.format(action_component_selected_opaque_predicate, action_component_basic_block_idx)))
        return inst_growth, vcp_growth
    
    # 垃圾指令插入
    def action_insert_junk_code(self, action_component_basic_block_idx, action_component_instruction_idx, action_component_selected_junk_code):
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: ========== obfuscator.action_insert_junk_code ==========', file_only=True)
            logger.info(f'[MixObfuscator Rank {self.rank}]: Insert junk code {action_component_selected_junk_code} at basic block {action_component_basic_block_idx} instruction {action_component_instruction_idx}', file_only=True)
        # 执行混淆
        inst_growth, vcp_growth = self.junk_code_inserter.insert_junk_code(self.function, action_component_basic_block_idx, action_component_instruction_idx, action_component_selected_junk_code)

        if self.draw_cfg:
            self.function.CFG().draw('logs/log_files/CFG_'+ str(self.rank) + '_junk_code' + '.png')

        # 操作记录
        self.operation_records.append('Insert junk code {} at basic block {} instruction {}'.format(action_component_selected_junk_code, action_component_basic_block_idx, action_component_instruction_idx))
        return inst_growth, vcp_growth

    # 在每次reset时切换被混淆函数
    def reset(self, function, function_address, function_name, binary_name):
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: ========== obfuscator.reset ==========', file_only=True)
        
        self.function = function
        self.function_address = function_address
        self.function_name = function_name
        
        self.binary_name = binary_name
        
        if self.debug:
            logger.info(f'[MixObfuscator Rank {self.rank}]: Reseted function {self.function_address} name {self.function_name}.', file_only=True)
    
    def show_operation_record(self):
        print('Operation record:')
        for step, operation_record in enumerate(self.operation_records):
            print(f'Step {step}: {operation_record}')