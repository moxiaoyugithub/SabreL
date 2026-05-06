from envs.binary_process_editor import BPE_CFR
from envs.binary_process_editor import BPE_utils

# 单条不透明谓词头：由若干basic_block构成
class OpaquePredicate:
    def __init__(self):
        self.basic_block_i_groups = []
    
    # 该混淆的总指令计数
    def instruction_count(self):
        instruction_count = 0
        for basic_block_i_group in self.basic_block_i_groups:
            for basic_block_name in basic_block_i_group.keys():
                bb_instructions_str = basic_block_i_group[basic_block_name]
                instruction_count += len(bb_instructions_str)
        return instruction_count
    
    def to_i_group(self, selected_basic_block_symbol, i):      # 插入次数i
        opaque_predicate_basic_block_i_groups_list = []
        for basic_block_i_group in self.basic_block_i_groups:
            opaque_predicate_i_basic_block_list = []
            for basic_block_name in basic_block_i_group.keys():
                # 获得basic_block中的指令
                bb_instructions_str = basic_block_i_group[basic_block_name]
                iis_list = []
                # 逐条包装
                for bb_instruction_str in bb_instructions_str:
                    if bb_instruction_str[0] == 'j':
                        if 'SELECT' in bb_instruction_str:
                            bb_instruction_str = 'jmp ' + selected_basic_block_symbol
                        else:
                            bb_instruction_str = bb_instruction_str + '_' + str(i)   # 跳转指令加上插入次数标记，保证唯一性
                    iis_list.append(BPE_CFR.IInstruction(bb_instruction_str))
                opaque_predicate_i_basic_block_list.append(BPE_CFR.IBasicBlock(iis_list, symbol='.' + basic_block_name + '_' + str(i)))        # 基础symbol不保证唯一性，需要加上插入次数标记
            ig = BPE_CFR.IGroup(opaque_predicate_i_basic_block_list)
            opaque_predicate_basic_block_i_groups_list.append(ig)
        return opaque_predicate_basic_block_i_groups_list

class OpaquePredicateInserter:
    # 不透明谓词组
    opaque_predicate_64_0 = OpaquePredicate()
    opaque_predicate_64_0.basic_block_i_groups = [
        {
        'entry': [
                    # 保存容器状态
                    'push rax',
                    'pushf',
                    'push rbx',
                    'push rcx',
                    # 溢出保护
                    'and rax, 0xF',
                    'and rbx, 0xF',
                    'and rcx, 0xF',
                    # 计算2*x*y
                    'mov rcx, rax',
                    'imul rcx, rbx',
                    'add rcx, rcx',
                    # 计算x^2 + y^2
                    'imul rax, rax',
                    'imul rbx, rbx',
                    'add rax, rbx',
                    # 进行比较
                    'cmp rax, rcx',
                    'jge .true'
                    ],
        'false': [
                    'leave'
                    ],
        'true': [
                    'pop rcx',
                    'pop rbx',
                    'popf',
                    'pop rax',
                    'jmp SELECT'
                    ]
        }
    ]
    
    opaque_predicate_64_1 = OpaquePredicate()
    opaque_predicate_64_1.basic_block_i_groups = [
        {
        'entry': [
                    # 保存容器状态
                    'push rax',
                    'pushf',
                    'push rbx',
                    # 溢出保护
                    'and rax, 0xF',
                    'and rbx, 0xF',
                    # 计算x * (x-1) % 2
                    'mov rbx, rax',
                    'sub rbx, 1',
                    'imul rax, rbx',
                    # 进行比较
                    'and rax, 1',
                    'cmp rax, 0',
                    'jne .true'
                    ],
        'false': [
                    'pop rbx',
                    'popf',
                    'pop rax',
                    'jmp SELECT'
                    ],
        },
        {
        'true': [
                    'leave'
                    ]
        }
    ]
    
    opaque_predicate_64_2 = OpaquePredicate()
    opaque_predicate_64_2.basic_block_i_groups = [
        {
        'entry': [
                    # 保存容器状态
                    'push rax',
                    'pushf',
                    'push rbx',
                    'push rcx',
                    # 溢出保护
                    'and rax, 0xF',
                    'and rbx, 0xF',
                    # 计算(y + 1) % 2
                    'mov rcx, rbx',
                    'add rcx, 1',
                    # 进行比较
                    'and rcx, 1',
                    'jz .true'
                    ],
        'transit': [
                     # 计算(2*x + 1) * y
                    'add rax, rax',
                    'add rax, 1',
                    'imul rax, rbx',
                    # 进行比较
                    'test rax, 1',
                    'jz .true'
                    ],
        'false': [
                    'leave'
                    ],
        'true': [
                    'pop rcx',
                    'pop rbx',
                    'popf',
                    'pop rax',
                    'jmp SELECT'
                    ]
        }
    ]
    
    opaque_predicate_64_3 = OpaquePredicate()
    opaque_predicate_64_3.basic_block_i_groups = [
        {
        'entry': [
                    # 保存容器状态
                    'push rax',
                    'pushf',
                    'push rbx',
                    'push rcx',
                    # 溢出保护
                    'and rax, 0xF',
                    'and rbx, 0xF',
                    'and rcx, 0xF',
                    # 计算y^2 - 1
                    'imul rax, rax',
                    'imul rbx, rbx',
                    'mov rcx, rax',
                    # 进行比较
                    'cmp rcx, 1',
                    'jl .transit'
                    ],
        },
        {
        'true': [
                    'pop rcx',
                    'pop rbx',
                    'popf',
                    'pop rax',
                    'jmp SELECT'
                    ]
        },
        {
        'transit': [
                    # 计算(x * y)^2
                    'mov rcx, rbx',
                    'imul rcx, rax',
                    # 计算x^2 + 1
                    'add rbx, 1',
                    # 进行比较
                    'cmp rcx, rbx',
                    'jnge .true'
                    ],
        'false': [
                    'leave'
                    ]
        }
    ]

    opaque_predicate_32_0 = OpaquePredicate()
    opaque_predicate_32_0.basic_block_i_groups = [
        {
        'entry': [
                    # 保存容器状态
                    'push eax',
                    'pushf',
                    'push ebx',
                    'push ecx',
                    # 溢出保护
                    'and eax, 0xF',
                    'and ebx, 0xF',
                    'and ecx, 0xF',
                    # 计算2*x*y
                    'mov ecx, eax',
                    'imul ecx, ebx',
                    'add ecx, ecx',
                    # 计算x^2 + y^2
                    'imul eax, eax',
                    'imul ebx, ebx',
                    'add eax, ebx',
                    # 进行比较
                    'cmp eax, ecx',
                    'jge .true'
                    ],
        'false': [
                    'leave'
                    ],
        'true': [
                    'pop ecx',
                    'pop ebx',
                    'popf',
                    'pop eax',
                    'jmp SELECT'
                    ]
        }
    ]
    
    opaque_predicate_32_1 = OpaquePredicate()
    opaque_predicate_32_1.basic_block_i_groups = [
        {
        'entry': [
                    # 保存容器状态
                    'push eax',
                    'pushf',
                    'push ebx',
                    # 溢出保护
                    'and eax, 0xF',
                    'and ebx, 0xF',
                    # 计算x * (x-1) % 2
                    'mov ebx, eax',
                    'sub ebx, 1',
                    'imul eax, ebx',
                    # 进行比较
                    'and eax, 1',
                    'cmp eax, 0',
                    'jne .true'
                    ],
        'false': [
                    'pop ebx',
                    'popf',
                    'pop eax',
                    'jmp SELECT'
                    ],
        },
        {
        'true': [
                    'leave'
                    ]
        }
    ]
    
    opaque_predicate_32_2 = OpaquePredicate()
    opaque_predicate_32_2.basic_block_i_groups = [
        {
        'entry': [
                    # 保存容器状态
                    'push eax',
                    'pushf',
                    'push ebx',
                    'push ecx',
                    # 溢出保护
                    'and eax, 0xF',
                    'and ebx, 0xF',
                    # 计算(y + 1) % 2
                    'mov ecx, ebx',
                    'add ecx, 1',
                    # 进行比较
                    'and ecx, 1',
                    'jz .true'
                    ],
        'transit': [
                     # 计算(2*x + 1) * y
                    'add eax, eax',
                    'add eax, 1',
                    'imul eax, ebx',
                    # 进行比较
                    'test eax, 1',
                    'jz .true'
                    ],
        'false': [
                    'leave'
                    ],
        'true': [
                    'pop ecx',
                    'pop ebx',
                    'popf',
                    'pop eax',
                    'jmp SELECT'
                    ]
        }
    ]
    
    opaque_predicate_32_3 = OpaquePredicate()
    opaque_predicate_32_3.basic_block_i_groups = [
        {
        'entry': [
                    # 保存容器状态
                    'push eax',
                    'pushf',
                    'push ebx',
                    'push ecx',
                    # 溢出保护
                    'and eax, 0xF',
                    'and ebx, 0xF',
                    'and ecx, 0xF',
                    # 计算y^2 - 1
                    'imul eax, eax',
                    'imul ebx, ebx',
                    'mov ecx, eax',
                    # 进行比较
                    'cmp ecx, 1',
                    'jl .transit'
                    ],
        },
        {
        'true': [
                'pop ecx',
                'pop ebx',
                'popf',
                'pop eax',
                'jmp SELECT'
                ]
        },
        {
        'transit': [
                    # 计算(x * y)^2
                    'mov ecx, ebx',
                    'imul ecx, eax',
                    # 计算x^2 + 1
                    'add ebx, 1',
                    # 进行比较
                    'cmp ecx, ebx',
                    'jnge .true'
                    ],
        'false': [
                    'leave'
                    ]
        }
    ]
    
    opaque_predicates_str = {
            0: opaque_predicate_64_0,
            1: opaque_predicate_64_1,
            2: opaque_predicate_64_2,
            3: opaque_predicate_64_3
        }

    total_instruction_count = 0
    for key in opaque_predicates_str.keys():
        instruction_count = opaque_predicates_str[key].instruction_count()
        total_instruction_count += instruction_count
    
    # 预设不透明谓词头种类个数
    opaque_predicate_function_num = len(opaque_predicates_str.keys())
    opaque_predicate_avg_len = int(total_instruction_count / opaque_predicate_function_num)
    
    def __init__(self):

        # 不透明谓词构造函数对应表
        self.opaque_predicate_function_dict = {
            0: self.insert_opaque_predicate_0,
            1: self.insert_opaque_predicate_1,
            2: self.insert_opaque_predicate_2,
            3: self.insert_opaque_predicate_3
        }
    
    def note_selected_basic_block_in_edges(self, selected_basic_block):
        # 解析目标code_block
        target_code_block = selected_basic_block.ir_basic_block[0]
        # 找出被选择节点的所有原始前驱code_block
        selected_basic_block_sequential_precursors = []
        for edge in target_code_block.incoming_edges:
            # 只记录跳转边
            if edge.label.type == BPE_CFR.gtirb.Edge.Type.Branch:
                selected_basic_block_sequential_precursors.append(edge.source)
        return selected_basic_block_sequential_precursors
    
    def include_indirect_jump_to_entry(self, selected_basic_block_sequential_precursors):
        for source_code_block in selected_basic_block_sequential_precursors:
            # 获取该代码块的所有出边
            for edge in source_code_block.outgoing_edges:
                # 检查边的标签是否为间接跳转
                if edge.label and not edge.label.direct:
                    # 这是间接跳转
                    return True
        return False
    
    def redirect_to_entry_basic_block(self, function, selected_basic_block_sequential_precursors, entry_symbol, decoder):
        # 重定向被选择节点的原始顺序前驱 -> entry节点
        for source_code_block in selected_basic_block_sequential_precursors:
            # 重建
            self.rebuild_edge(function, source_code_block, entry_symbol, decoder)
    
    # 检查selected_basic_block_index的前一个块，决定是否需要提前在段首插入一个指向当前块的jmp来确保entry分块
    def check_fallthrough_block_to_entry(self, function, selected_basic_block_index, i_group, decoder):
        # 定位插入点
        target_code_block, offset_in_code_block, target_instruction = function.index_to_location_pre(selected_basic_block_index, 0, decoder)
        add_jmp = False
        if i_group.i_basic_blocks[0].symbol:
            if target_instruction:
                # 根据插入点的前一条指令来判断是否需要加入额外的jmp连接到插入的段
                if not target_instruction.group(BPE_CFR.CS_GRP_JUMP) and not target_instruction.group(BPE_CFR.CS_GRP_RET):
                    # 构造一条新的i_instruction单独成块
                    jmp_iis_list = [BPE_CFR.IInstruction('jmp {}\n'.format(i_group.i_basic_blocks[0].symbol))]
                    jmp_ibb_list = [BPE_CFR.IBasicBlock(jmp_iis_list, symbol=None)]
                    # 将新块置于要插入的i_group最前面
                    i_group.i_basic_blocks = jmp_ibb_list + i_group.i_basic_blocks
                    add_jmp = True
        return add_jmp, i_group
    
    # 在全部插入完成后，提前插入的jmp会被重定向到新entry
    def rebuild_edge(self, function, source_code_block, target_symbol, decoder):
        # 找到跳转指令的位置
        find_flag_at_end, jmp_instruction_offset, jmp_instruction_len, jmp_str = function._find_jmp_in_code_block(source_code_block, decoder)
        # 做为边的起始，source_code_block的最后一条指令必须是jmp或cjmp，这要求在code_block插入时就包含jmp指令，必要时需要使用原地跳转作为占位符
        assert find_flag_at_end
        # 构造实体边
        ctx = function.get_context()
        # target_symbol = target_basic_block.symbol[0].name
        ctx.replace_at(source_code_block, jmp_instruction_offset, jmp_instruction_len, BPE_CFR.literal_patch(jmp_str + ' ' + target_symbol))
        ctx.apply()
        # 刷新cfr边
        function.update_basic_blocks_successors()
    
    # 均值不等式：x^2 + y^2 >= 2*x*y
    # 不需要分段插入
    def insert_opaque_predicate_0(self, function, selected_basic_block_index, decoder, i):
        # 处理在函数开头插入的情况（此时没有上一个block可以用于参照）
        if selected_basic_block_index == 0:
            byte_interval, start_offset, target_code_block, function_start_symbol, target_code_block_symbol = function.insert_group_at_head_preparation(decoder)
            # 选择使用的不透明谓词
            opaque_predicate = self.opaque_predicates_str[0].to_i_group(target_code_block_symbol[0].name, i)
            # 执行插入
            function.insert_group_at_head(target_code_block, function_start_symbol, byte_interval, start_offset, opaque_predicate[0], decoder)
        else:
            # 解析选择节点
            selected_basic_block = function[selected_basic_block_index]
            # 处理被选择的基本块没有symbol的情况，生成一个symbol并绑定
            if len(selected_basic_block.symbol) == 0:
                created_selected_basic_block_symbol = BPE_CFR.gtirb.Symbol(name=".L_{}".format(selected_basic_block.ir_basic_block[0].address))
                created_selected_basic_block_symbol.referent = selected_basic_block.ir_basic_block[0]
                selected_basic_block.ir_basic_block[0].byte_interval.module.symbols.add(created_selected_basic_block_symbol)
                selected_basic_block.symbol.append(created_selected_basic_block_symbol)

            selected_basic_block_symbol = selected_basic_block.symbol[0].name
            # 选择使用的不透明谓词
            opaque_predicate = self.opaque_predicates_str[0].to_i_group(selected_basic_block_symbol, i)
            # 记录指向入口的跳转（只记录跳转边）
            selected_basic_block_sequential_precursors = self.note_selected_basic_block_in_edges(selected_basic_block)
            if self.include_indirect_jump_to_entry(selected_basic_block_sequential_precursors):
                return 0, 0   # 如果存在指向入口的间接跳转，则不执行插入，返回0供reward计算使用
            # 检查selected_basic_block_index的前一个块，决定是否需要提前在段首插入一个指向当前块的jmp来确保entry分块
            add_jmp, opaque_predicate_i_group = self.check_fallthrough_block_to_entry(function, selected_basic_block_index, opaque_predicate[0], decoder)
            # 执行插入
            # 一次性插入所有块
            function.insert_group(selected_basic_block_index, 0, opaque_predicate_i_group, decoder)
            #print('插入混淆块后：')
            #function.show(decoder)
            #function.CFG(decoder).draw('CFG_opaque_predicate_insert_all.png', decoder=decoder)
            # 将指向原入口的跳转重定向指向到entry块
            self.redirect_to_entry_basic_block(function, selected_basic_block_sequential_precursors, '.entry_' + str(i), decoder)
            #print('入口跳转重定向后：')
            #function.show(decoder)
            #function.CFG(decoder).draw('CFG_opaque_predicate_insert_000.png', decoder=decoder)
            if add_jmp:
                # 补全entry的顺序关系，将其转换为跳转，以保证分块
                self.rebuild_edge(function, function[selected_basic_block_index - 1].ir_basic_block[-1], '.entry_' + str(i), decoder)
        return 21, 1   # 返回插入的指令数，供reward计算使用
    
    # 奇偶数：x * (x- 1) % 2 == 1
    # 需要分段插入，易出问题
    def insert_opaque_predicate_1(self, function, selected_basic_block_index, decoder, i):
        # 处理在函数开头插入的情况（此时没有上一个block可以用于参照）
        if selected_basic_block_index == 0:
            byte_interval, start_offset, target_code_block, function_start_symbol, target_code_block_symbol = function.insert_group_at_head_preparation(decoder)
            # 选择使用的不透明谓词
            opaque_predicate = self.opaque_predicates_str[1].to_i_group(target_code_block_symbol[0].name, i)
            # 执行插入
            function.append_group(opaque_predicate[1], decoder)
            function.insert_group_at_head(target_code_block, function_start_symbol, byte_interval, start_offset, opaque_predicate[0], decoder)
        else:
            # 解析选择节点
            selected_basic_block = function[selected_basic_block_index]
            # 处理被选择的基本块没有symbol的情况，生成一个symbol并绑定
            if len(selected_basic_block.symbol) == 0:
                created_selected_basic_block_symbol = BPE_CFR.gtirb.Symbol(name=".L_{}".format(selected_basic_block.ir_basic_block[0].address))
                created_selected_basic_block_symbol.referent = selected_basic_block.ir_basic_block[0]
                selected_basic_block.ir_basic_block[0].byte_interval.module.symbols.add(created_selected_basic_block_symbol)
                selected_basic_block.symbol.append(created_selected_basic_block_symbol)
                
            selected_basic_block_symbol = selected_basic_block.symbol[0].name
            # 选择使用的不透明谓词
            opaque_predicate = self.opaque_predicates_str[1].to_i_group(selected_basic_block_symbol, i)
            # 检查selected_basic_block_index的前一个块，决定是否需要提前在段首插入一个指向当前块的jmp来确保entry分块
            add_jmp, opaque_predicate_i_group = self.check_fallthrough_block_to_entry(function, selected_basic_block_index, opaque_predicate[0], decoder)
            # 记录指向入口的跳转（只记录跳转边）
            selected_basic_block_sequential_precursors = self.note_selected_basic_block_in_edges(selected_basic_block)
            if self.include_indirect_jump_to_entry(selected_basic_block_sequential_precursors):
                return 0, 0   # 如果存在指向入口的间接跳转，则不执行插入，返回0供reward计算使用
            # 执行插入
            # 在函数末尾插入true块
            function.append_group(opaque_predicate[1], decoder)
            #print('插入true块后：')
            #function.show(decoder)
            #function.CFG(decoder).draw('CFG_opaque_predicate_insert_transit_true.png', decoder=decoder)
            # 在selected_basic_block前面插入entry块和false块
            function.insert_group(selected_basic_block_index, 0, opaque_predicate_i_group, decoder)
            #print('插入entry、false块后：')
            #function.show(decoder)
            #function.CFG(decoder).draw('CFG_opaque_predicate_insert_transit_entry_false.png', decoder=decoder)
            # 将指向原入口的跳转重定向指向到entry块
            self.redirect_to_entry_basic_block(function, selected_basic_block_sequential_precursors, '.entry_' + str(i), decoder)
            #print('入口跳转重定向后：')
            #function.show(decoder)
            #function.CFG(decoder).draw('CFG_opaque_predicate_insert_000.png', decoder=decoder)
            if add_jmp:
                # 补全entry的顺序关系，将其转换为跳转，以保证分块
                self.rebuild_edge(function, function[selected_basic_block_index - 1].ir_basic_block[-1], '.entry_' + str(i), decoder)
        return 16, 1   # 返回插入的指令数，供reward计算使用
    
    # 奇偶数：(y + 1) % 2 == 0 or ((2*x + 1) * y ) % 2 == 0
    # 不需要分段插入
    def insert_opaque_predicate_2(self, function, selected_basic_block_index, decoder, i):
        # 处理在函数开头插入的情况（此时没有上一个block可以用于参照）
        if selected_basic_block_index == 0:
            byte_interval, start_offset, target_code_block, function_start_symbol, target_code_block_symbol = function.insert_group_at_head_preparation(decoder)
            # 选择使用的不透明谓词
            opaque_predicate = self.opaque_predicates_str[2].to_i_group(target_code_block_symbol[0].name, i)
            # 执行插入
            function.insert_group_at_head(target_code_block, function_start_symbol, byte_interval, start_offset, opaque_predicate[0], decoder)
        else:
            # 解析选择节点
            selected_basic_block = function[selected_basic_block_index]
            # 处理被选择的基本块没有symbol的情况，生成一个symbol并绑定
            if len(selected_basic_block.symbol) == 0:
                created_selected_basic_block_symbol = BPE_CFR.gtirb.Symbol(name=".L_{}".format(selected_basic_block.ir_basic_block[0].address))
                created_selected_basic_block_symbol.referent = selected_basic_block.ir_basic_block[0]
                selected_basic_block.ir_basic_block[0].byte_interval.module.symbols.add(created_selected_basic_block_symbol)
                selected_basic_block.symbol.append(created_selected_basic_block_symbol)

            selected_basic_block_symbol = selected_basic_block.symbol[0].name
            # 选择使用的不透明谓词
            opaque_predicate = self.opaque_predicates_str[2].to_i_group(selected_basic_block_symbol, i)
            # 检查selected_basic_block_index的前一个块，决定是否需要提前在段首插入一个指向当前块的jmp来确保entry分块
            add_jmp, opaque_predicate_i_group = self.check_fallthrough_block_to_entry(function, selected_basic_block_index, opaque_predicate[0], decoder)
            # 记录指向入口的跳转（只记录跳转边）
            selected_basic_block_sequential_precursors = self.note_selected_basic_block_in_edges(selected_basic_block)
            if self.include_indirect_jump_to_entry(selected_basic_block_sequential_precursors):
                return 0, 0   # 如果存在指向入口的间接跳转，则不执行插入，返回0供reward计算使用
            # 执行插入
            # 一次性插入所有块
            function.insert_group(selected_basic_block_index, 0, opaque_predicate_i_group, decoder)
            #print('插入混淆块后：')
            #function.show(decoder)
            #function.CFG(decoder).draw('CFG_opaque_predicate_insert_all.png', decoder=decoder)
            # 将指向原入口的跳转重定向指向到entry块
            self.redirect_to_entry_basic_block(function, selected_basic_block_sequential_precursors, '.entry_' + str(i), decoder)
            #print('入口跳转重定向后：')
            #function.show(decoder)
            #function.CFG(decoder).draw('CFG_opaque_predicate_insert_000.png', decoder=decoder)
            if add_jmp:
                # 补全entry的顺序关系，将其转换为跳转，以保证分块
                self.rebuild_edge(function, function[selected_basic_block_index - 1].ir_basic_block[-1], '.entry_' + str(i), decoder)
        return 21, 2   # 返回插入的指令数，供reward计算使用

    # 倒数：(y^2 - 1 > 0) or ((x*y)^2 < x^2 +1)
    # 需要分段插入，易出问题
    def insert_opaque_predicate_3(self, function, selected_basic_block_index, decoder, i):
        # 处理在函数开头插入的情况（此时没有上一个block可以用于参照）
        if selected_basic_block_index == 0:
            byte_interval, start_offset, target_code_block, function_start_symbol, target_code_block_symbol = function.insert_group_at_head_preparation(decoder)
            # 选择使用的不透明谓词
            opaque_predicate = self.opaque_predicates_str[3].to_i_group(target_code_block_symbol[0].name, i)
            # 执行插入
            function.insert_group_at_head(target_code_block, function_start_symbol, byte_interval, start_offset, opaque_predicate[1], decoder, with_selfload=False, over=False)
            function.append_group(opaque_predicate[2], decoder)
            function.insert_group_at_head(target_code_block, function_start_symbol, byte_interval, start_offset, opaque_predicate[0], decoder)
        else:
            # 解析选择节点
            selected_basic_block = function[selected_basic_block_index]
            # 处理被选择的基本块没有symbol的情况，生成一个symbol并绑定
            if len(selected_basic_block.symbol) == 0:
                created_selected_basic_block_symbol = BPE_CFR.gtirb.Symbol(name=".L_{}".format(selected_basic_block.ir_basic_block[0].address))
                created_selected_basic_block_symbol.referent = selected_basic_block.ir_basic_block[0]
                selected_basic_block.ir_basic_block[0].byte_interval.module.symbols.add(created_selected_basic_block_symbol)
                selected_basic_block.symbol.append(created_selected_basic_block_symbol)

            selected_basic_block_symbol = selected_basic_block.symbol[0].name
            # 选择使用的不透明谓词
            opaque_predicate = self.opaque_predicates_str[3].to_i_group(selected_basic_block_symbol, i)
            # 检查selected_basic_block_index的前一个块，决定是否需要提前在段首插入一个指向当前块的jmp来确保entry分块
            add_jmp, opaque_predicate_i_group = self.check_fallthrough_block_to_entry(function, selected_basic_block_index, opaque_predicate[1], decoder)
            # 记录指向入口的跳转（只记录跳转边）
            selected_basic_block_sequential_precursors = self.note_selected_basic_block_in_edges(selected_basic_block)
            if self.include_indirect_jump_to_entry(selected_basic_block_sequential_precursors):
                return 0, 0   # 如果存在指向入口的间接跳转，则不执行插入，返回0供reward计算使用
            # 执行插入
            # 在selected_basic_block前面插入true块
            function.insert_group(selected_basic_block_index, 0, opaque_predicate_i_group, decoder, with_selfload=False)
            #print('插入true块后：')
            #function.show(decoder)
            #function.CFG(decoder).draw('CFG_opaque_predicate_insert_true.png', decoder=decoder)
            # 在函数末尾插入transit块和false块
            function.append_group(opaque_predicate[2], decoder)
            #print('插入transit、false块后：')
            #function.show(decoder)
            #function.CFG(decoder).draw('CFG_opaque_predicate_insert_transit_false.png', decoder=decoder)
            # 在selected_basic_block前面插入entry块
            function.insert_group(selected_basic_block_index, 0, opaque_predicate[0], decoder)
            #print('插入entry块后：')
            #function.show(decoder)
            #function.CFG(decoder).draw('CFG_opaque_predicate_insert_entry.png', decoder=decoder)
            # 将指向原入口的跳转重定向指向到entry块
            self.redirect_to_entry_basic_block(function, selected_basic_block_sequential_precursors, '.entry_' + str(i), decoder)
            #print('入口跳转重定向后：')
            #function.show(decoder)
            #function.CFG(decoder).draw('CFG_opaque_predicate_insert_000.png', decoder=decoder)
            # 补全entry的顺序关系，将其转换为跳转，以保证分块
            if add_jmp:
                self.rebuild_edge(function, function[selected_basic_block_index - 1].ir_basic_block[-1], '.entry_' + str(i), decoder)
        return 23, 2   # 返回插入的指令数，供reward计算使用
                
# ============================================ 影子模式 ===================================================
class ShadowOpaquePredicate:
    def __init__(self):
        self.bb_template = []
    
    def instruction_count(self):
        """计算该模板中所有基本块的指令总数"""
        count = 0
        for name in self.bb_template.keys():
            count += len(self.bb_template[name])
        return count

    def to_shadow_groups(self):
        """
        将模板指令转化影子块
        """
        group_blocks = {}
        for name, insts in self.bb_template.items():
            s_bb = BPE_CFR.ShadowBasicBlock(idx=None)
            s_bb_list = []
            for asm in insts:
                asm = BPE_utils.normalization(asm)
                s_bb_list.append(BPE_CFR.ShadowInstruction(asm, idx=None))
            s_bb.load(s_bb_list)
            
            group_blocks[name] = s_bb
        return group_blocks

class ShadowOpaquePredicateInserter:
    # 不透明谓词组
    opaque_predicate_64_0 = ShadowOpaquePredicate()
    opaque_predicate_64_0.bb_template = {
        'entry': [
                    # 保存容器状态
                    'push rax',
                    'pushf',
                    'push rbx',
                    'push rcx',
                    # 溢出保护
                    'and rax, 0xF',
                    'and rbx, 0xF',
                    'and rcx, 0xF',
                    # 计算2*x*y
                    'mov rcx, rax',
                    'imul rcx, rbx',
                    'add rcx, rcx',
                    # 计算x^2 + y^2
                    'imul rax, rax',
                    'imul rbx, rbx',
                    'add rax, rbx',
                    # 进行比较
                    'cmp rax, rcx',
                    'jge .true'
                    ],
        'false': [
                    'leave'
                    ],
        'true': [
                    'pop rcx',
                    'pop rbx',
                    'popf',
                    'pop rax',
                    'jmp SELECT'
                    ]
        }   # 21
    
    opaque_predicate_64_1 = ShadowOpaquePredicate()
    opaque_predicate_64_1.bb_template = {
        'entry': [
                    # 保存容器状态
                    'push rax',
                    'pushf',
                    'push rbx',
                    # 溢出保护
                    'and rax, 0xF',
                    'and rbx, 0xF',
                    # 计算x * (x-1) % 2
                    'mov rbx, rax',
                    'sub rbx, 1',
                    'imul rax, rbx',
                    # 进行比较
                    'and rax, 1',
                    'cmp rax, 1',
                    'je .true'
                    ],
        'false': [
                    'pop rbx',
                    'popf',
                    'pop rax',
                    'jmp SELECT'
                    ],
        'true': [
                    'leave'
                    ]
        }   # 16

    
    opaque_predicate_64_2 = ShadowOpaquePredicate()
    opaque_predicate_64_2.bb_template = {
        'entry': [
                    # 保存容器状态
                    'push rax',
                    'pushf',
                    'push rbx',
                    'push rcx',
                    # 溢出保护
                    'and rax, 0xF',
                    'and rbx, 0xF',
                    # 计算(y + 1) % 2
                    'mov rcx, rbx',
                    'add rcx, 1',
                    # 进行比较
                    'and rcx, 1',
                    'jz .true'
                    ],
        'transit': [
                     # 计算(2*x + 1) * y
                    'add rax, rax',
                    'add rax, 1',
                    'imul rax, rbx',
                    # 进行比较
                    'test rax, 1',
                    'jz .true'
                    ],
        'false': [
                    'leave'
                    ],
        'true': [
                    'pop rcx',
                    'pop rbx',
                    'popf',
                    'pop rax',
                    'jmp SELECT'
                    ]
        }   # 21
    
    opaque_predicate_64_3 = ShadowOpaquePredicate()
    opaque_predicate_64_3.bb_template = {
        'entry': [
                    # 保存容器状态
                    'push rax',
                    'pushf',
                    'push rbx',
                    'push rcx',
                    # 溢出保护
                    'and rax, 0xF',
                    'and rbx, 0xF',
                    'and rcx, 0xF',
                    # 计算y^2 - 1
                    'imul rax, rax',
                    'imul rbx, rbx',
                    'mov rcx, rax',
                    # 进行比较
                    'cmp rcx, 1',
                    'jl .transit'
                    ],
        'true': [
                    'pop rcx',
                    'pop rbx',
                    'popf',
                    'pop rax',
                    'jmp SELECT'
                    ],
        'transit': [
                    # 计算(x * y)^2
                    'mov rcx, rbx',
                    'imul rcx, rax',
                    # 计算x^2 + 1
                    'add rbx, 1',
                    # 进行比较
                    'cmp rcx, rbx',
                    'jnge .true'
                    ],
        'false': [
                    'leave'
                    ]
        }   # 23

    opaque_predicate_32_0 = ShadowOpaquePredicate()
    opaque_predicate_32_0.bb_template = {
        'entry': [
                    # 保存容器状态
                    'push eax',
                    'pushf',
                    'push ebx',
                    'push ecx',
                    # 溢出保护
                    'and eax, 0xF',
                    'and ebx, 0xF',
                    'and ecx, 0xF',
                    # 计算2*x*y
                    'mov ecx, eax',
                    'imul ecx, ebx',
                    'add ecx, ecx',
                    # 计算x^2 + y^2
                    'imul eax, eax',
                    'imul ebx, ebx',
                    'add eax, ebx',
                    # 进行比较
                    'cmp eax, ecx',
                    'jge .true'
                    ],
        'false': [
                    'leave'
                    ],
        'true': [
                    'pop ecx',
                    'pop ebx',
                    'popf',
                    'pop eax',
                    'jmp SELECT'
                    ]
        }
    
    opaque_predicate_32_1 = ShadowOpaquePredicate()
    opaque_predicate_32_1.bb_template = {
        'entry': [
                    # 保存容器状态
                    'push eax',
                    'pushf',
                    'push ebx',
                    # 溢出保护
                    'and eax, 0xF',
                    'and ebx, 0xF',
                    # 计算x * (x-1) % 2
                    'mov ebx, eax',
                    'sub ebx, 1',
                    'imul eax, ebx',
                    # 进行比较
                    'and eax, 1',
                    'cmp eax, 1',
                    'je .true'
                    ],
        'false': [
                    'pop ebx',
                    'popf',
                    'pop eax',
                    'jmp SELECT'
                    ],
        'true': [
                    'leave'
                    ]
        }
    
    opaque_predicate_32_2 = ShadowOpaquePredicate()
    opaque_predicate_32_2.bb_template = {
        'entry': [
                    # 保存容器状态
                    'push eax',
                    'pushf',
                    'push ebx',
                    'push ecx',
                    # 溢出保护
                    'and eax, 0xF',
                    'and ebx, 0xF',
                    # 计算(y + 1) % 2
                    'mov ecx, ebx',
                    'add ecx, 1',
                    # 进行比较
                    'and ecx, 1',
                    'jz .true'
                    ],
        'transit': [
                    # 计算(2*x + 1) * y
                    'add eax, eax',
                    'add eax, 1',
                    'imul eax, ebx',
                    # 进行比较
                    'test eax, 1',
                    'jz .true'
                    ],
        'false': [
                    'leave'
                    ],
        'true': [
                    'pop ecx',
                    'pop ebx',
                    'popf',
                    'pop eax',
                    'jmp SELECT'
                    ]
        }
    
    opaque_predicate_32_3 = ShadowOpaquePredicate()
    opaque_predicate_32_3.bb_template = {
        'entry': [
                    # 保存容器状态
                    'push eax',
                    'pushf',
                    'push ebx',
                    'push ecx',
                    # 溢出保护
                    'and eax, 0xF',
                    'and ebx, 0xF',
                    'and ecx, 0xF',
                    # 计算y^2 - 1
                    'imul eax, eax',
                    'imul ebx, ebx',
                    'mov ecx, eax',
                    # 进行比较
                    'cmp ecx, 1',
                    'jl .transit'
                    ],
        'true': [
                'pop ecx',
                'pop ebx',
                'popf',
                'pop eax',
                'jmp SELECT'
                ], 
        'transit': [
                    # 计算(x * y)^2
                    'mov ecx, ebx',
                    'imul ecx, eax',
                    # 计算x^2 + 1
                    'add ebx, 1',
                    # 进行比较
                    'cmp ecx, ebx',
                    'jnge .true'
                    ],
        'false': [
                    'leave'
                    ]
        }
    
    opaque_predicates_str = {
            0: opaque_predicate_64_0,
            1: opaque_predicate_64_1,
            2: opaque_predicate_64_2,
            3: opaque_predicate_64_3
        }

    total_instruction_count = 0
    for key in opaque_predicates_str.keys():
        instruction_count = opaque_predicates_str[key].instruction_count()
        total_instruction_count += instruction_count
    
    # 预设不透明谓词头种类个数
    opaque_predicate_function_num = len(opaque_predicates_str.keys())
    opaque_predicate_avg_len = int(total_instruction_count / opaque_predicate_function_num)
    
    def __init__(self):
        
        # 映射 ID 到具体的影子插入函数
        self.opaque_predicate_function_dict = {
            0: self.insert_opaque_predicate_0,
            1: self.insert_opaque_predicate_1,
            2: self.insert_opaque_predicate_2,
            3: self.insert_opaque_predicate_3
        }
    
    def include_indirect_jump_to_entry(self, selected_basic_block_in_edges):
        for selected_basic_block_in_edge in selected_basic_block_in_edges:
            if selected_basic_block_in_edge.source[-1]._is_indirect_jump():
                return True

    # 重定向被选择节点的前驱到不透明谓词入口节点
    def redirect(self, function, selected_basic_block, entry_basic_block):
        # 找出被选择节点的所有原始前驱
        selected_basic_block_sequential_precursors = set()
        selected_basic_block_in_edges = function.CFG().in_edges(selected_basic_block)
        
        # 拒绝处理带有间接跳转的重定向
        if self.include_indirect_jump_to_entry(selected_basic_block_in_edges):
            return False

        for selected_basic_block_in_edge in selected_basic_block_in_edges:
            selected_basic_block_sequential_precursors.add(selected_basic_block_in_edge.source)
        # 重定向被选择节点的原始顺序前驱 -> entry节点
        for selected_basic_block_sequential_precursor in selected_basic_block_sequential_precursors:
            # 删除
            function.remove_edge(selected_basic_block_sequential_precursor.idx, selected_basic_block.idx)
            # 重建
            function.build_edge(selected_basic_block_sequential_precursor.idx, entry_basic_block.idx)
            # 更新跳转指令
            function.update_jump()
        return True
    
    # 均值不等式：x^2 + y^2 >= 2*x*y
    def insert_opaque_predicate_0(self, function, selected_basic_block_index):
        # 解析选择节点
        selected_basic_block = function[selected_basic_block_index]
        # 选择使用的不透明谓词
        opaque_predicate = self.opaque_predicates_str[0].to_shadow_groups()

        # 1.插入节点并填充内容
        # 在被选择节点前面插入true分支节点
        true_basic_block = opaque_predicate['true']
        function.insert(selected_basic_block.idx, true_basic_block)
        # 在true分支节点前面插入false分支节点
        false_basic_block = opaque_predicate['false']
        function.insert(true_basic_block.idx, false_basic_block)
        # 在false分支节点前面插入entry节点
        entry_basic_block = opaque_predicate['entry']
        function.insert(false_basic_block.idx, entry_basic_block)

        # 2.构造节点关系
        # 入口重定向
        redirect_flag = self.redirect(function, selected_basic_block, entry_basic_block)
        if not redirect_flag:
            return 0, 0   # 返回插入的指令数，供reward计算使用
        # entry节点 -> false分支（顺序后继）
        function.build_edge(entry_basic_block.idx, false_basic_block.idx)
        # entry节点 -> true分支（跳转后继）
        function.build_edge(entry_basic_block.idx, true_basic_block.idx)
        # true分支 -> 被选择节点（跳转后继）
        function.build_edge(true_basic_block.idx, selected_basic_block.idx)
        # 更新跳转指令
        function.update_jump()
        return 21, 1   # 返回插入的指令数，供reward计算使用
    
    # 奇偶数：x * (x- 1) % 2 == 1
    def insert_opaque_predicate_1(self, function, selected_basic_block_index):
        # 解析选择节点
        selected_basic_block = function[selected_basic_block_index]
        # 选择使用的不透明谓词
        opaque_predicate = self.opaque_predicates_str[1].to_shadow_groups()

        # 1.插入节点并填充内容
        # 在被选择节点前面插入false分支节点
        false_basic_block = opaque_predicate['false']
        function.insert(selected_basic_block.idx, false_basic_block)
        # 在false分支节点前面插入entry节点
        entry_basic_block = opaque_predicate['entry']
        function.insert(false_basic_block.idx, entry_basic_block)
        # 在结尾插入true分支节点
        true_basic_block = opaque_predicate['true']
        function.append(true_basic_block)

        # 2.构造节点关系
        # 入口重定向
        redirect_flag = self.redirect(function, selected_basic_block, entry_basic_block)
        if not redirect_flag:
            return 0, 0   # 返回插入的指令数，供reward计算使用
        # entry节点 -> false分支（顺序后继）
        function.build_edge(entry_basic_block.idx, false_basic_block.idx)
        # false分支 -> 被选择节点（跳转后继）
        function.build_edge(false_basic_block.idx, selected_basic_block.idx)
        # entry节点 -> true分支（跳转后继）
        function.build_edge(entry_basic_block.idx, true_basic_block.idx)
        # 更新跳转指令
        function.update_jump()
        return 16, 1   # 返回插入的指令数，供reward计算使用
    
    # 奇偶数：(y + 1) % 2 == 0 or ((2*x + 1) * y ) % 2 == 0
    def insert_opaque_predicate_2(self, function, selected_basic_block_index):
        # 解析选择节点
        selected_basic_block = function[selected_basic_block_index]
        # 选择使用的不透明谓词
        opaque_predicate = self.opaque_predicates_str[2].to_shadow_groups()

        # 1.插入节点并填充内容
        # 在被选择节点前面插入true分支节点
        true_basic_block = opaque_predicate['true']
        function.insert(selected_basic_block.idx, true_basic_block)
        # 在true分支节点前面插入false分支节点
        false_basic_block = opaque_predicate['false']
        function.insert(true_basic_block.idx, false_basic_block)
        # 在false分支节点前面插入transit分支节点
        transit_basic_block = opaque_predicate['transit']
        function.insert(false_basic_block.idx, transit_basic_block)
        # 在transit分支节点前面插入entry分支节点
        entry_basic_block = opaque_predicate['entry']
        function.insert(transit_basic_block.idx, entry_basic_block)
        
        # 2.构造节点关系
        # 入口重定向
        redirect_flag = self.redirect(function, selected_basic_block, entry_basic_block)
        if not redirect_flag:
            return 0, 0   # 返回插入的指令数，供reward计算使用
        # entry节点 -> transit分支（顺序后继）
        function.build_edge(entry_basic_block.idx, transit_basic_block.idx)
        # transit分支 -> false分支（顺序后继）
        function.build_edge(transit_basic_block.idx, false_basic_block.idx)
        # entry节点 -> true分支（跳转后继）
        function.build_edge(entry_basic_block.idx, true_basic_block.idx)
        # transit分支  -> true分支（跳转后继）
        function.build_edge(transit_basic_block.idx, true_basic_block.idx)
        # true分支  -> 被选择节点（跳转后继）
        function.build_edge(true_basic_block.idx, selected_basic_block.idx)
        # 更新跳转指令
        function.update_jump()
        return 21, 2   # 返回插入的指令数，供reward计算使用

    # 倒数：(y^2 - 1 > 0) or ((x*y)^2 < x^2 +1)
    def insert_opaque_predicate_3(self, function, selected_basic_block_index):
        # 解析选择节点
        selected_basic_block = function[selected_basic_block_index]
        # 选择使用的不透明谓词
        opaque_predicate = self.opaque_predicates_str[3].to_shadow_groups()

        # 1.插入节点并填充内容
        # 在被选择节点前面插入true分支节点
        true_basic_block = opaque_predicate['true']
        function.insert(selected_basic_block.idx, true_basic_block)
        # 在true分支节点前面插入entry分支节点
        entry_basic_block = opaque_predicate['entry']
        function.insert(true_basic_block.idx, entry_basic_block)
        # 在结尾插入transit分支节点
        transit_basic_block = opaque_predicate['transit']
        function.append(transit_basic_block)
        # 在结尾（transit分支节点后面）插入false分支节点
        false_basic_block = opaque_predicate['false']
        function.append(false_basic_block)

        # 2.构造节点关系
        # 入口重定向
        redirect_flag = self.redirect(function, selected_basic_block, entry_basic_block)
        if not redirect_flag:
            return 0, 0   # 返回插入的指令数，供reward计算使用
        # entry节点 -> true分支（顺序后继）
        function.build_edge(entry_basic_block.idx, true_basic_block.idx)
        # true分支 -> 被选择节点（跳转后继）
        function.build_edge(true_basic_block.idx, selected_basic_block.idx)
        # entry节点 -> transit分支（跳转后继）
        function.build_edge(entry_basic_block.idx, transit_basic_block.idx)
        # transit分支  -> true分支（跳转后继）
        function.build_edge(transit_basic_block.idx, true_basic_block.idx)
        # transit分支  -> false分支（顺序后继）
        function.build_edge(transit_basic_block.idx, false_basic_block.idx)
        # 更新跳转指令
        function.update_jump()
        return 23, 2   # 返回插入的指令数，供reward计算使用
        