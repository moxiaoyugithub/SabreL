from envs.binary_process_editor import BPE_CFR


class BasicBlockSpliter:
    def __init__(self):
        pass

    def basic_block_splitting(self, function, selected_basic_block_index, selected_instruction_index, decoder, i):
        """
        在给定指令处拆分一个基本块。

        - 原始基本块 (basicblock) 保留拆分点之前的所有指令。
        - 新基本块 (new_basicblock) 包含拆分点及其之后的所有指令。
        - 更新控制流图以反映新的基本块结构。

        Args:
            selected_basic_block_index: 要拆分的基本块的索引。
            selected_instruction_index: 拆分点参考的指令索引（代码将在此指令处拆分）。
        """
        # 不处理在函数开头插入的情况
        if selected_basic_block_index == 0 and selected_instruction_index == 0:
            return 0, 0
        else:   # 不会出现在结尾之后切分的情况
            # 解析选择节点
            #selected_basic_block = function[selected_basic_block_index]
            #selected_instruction = selected_basic_block[selected_instruction_index]
            # 构造拆分指令块
            jmp_iis_list = [BPE_CFR.IInstruction('jmp {}\n'.format('.splitting_' + str(i)))]
            jmp_ibb = BPE_CFR.IBasicBlock(jmp_iis_list, symbol=None)

            entry_iis_list = []
            entry_ibb = BPE_CFR.IBasicBlock(entry_iis_list, symbol='.splitting_' + str(i))
            ibbs_list = [
                jmp_ibb,
                entry_ibb,
            ]
            splitting_i_group = BPE_CFR.IGroup(ibbs_list)
            # 执行拆分
            function.insert_group(selected_basic_block_index, selected_instruction_index, splitting_i_group, decoder, with_selfload=True)
        return 1, 0   # 返回插入的指令数，供reward计算使用

# ============================================ 影子模式 ===================================================
class ShadowBasicBlockSpliter:
    def __init__(self):
        pass
    
    # 重定向被选择节点的前驱到新的入口节点
    def redirect(self, function, selected_basic_block, entry_basic_block):
        # 找出被选择节点的所有原始前驱
        selected_basic_block_sequential_precursors = set()
        selected_basic_block_in_edges = function.CFG().in_edges(selected_basic_block)

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

    def basic_block_splitting(self, function, selected_basic_block_index, selected_instruction_index):
        """
        在给定指令处拆分一个基本块。

        - 新基本块 (new_basicblock) 包含拆分点之前的所有指令。
        - 原始基本块 (basicblock) 保留拆分点及其之后的所有指令。
        - 更新控制流图以反映新的基本块结构。

        Args:
            selected_basic_block_index: 要拆分的基本块的索引。
            selected_instruction_index: 拆分点参考的指令索引（代码将在此指令处拆分）。
        """
        # 不处理在函数开头插入的情况
        if selected_basic_block_index == 0 and selected_instruction_index == 0:
            return 0, 0
        else:   # 不会出现在结尾之后切分的情况
            # 解析选择节点
            selected_basic_block = function[selected_basic_block_index]

            # 1.插入节点并填充内容
            entry_basic_block = BPE_CFR.ShadowBasicBlock(idx=None)
            for _ in range(selected_instruction_index):
                # 弹出被选择基本块指定位置之前的指令
                instruction = selected_basic_block.pop(0)
                # 收集需要转移至新基本块的指令
                entry_basic_block.append(instruction)
            
            # 拆分用的jmp指令
            entry_basic_block.append(BPE_CFR.ShadowInstruction('jmp SELECT', idx=None))
            
            # 插入新的入口块
            function.insert(selected_basic_block.idx, entry_basic_block)
            
            # 2.构造节点关系
            # 入口重定向
            self.redirect(function, selected_basic_block, entry_basic_block)
            # entry节点 -> 被选择节点
            function.build_edge(entry_basic_block.idx, selected_basic_block.idx)
            # 更新跳转指令
            function.update_jump()
        return 1, 0   # 返回插入的指令数，供reward计算使用