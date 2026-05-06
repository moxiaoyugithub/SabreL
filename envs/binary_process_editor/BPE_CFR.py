import os
import re
import gtirb
import gtirb_functions
import gtirb_rewriting
import capstone_gt

import networkx as nx
import pygraphviz as pgv

from copy import deepcopy
from typing import List, Set, MutableSet, NamedTuple
from gtirb_capstone.instructions import GtirbInstructionDecoder

from keystone import *
from capstone import *

from envs.binary_process_editor import BPE_utils

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# Capstone架构对应表
CAPSTONE_ARCH_MAP = {
    'X86': CS_ARCH_X86,
    'ARM': CS_ARCH_ARM,
    'ARM64': CS_ARCH_ARM64,
    'MIPS': CS_ARCH_MIPS,
    'PPC': CS_ARCH_PPC,
    'SPARC': CS_ARCH_SPARC,
    # 添加其他架构
}

# Keystone架构对应表
KEYSTONE_ARCH_MAP = {
    'X86': KS_ARCH_X86,
    'ARM': KS_ARCH_ARM,
    'ARM64': KS_ARCH_ARM64,
    'MIPS': KS_ARCH_MIPS,
    'PPC': KS_ARCH_PPC,
    'SPARC': KS_ARCH_SPARC,
    # 添加其他架构
}

# Capstone模式对应表
CAPSTONE_MODE_MAP = {
    'LITTLE_ENDIAN': CS_MODE_LITTLE_ENDIAN,
    'BIG_ENDIAN': CS_MODE_BIG_ENDIAN,
    'ARM': CS_MODE_ARM,
    'THUMB': CS_MODE_THUMB,
    'V8': CS_MODE_V8,
    '32': CS_MODE_32,
    '64': CS_MODE_64,
    # 添加其他模式
}

# Keystone模式对应表
KEYSTONE_MODE_MAP = {
    'LITTLE_ENDIAN': KS_MODE_LITTLE_ENDIAN,
    'BIG_ENDIAN': KS_MODE_BIG_ENDIAN,
    'ARM': KS_MODE_ARM,
    'THUMB': KS_MODE_THUMB,
    'V8': KS_MODE_V8,
    '32': KS_MODE_32,
    '64': KS_MODE_64,
    # 添加其他模式
}

def literal_patch(asm: str) -> gtirb_rewriting.Patch:  
    """  
    Creates a patch from a literal string. The patch will have an empty  
    constraints object.  
    """  
      
    @gtirb_rewriting.patch_constraints(x86_syntax=gtirb_rewriting.X86Syntax.INTEL)  
    def patch(ctx):
        return asm
      
    return gtirb_rewriting.Patch.from_function(patch)

class IInstruction:
    def __init__(self, instruction_str):
        self.str = instruction_str

# 指令
class Instruction:

    NON_JMP = 0
    JMP = 1
    CJMP = 2
    CALL = 3
    RET = 4

    def __init__(self, idx, basic_block=None):        
        self.basic_block = basic_block

        # 特殊绘图颜色标记
        self.mark = False
        
        self.idx = idx
    
    def disassemble(self, decoder):
        instructions_capstone = self.basic_block.disassemble(decoder)
        # instructions_capstone = decoder.get_instructions(self.basic_block)
        return instructions_capstone[self.idx]

    def str(self, decoder):
        instruction_capstone = self.disassemble(decoder)
        instruction_str = instruction_capstone.mnemonic + ' ' + instruction_capstone.op_str
        return instruction_str
    
    # 输出该条指令的跳转类型
    def jump_type(self, decoder):
        if self.is_jmp(decoder):
            return self.JMP        # jmp
        elif self.is_cjmp(decoder):
            return self.CJMP       # cjmp
        elif self.is_call(decoder):
            return self.CALL       # call
        elif self.is_ret(decoder):
            return self.RET        # ret
        else:
            return self.NON_JMP    # 非跳转种类

        # 检查指令是否属于无条件跳转指令
    def is_jmp(self, decoder):
        instruction_capstone = self.disassemble(decoder)
        if instruction_capstone.group(CS_GRP_JUMP) and instruction_capstone.mnemonic == 'jmp':
            return True
        else:
            return False
    
    # 检查指令是否属于条件跳转指令
    def is_cjmp(self, decoder):
        instruction_capstone = self.disassemble(decoder)
        if instruction_capstone.group(CS_GRP_JUMP) and not instruction_capstone.mnemonic.startswith('jmp'):
            return True
        else:
            return False
    
    # 检查指令是否属于函数调用
    def is_call(self, decoder):
        instruction_capstone = self.disassemble(decoder)
        return instruction_capstone.group(CS_GRP_CALL)
    
    # 检查指令是否属于返回
    def is_ret(self, decoder):
        instruction_capstone = self.disassemble(decoder)
        return instruction_capstone.group(CS_GRP_RET)
    
    def show(self, decoder):
        instruction_capstone = self.disassemble(decoder)
        print('\tBytes: ({})'.format(instruction_capstone.bytes))
        print('\t[{}] {} {}'.format(instruction_capstone.address, instruction_capstone.mnemonic, instruction_capstone.op_str))
    
    def __hash__(self):
        return hash(id(self))
    
    def __eq__(self, other):
        if not isinstance(other, Instruction):
            return False
        return id(self) == id(other)
    
    def __repr__(self):
        return f"Instruction(idx={self.idx})"

# 基本块-加载体
class IBasicBlock:
    def __init__(self, i_instructions, symbol):
        self.i_instructions = i_instructions
        self.symbol = symbol


# 基本块
class BasicBlock(list):
    def __init__(self, idx, decoder, code_blocks=None, function=None):
        super().__init__()
        self.function = function

        self.ir_basic_block = code_blocks

        if code_blocks:
            self.symbol = list(code_blocks[0].references)
        else:
            self.symbol = None

        self.successors = []

        i = 0
        if code_blocks:
            for code_block in code_blocks:
                instructions_capstone = decoder.get_instructions(code_block)
                for instruction_capstone in instructions_capstone:
                    super().append(Instruction(i, self))
                    i += 1
        
        # 特殊绘图颜色标记
        self.mark = False
        
        self.idx = idx
    
    def update_instructions_idx(self):
        for idx, instruction in enumerate(self):
            instruction.idx = idx
    
    # 根据index解目标指令的所属的code_block，以及它在其中的偏移和长度
    def index_to_location(self, index, decoder):
        current_index = 0
        # 遍历逻辑基本块中的所有物理块
        for code_block in self.ir_basic_block:
            # 反汇编当前块的所有指令
            offset_in_block = 0
            for instruction in decoder.get_instructions(code_block):
                # offset_in_block += instruction.size
                if current_index == index:
                    # 找到目标指令
                    return code_block, offset_in_block, instruction.size
                current_index += 1
                offset_in_block += instruction.size
        raise IndexError(f"Instruction index {index} out of range (total instructions: {current_index})")
    
    # 获得重写上下文
    def get_context(self):
        module = self.function.cfr.ir.modules[0]
        functions = gtirb_functions.Function.build_functions(module)
        ctx = gtirb_rewriting.RewritingContext(module, functions)
        return ctx
    
    # 收集该范围内的所有code_block块
    def collect_code_blocks(self, byte_interval, start_offset, end_offset):
        new_blocks = []  
        for block in sorted(byte_interval.blocks, key=lambda b: b.offset):  
            if isinstance(block, gtirb.CodeBlock):  
                if start_offset <= block.offset < end_offset:
                    new_blocks.append(block)
        
        self.ir_basic_block = new_blocks
    
    # 在byte_interval中查找offset出的块
    def _find_code_block_at_offset(self, byte_interval, offset):
        """查找指定偏移的块"""  
        for code_block in byte_interval.blocks:
            if isinstance(code_block, gtirb.CodeBlock) and code_block.offset == offset:
                return code_block
        return None
    
    # 记录插入前的位置信息
    def record_before_modification(self, target_code_block):
        byte_interval = self.ir_basic_block[0].byte_interval
        target_code_block_offset = target_code_block.offset
        start_offset = min(b.offset for b in self.ir_basic_block if b.byte_interval)
        old_end_offset = max(b.offset + b.size for b in self.ir_basic_block if b.byte_interval)
        return byte_interval, target_code_block_offset, start_offset, old_end_offset
    
    # 增量操作，此时只需考虑是否产生新code_block来计算end_offset
    def get_new_end_offset_after_add(self, byte_interval, old_code_block_at_offset, target_code_block_offset, old_end_offset):
        # 查找插入点原偏移位置的code_block
        code_block = self._find_code_block_at_offset(byte_interval, target_code_block_offset)
        # 原插入处还是以前的code_block，这意味着没有新块产生，此时end_offset就是最后一个code_block的offset+size，由于apply已经新
        if old_code_block_at_offset.uuid == code_block.uuid:
            new_end_offset = self.ir_basic_block[-1].offset + self.ir_basic_block[-1].size
        # 有新code_block产生，由于所有插入操作一次只插入一条指令，所以最多只可能产生一个新的code_block，此时end_offset等于原本end_offset+new_code_block.size
        else:
            new_end_offset = old_end_offset + code_block.size
        
        return new_end_offset
    
    # 去除操作，被去掉的instruction_len是可以得知的，可直接用来计算end_offset
    def get_new_end_offset_after_delete(self, instruction_len, old_end_offset):
        new_end_offset = old_end_offset - instruction_len
        return new_end_offset

    # 重写以增加对实体的操作
    def append(self, i_instruction):
        # 解位置
        offset_in_code_block = 0
        for code_block in self.ir_basic_block:
            offset_in_code_block += code_block.size
        target_code_block = self.ir_basic_block[-1]
        # 记录
        byte_interval, target_code_block_offset, start_offset, old_end_offset = self.record_before_modification(target_code_block)
        # 构造引用
        instruction = Instruction(len(self), self)
        instruction.mark = True
        # 插入引用
        super().append(instruction)
        self.update_instructions_idx()
        # 插入实体
        ctx = self.get_context()
        ctx.insert_at(target_code_block, offset_in_code_block, literal_patch(i_instruction.str))
        ctx.apply()
        # 重建code_block列表
        new_end_offset = self.get_new_end_offset_after_add(byte_interval, target_code_block, target_code_block_offset, old_end_offset)
        self.collect_code_blocks(byte_interval, start_offset, new_end_offset)
    
    # 重写以增加对实体的操作
    def insert(self, index, i_instruction, decoder):
        # 在len处插入和append等价
        if index == len(self) or index == -1:
            self.append(i_instruction)
        else:
            # 解位置
            target_code_block, offset_in_code_block, instruction_size = self.index_to_location(index, decoder)
            # 记录
            byte_interval, target_code_block_offset, start_offset, old_end_offset = self.record_before_modification(target_code_block)
            # 构造引用
            instruction = Instruction(index, self)
            instruction.mark = True
            # 插入引用
            super().insert(index, instruction)
            self.update_instructions_idx()
            # 插入实体
            ctx = self.get_context()
            ctx.insert_at(target_code_block, offset_in_code_block, literal_patch(i_instruction.str))
            ctx.apply()
            # 重建code_block列表
            new_end_offset = self.get_new_end_offset_after_add(byte_interval, target_code_block, target_code_block_offset, old_end_offset)
            self.collect_code_blocks(byte_interval, start_offset, new_end_offset)
    
    # 删除index处的指令
    def pop(self, index, decoder):
        # 解位置
        target_code_block, offset_in_code_block, instruction_len = self.index_to_location(index, decoder)
        # 记录
        byte_interval, target_code_block_offset, start_offset, old_end_offset = self.record_before_modification(target_code_block)
        # 删除实体
        ctx = self.get_context()
        ctx.delete_at(target_code_block, offset_in_code_block, instruction_len)
        ctx.apply()
        # 删除引用
        super().pop(index)
        self.update_instructions_idx()
        # 重建code_block列表
        new_end_offset = self.get_new_end_offset_after_delete(instruction_len, old_end_offset)
        self.collect_code_blocks(byte_interval, start_offset, new_end_offset)

    def remove(self, index):
        pass
    
    # 清空指令和实体
    def clear(self):
        # 清空实体
        ctx = self.get_context()
        for code_block in self.ir_basic_block:
            ctx.delete_at(code_block, 0, code_block.size)
            ctx.apply()
        # 清空引用
        super().clear()
        self.successors = []
        # 清空code_block列表
        self.ir_basic_block = []
    
    # 将整个基本块反汇编
    def disassemble(self, decoder):
        instructions_capstone = []
        for code_block in self.ir_basic_block:
            instructions_capstone += decoder.get_instructions(code_block)
        return instructions_capstone
    
    def str(self, decoder):
        instructions_capstone = self.disassemble(decoder)
        basic_block_str = ''
        for instruction_capstone in instructions_capstone:
            basic_block_str += instruction_capstone.mnemonic + ' ' + instruction_capstone.op_str + '\n'
        return basic_block_str

    def normalization(self, decoder):
        instructions_capstone = self.disassemble(decoder)
        instruction_str_normalization = []
        for instruction_capstone in instructions_capstone:
            # 处理跳转指令
            if instruction_capstone.group(CS_GRP_JUMP):
                jump_target = '?'
                if len(self.successors) == 2:
                    for successor in self.successors:
                        if successor.idx != self.idx + 1:
                            jump_target = successor.idx
                elif len(self.successors) == 1:
                    jump_target = self.successors[0].idx
                instruction_str = instruction_capstone.mnemonic + f' Block_{jump_target}'
            else:
                instruction_str = instruction_capstone.mnemonic + ' ' + instruction_capstone.op_str
            instruction_str_normalization.append(BPE_utils.normalization(instruction_str))
            
        return instruction_str_normalization
    
    # 计算整个basic_block的字节长度
    def bytes_size(self, decoder):
        instructions_capstone = self.disassemble(decoder)
        bytes_size = 0
        for instruction_capstone in instructions_capstone:
            bytes_size += instruction_capstone.size
        return bytes_size
    
    # 显示该基本块的符号
    def show_symbol(self):
        print('symbol: {}'.format(self.symbol))
    
    # 显示该基本块的后继
    def show_successors(self):
        suc = []
        for suc_block in self.successors:
            suc.append(suc_block.idx)
        print('\tSuccessors: {}'.format(suc))
    
    # 反汇编并显示所有指令
    def show_instructions(self, decoder):
        current_code_block = set()
        for code_block in self.ir_basic_block:
            if code_block not in current_code_block:
                current_code_block.add(code_block)
                print('\t[{}]:\t{}'.format(hex(code_block.address), code_block))
            instructions_capstone = decoder.get_instructions(code_block)
            for instruction_capstone in instructions_capstone:
                print('\t[{}] {} {}'.format(hex(instruction_capstone.address), instruction_capstone.mnemonic, instruction_capstone.op_str))
    
    # 显示全部
    def show(self, decoder):
        self.show_symbol()
        self.show_instructions(decoder)
        self.show_successors()
        print('\n')
    
    def __hash__(self):
        return hash(id(self))
    
    def __eq__(self, other):
        if not isinstance(other, BasicBlock):
            return False
        return id(self) == id(other)
    
    def __repr__(self):
        return f"BasicBlock(idx={self.idx, [instruction for instruction in self]})"

# BUG 目前的实现状态：
# CFR
    # append()      ×
    # insert()      ×
    # pop()         ×
    # remove()      ×
# Function
    # append()      √
    # insert()      √
    # pop()         ×
    # remove()      ×
# BasicBlock
    # append()      √
    # insert()      √   不能解决插入jmp和ret导致的基本块分裂问题，只能处理call
    # pop()         ×   还需要实体层支持
    # remove()      ×   还需要实体层支持
# 关于基本块的分裂，目前的方案是：不支持jmp和ret插入导致的自动基本块分裂行为（通过禁止向有实体的code_block内插入jmp和ret实现），而是通过pop和remove移除，再通过新建BasicBlock并append的方式实现

# CFG
    # 添加边                        √
    # 删除边                        √
    # 自动维护插入call产生的内部边    √

# 数据段

# 边
class Edge(
    NamedTuple(
        "NamedTuple",
        (
            ("source", BasicBlock),
            ("target", BasicBlock),
            ("label", BasicBlock),
        ),
    )
):
    class Type:
        Branch = gtirb.EdgeType.Branch              # 跳转边
        # Call = gtirb.EdgeType.Call                # 调用边
        Fallthrough = gtirb.EdgeType.Fallthrough    # 直接边

    def __new__(
            cls,
            source: BasicBlock,
            target: BasicBlock,
            label: BasicBlock = None,
        ) -> "Edge":
            return super().__new__(cls, source, target, label)
    
    def __repr__(self):
        return f"Edge(source idx = {self.source.idx}, target idx = {self.target.idx}, label = {self.label})"

# 图
class CFG(MutableSet[Edge]):

    True_color = 'forestgreen'
    False_color = 'tomato'
    Normal_color = 'royalblue'

    def __init__(self, edges = None):
        self._nxg = nx.MultiDiGraph()

        if edges is not None:
            self.update(edges)

    def _edge_key(self, edge: Edge):
        if edge.source in self._nxg:
            neighbors = self._nxg[edge.source]
            if edge.target in neighbors:
                for key, e in neighbors[edge.target].items():
                    if "label" in e and e["label"] == edge.label:
                        return key
        return None

    def __contains__(self, edge):
        return isinstance(edge, Edge) and self._edge_key(edge) is not None

    def __iter__(self):
        for s, t, l in self._nxg.edges(data="label"):
            yield Edge(s, t, l)

    def __len__(self):
        return len(self._nxg.edges())

    def update(self, edges):
        for edge in edges:
            self.add(edge)

    def add(self, edge: Edge):
        if edge not in self:
            self._nxg.add_edge(edge.source, edge.target, label=edge.label)
    
    def nodes(self):
        return self._nxg.nodes()
    
    def edges(self):
        for s, t, l in self._nxg.edges(data="label"):
            yield Edge(s, t, l)

    def clear(self):
        self._nxg.clear()

    def discard(self, edge: Edge):
        key = self._edge_key(edge)
        if key is not None:
            self._nxg.remove_edge(edge.source, edge.target, key=key)

    def out_edges(self, node):
        if node in self._nxg:
            for s, t, l in self._nxg.out_edges(node, data="label"):
                yield Edge(s, t, l)

    def in_edges(self, node):
        if node in self._nxg:
            for s, t, l in self._nxg.in_edges(node, data="label"):
                yield Edge(s, t, l)

    def nx(self):
        return self._nxg

    # 绘制CFG
    def draw(self, png_name, decoder):
        CFG = self.nx()
        for node in CFG:
            address = node.ir_basic_block[0].address
            name = ': '
            if node.symbol:
                name += node.symbol[0].name
            CFG.nodes[node]['index'] = node.idx
            idx = '(' + str(node.idx) + ')'
            label = idx + '#' + hex(address) + '#' + name 
            label = f'''<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                        <tr><td align="left">{label}</td></tr>
                        '''
            instructions_capstone = node.disassemble(decoder)
            for instruction_capstone, instruction in zip(instructions_capstone, node):
                    line = f"[0x{instruction_capstone.address:x}]:&nbsp;{instruction_capstone.mnemonic}&nbsp;{instruction_capstone.op_str}"
                    if instruction.mark:
                        label += f'''<tr><td align="left"><font color="red">{line}</font></td></tr>'''
                    else:
                        label += f'''<tr><td align="left">{line}</td></tr>'''
            label += f'''</TABLE>>'''
            CFG.nodes[node]['label'] = label
            CFG.nodes[node]['color'] = 'darkcyan'
            if node.mark:
                CFG.nodes[node]['fillcolor'] = 'gold'
            else:
                CFG.nodes[node]['fillcolor'] = 'aquamarine'
        
        # 设定边颜色
        for edge in self:
            source = edge.source
            target = edge.target
            if source[-1].jump_type(decoder) == Instruction.CJMP:
                if edge.label.type == gtirb.EdgeType.Branch:
                    CFG[source][target][0]['color'] = self.True_color
                else:
                    CFG[source][target][0]['color'] = self.False_color
            else:
                CFG[source][target][0]['color'] = self.Normal_color
        
        if len(CFG.nodes()) == 1:
            return

        # 创建一个空的 PyGraphviz 图
        CFG_pgv = pgv.AGraph(strict=True, directed=True)

        # 将 NetworkX 的节点和属性添加到 PyGraphviz 图中
        for node in CFG.nodes():
            index = CFG.nodes[node]['index']

            label = CFG.nodes[node]['label']
            color = CFG.nodes[node]['color']
            fillcolor = CFG.nodes[node]['fillcolor']
            
            CFG_pgv.add_node(index, label=label, color=color, style='filled', fillcolor=fillcolor, shape='box')

        # 将 NetworkX 的边添加到 PyGraphviz 图中
        for source, target, attr in CFG.edges(data=True):
            source_index = CFG.nodes[source]['index']
            target_index = CFG.nodes[target]['index']

            color = attr['color']
            label = ''
            if color == self.True_color:
                label = 'T'
            elif color == self.False_color:
                label = 'F'
            CFG_pgv.add_edge(source_index, target_index, color=color, label=label, fontcolor=color)
        
        # 绘图并保存
        CFG_pgv.layout(prog='dot')
        CFG_pgv.draw(png_name)
        return CFG_pgv

    def __repr__(self) -> str:
        return "CFG(%r)" % list(self)

# 基本块-加载体
class IFunction:
    def __init__(self, i_basic_blocks, label):
        self.i_basic_blocks = i_basic_blocks
        self.label = label

class IGroup:
    def __init__(self, i_basic_blocks):
        self.i_basic_blocks = i_basic_blocks
    
    def show(self):
        print('i_group:')
        for block_idx, i_basic_block in enumerate(self.i_basic_blocks):
            print('block_idx', block_idx)
            print('symbol:', i_basic_block.symbol)
            for i_instruction in i_basic_block.i_instructions:
                print('\t', i_instruction.str)

# 函数
class Function(list):
    def __init__(self, decoder, function, name, cfr=None):
        super().__init__()
        self.cfr = cfr
        
        # 标记
        self.name = name

        # 实体
        self.function = function

        # 加载引用
        code_blocks = list(self.function.get_all_blocks())
        basic_block_list = self.group_blocks_ida_style(code_blocks, decoder)
        self.selfload(basic_block_list, decoder)

        # 特殊绘图颜色标记
        self.mark = False
    
    def selfload(self, basic_block_list, decoder):
        super().clear()
        # 加载引用
        for idx, basic_block in enumerate(basic_block_list):
            super().append(BasicBlock(idx, decoder, basic_block, self))
        
        self.update_basic_blocks_successors()
    
    def group_blocks_ida_style(self, code_blocks, decoder) -> List[List[gtirb.CodeBlock]]:
        """  
        按照 IDA 风格划分基本块:  
        1. call 不分割块  
        2. 跳转目标开始新块  
        3. 条件/无条件跳转结束当前块  
        """  
          
        if not code_blocks:  
            return []  
        
        # 识别所有跳转目标块  
        jump_targets: Set[gtirb.CodeBlock] = set()  
        for block in code_blocks:  
            for edge in block.incoming_edges:  
                # Branch 边(非 Call)表示跳转  
                if edge.label and edge.label.type == gtirb.Edge.Type.Branch:  
                    if isinstance(edge.target, gtirb.CodeBlock):  
                        jump_targets.add(edge.target)  
        
        # 识别以跳转指令结束的块  
        blocks_ending_with_jump: Set[gtirb.CodeBlock] = set()  
        for block in code_blocks:  
            instructions = list(decoder.get_instructions(block))  
            if instructions:  
                last_inst = instructions[-1]  
                # 检查是否是跳转指令(条件或无条件)  
                if (last_inst.group(capstone_gt.CS_GRP_JUMP) and   
                    not last_inst.group(capstone_gt.CS_GRP_CALL)):  
                    blocks_ending_with_jump.add(block)  
        
        visited = set()  
        basic_block_list = []  
        
        # 按地址排序  
        sorted_blocks = sorted(code_blocks, key=lambda b: b.address if b.address else 0)  
        
        for start_block in sorted_blocks:  
            if start_block in visited:  
                continue  
            
            current_group = []  
            current = start_block  
            
            while current and current not in visited:  
                visited.add(current)  
                current_group.append(current)  
                
                # 如果当前块以跳转指令结束,停止合并  
                if current in blocks_ending_with_jump:  
                    break  
                
                # 查找 fallthrough 出边  
                fallthrough_edges = [  
                    edge for edge in current.outgoing_edges  
                    if edge.label and edge.label.type == gtirb.Edge.Type.Fallthrough  
                ]  
                
                if len(fallthrough_edges) == 1:  
                    next_block = fallthrough_edges[0].target  
                    
                    # 如果下一个块是跳转目标,停止合并  
                    if isinstance(next_block, gtirb.CodeBlock) and next_block in code_blocks:  
                        if next_block in jump_targets:  
                            break  
                        current = next_block  
                    else:  
                        break  
                else:  
                    break  
            
            if current_group:  
                basic_block_list.append(current_group)  
        
        return basic_block_list 
    
    def update_basic_blocks_idx(self):
        for idx, basic_block in enumerate(self):
            basic_block.idx = idx

    # 在byte_interval中查找offset出的块
    def _find_code_block_at_offset(self, byte_interval, offset):
        """查找指定偏移的块"""  
        for code_block in byte_interval.blocks:
            if isinstance(code_block, gtirb.CodeBlock) and code_block.offset == offset:
                return code_block
        return None
    
    # 获得重写上下文
    def get_context(self):
        module = self.cfr.ir.modules[0]
        functions = gtirb_functions.Function.build_functions(module)
        ctx = gtirb_rewriting.RewritingContext(module, functions)
        return ctx

    # 计算该函数所有的code_block总大小
    def total_size(self):
        code_blocks = self.function.get_all_blocks()
        total_size = 0
        for code_block in code_blocks:
            total_size += code_block.size
        return total_size
    
    # 整理后继
    def update_basic_blocks_successors(self):
        for basic_block in self:
            basic_block.successors.clear()
            for out_edge in basic_block.ir_basic_block[-1].outgoing_edges:
                if isinstance(out_edge.target, gtirb.CodeBlock):
                    find = False
                    for find_basic_block in self:
                        #if out_edge.target in find_basic_block.code_blocks:
                        if out_edge.target == find_basic_block.ir_basic_block[0]:
                            find = find_basic_block.idx
                    if find:
                        basic_block.successors.append(self[find])
    
    # 输出CFG
    def CFG(self, decoder):
        cfg = CFG()
        if len(self) == 1:
            cfg._nxg.add_node(self[0])
            return cfg
        
        for basic_block in self:
            for target_basic_block in basic_block.successors:
                label = self.edge_type(basic_block.idx, target_basic_block.idx, decoder)
                edge = Edge(source=basic_block, target=target_basic_block, label=label)
                cfg.add(edge)
        return cfg
    
    # 重写以增加对实体的操作
    def append(self, i_basic_block, decoder):
        # 解位置
        target_code_block = self[-1].ir_basic_block[-1]
        offset_in_code_block = target_code_block.size
        # 记录
        start_offset = self[0].ir_basic_block[0].offset
        byte_interval = target_code_block.byte_interval
        old_blocks = set(byte_interval.blocks)
        # 插入实体
        basic_block_str = "{}:\n".format(i_basic_block.symbol)
        for i_instruction in i_basic_block.i_instructions:
            basic_block_str += i_instruction.str
            basic_block_str += '\n'
            
        ctx = self.get_context()
        ctx.insert_at(target_code_block, offset_in_code_block, literal_patch(basic_block_str))
        ctx.apply()
        # 同步引用
        end_offset = self[-1].ir_basic_block[-1].offset + self[-1].ir_basic_block[-1].size
        code_blocks = self.collect_code_blocks(byte_interval, start_offset, end_offset)
        new_blocks = set(byte_interval.blocks) - old_blocks  
        new_code_blocks = [b for b in new_blocks if isinstance(b, gtirb.CodeBlock)]
        code_blocks += list(new_code_blocks)
        basic_block_list = self.group_blocks_ida_style(code_blocks, decoder)
        # 这要求每次插入basic_block产生全局ir范围内唯一的symbol
        symbol = target_code_block.module.symbols_named(i_basic_block.symbol)
        self[-1].symbol.append(symbol)
        self.selfload(basic_block_list, decoder)
        self[-1].mark = True
        return self[-1]
    
    # 重写以增加对实体的操作
    def insert(self, index, i_basic_block, decoder):
        # 在len处插入和append等价
        if index == len(self) or index == -1:
            return self.append(i_basic_block, decoder)
        else:
            # 解位置
            target_code_block, offset_in_code_block = self.index_to_location(index, 0, decoder)
            #target_code_block = self[index].ir_basic_block[0]
            #offset_in_code_block = 0
            # 记录
            start_offset = self[0].ir_basic_block[0].offset
            # 插入实体
            if i_basic_block.symbol:
                basic_block_str = "{}:\n".format(i_basic_block.symbol)
            else:
                basic_block_str = ''
            for i_instruction in i_basic_block.i_instructions:
                basic_block_str += i_instruction.str
                basic_block_str += '\n'

            ctx = self.get_context()
            ctx.insert_at(target_code_block, offset_in_code_block, literal_patch(basic_block_str))
            ctx.apply()
            # 同步引用
            end_offset = self[-1].ir_basic_block[-1].offset + self[-1].ir_basic_block[-1].size
            byte_interval = target_code_block.byte_interval
            code_blocks = self.collect_code_blocks(byte_interval, start_offset, end_offset)
            basic_block_list = self.group_blocks_ida_style(code_blocks, decoder)
            self.selfload(basic_block_list, decoder)
            # 这要求每次插入basic_block产生全局ir范围内唯一的symbol
            symbol = target_code_block.module.symbols_named(i_basic_block.symbol)
            self[index].symbol.append(symbol)
            self[index].mark = True
            return self[index]
    
    # 不产生新块的指令列表插入
    def insert_instruction_list(self, basic_block_index, instruction_index, i_instruction_list, decoder):
        target_basic_block = self[basic_block_index]
        
        # 倒序插入
        for ii_idx in range(len(i_instruction_list)-1, -1, -1):
            target_basic_block.insert(instruction_index, i_instruction_list[ii_idx], decoder)
    
    # 多块尾插
    def append_group(self, i_group, decoder, with_selfload=True):
        # 解位置
        target_code_block = self[-1].ir_basic_block[-1]
        offset_in_code_block = target_code_block.size
        # 记录
        start_offset = self[0].ir_basic_block[0].offset
        byte_interval = target_code_block.byte_interval
        old_blocks = set(byte_interval.blocks)
        # 插入实体
        basic_block_str = ''
        for i_basic_block in i_group.i_basic_blocks:        # 不加jmp是因为默认最后一个块以ret结尾，不需要跳转就可以独立出新的块
            basic_block_str += "{}:\n".format(i_basic_block.symbol)
            for i_instruction in i_basic_block.i_instructions:
                basic_block_str += i_instruction.str
                basic_block_str += '\n'

        ctx = self.get_context()
        ctx.insert_at(target_code_block, offset_in_code_block, literal_patch(basic_block_str))
        ctx.apply()
        # 同步引用
        if with_selfload:
            end_offset = self[-1].ir_basic_block[-1].offset + self[-1].ir_basic_block[-1].size
            code_blocks = self.collect_code_blocks(byte_interval, start_offset, end_offset)
            new_blocks = set(byte_interval.blocks) - old_blocks  
            new_code_blocks = [b for b in new_blocks if isinstance(b, gtirb.CodeBlock)]
            code_blocks += list(new_code_blocks)
            # self.show_code_blocks(code_blocks, decoder)
            basic_block_list = self.group_blocks_ida_style(code_blocks, decoder)
            self.selfload(basic_block_list, decoder)
        # 这要求每次插入basic_block产生全局ir范围内唯一的symbol
        #symbol = target_code_block.module.symbols_named(i_basic_block.symbol)
        #self[-1].symbol.append(symbol)
        #self[-1].mark = True
        #return self[-1]
    
    # 这个函数将位置（basic_block_index，instruction_index的前面）解析为target_instruction的所在code_block，当instruction在code_block的开头时，返回该code_block的开头
    def index_to_location_local(self, basic_block_index, instruction_index, decoder):
        target_basic_block = self[basic_block_index]
        current_index = 0
        # 遍历逻辑基本块中的所有物理块
        for code_block in target_basic_block.ir_basic_block:
            # 反汇编当前块的所有指令
            offset_in_block = 0
            for instruction in decoder.get_instructions(code_block):
                # offset_in_block += instruction.size
                if current_index == instruction_index:
                    # 找到目标指令
                    return code_block, offset_in_block, instruction
                current_index += 1
                offset_in_block += instruction.size
        raise IndexError(f"Instruction index {instruction_index} out of range (total instructions: {current_index})")
    
    # 这个函数将位置（basic_block_index，instruction_index的前面）转换解析为target_instruction的后面，当instruction在code_block的开头时，返回前一个code_block的结尾
    def index_to_location_pre(self, basic_block_index, instruction_index, decoder):
        if basic_block_index == 0 and instruction_index == 0:
            target_basic_block = self[basic_block_index]
            target_code_block = target_basic_block.ir_basic_block[0]
            offset_in_code_block = 0
            target_instruction = None
            # target_code_block, offset_in_code_block, _ = target_basic_block.index_to_location(instruction_index, decoder)
            
        elif basic_block_index > 0 and instruction_index == 0:
            target_basic_block = self[basic_block_index - 1]
            target_code_block = target_basic_block.ir_basic_block[-1]
            offset_in_code_block = target_code_block.size
            target_instruction = list(decoder.get_instructions(target_code_block))[-1]
        else:
            target_basic_block = self[basic_block_index]
            current_index = 0
            # 遍历逻辑基本块中的所有物理块
            for code_block_idx, code_block in enumerate(target_basic_block.ir_basic_block):
                # 反汇编当前块的所有指令
                offset_in_block = 0
                for instruction_idx_in_code_block, instruction in enumerate(decoder.get_instructions(code_block)):
                    if current_index == instruction_index:
                        if instruction_idx_in_code_block == 0:
                            target_code_block = target_basic_block.ir_basic_block[code_block_idx - 1]
                            offset_in_code_block = target_code_block.size
                            target_instruction = list(decoder.get_instructions(target_code_block))[-1]
                        else:
                            target_code_block = code_block
                            offset_in_code_block = offset_in_block
                            target_instruction = list(decoder.get_instructions(target_code_block))[instruction_idx_in_code_block - 1]
                        return target_code_block, offset_in_code_block, target_instruction
                    current_index += 1
                    offset_in_block += instruction.size
            raise IndexError(f"Instruction index {instruction_index} out of range (total instructions: {current_index})")
        return target_code_block, offset_in_code_block, target_instruction
    
    def insert_group_at_head_preparation(self, decoder):
        # 解位置
        # target_basic_block = self[basic_block_index]
        target_code_block, offset_in_code_block, target_instruction = self.index_to_location_pre(0, 0, decoder)
        #target_code_block = self[basic_block_index].ir_basic_block[0]
        #offset_in_code_block = 0
        # 记录
        start_offset = self[0].ir_basic_block[0].offset
        byte_interval = target_code_block.byte_interval

        basic_block_str = f'''
            jmp {self.name}
        '''
        ctx = self.get_context()
        ctx.insert_at(target_code_block, offset_in_code_block, literal_patch(basic_block_str))
        ctx.apply()
        # 这个行为会导致原始的target_code_block被jmp temp_label占据，其原内容被移动到下一个新建的code_block
        # 此时需要将属于target_code_block的symbol归还给新建的code_block
        # 找到新建的code_block
        new_code_block = self._find_code_block_at_offset(target_code_block.byte_interval, target_code_block.offset +target_code_block.size)
        # 找到target_code_block的symbol和函数起始symbol
        target_code_block_symbol = []
        function_start_symbol = None
        for symbol in target_code_block.references:
            if symbol.name.startswith(".L"):
                target_code_block_symbol.append(symbol)
            else:
                function_start_symbol = symbol
        # 处理target_code_block_symbol可能为空的情况
        if len(target_code_block_symbol) == 0:
            target_code_block_symbol.append(gtirb.Symbol(name=".L_old_start_{}".format(target_code_block.address)))
        # 将symbol归还给new_code_block
        for symbol in target_code_block_symbol:
            symbol.referent = new_code_block
            new_code_block.byte_interval.module.symbols.add(symbol)
        return byte_interval, start_offset, target_code_block, function_start_symbol, target_code_block_symbol
    
    def insert_group_at_head(self, target_code_block, function_start_symbol, byte_interval, start_offset, i_group, decoder, with_selfload=True, over=True):
        ctx = self.get_context()
        basic_block_str = ''
        for i_basic_block in i_group.i_basic_blocks:
            # 头插的块必须有symbol
            basic_block_str += "{}:\n".format(i_basic_block.symbol)
            for i_instruction in i_basic_block.i_instructions:
                basic_block_str += i_instruction.str
                basic_block_str += '\n'
        ctx.insert_at(target_code_block, target_code_block.size, literal_patch(basic_block_str))
        ctx.apply()
        # 找到插入后新的函数起始code_block
        new_function_start_block = self._find_code_block_at_offset(target_code_block.byte_interval, target_code_block.offset + target_code_block.size)
        # 将函数起始symbol归还给new_function_start_block
        function_start_symbol.referent = new_function_start_block       # 这个操作支持多次刷新
        new_function_start_block.byte_interval.module.symbols.add(function_start_symbol)
        if over:
            # 将临时的jmp指令删除
            ctx = self.get_context()
            ctx.delete_at(target_code_block, 0, target_code_block.size)
            ctx.apply()

        # 同步引用
        if with_selfload:
            end_offset = self[-1].ir_basic_block[-1].offset + self[-1].ir_basic_block[-1].size
            code_blocks = self.collect_code_blocks(byte_interval, start_offset, end_offset)
            # self.show_code_blocks(code_blocks, decoder)
            basic_block_list = self.group_blocks_ida_style(code_blocks, decoder)
            self.selfload(basic_block_list, decoder)
    
    # 多块前向插入（以xx之后的方式定位，实现在basic_block_index, instruction_index之前插入的效果）
    def insert_group(self, basic_block_index, instruction_index, i_group, decoder, mode='pre', with_selfload=True):
        # 解位置
        # target_basic_block = self[basic_block_index]
        # 这两种模式返回的target_instruction是同一个，但是target_code_block是不同的，结果都是在target_instruction后插入，但是local是在target_instruction所在的code_block，而pre是在上一个target_code_block结尾
        if mode == 'local':
            target_code_block, offset_in_code_block, target_instruction = self.index_to_location_local(basic_block_index, instruction_index, decoder)
        else:
            target_code_block, offset_in_code_block, target_instruction = self.index_to_location_pre(basic_block_index, instruction_index, decoder)
        #target_code_block = self[basic_block_index].ir_basic_block[0]
        #offset_in_code_block = 0
        # 记录
        start_offset = self[0].ir_basic_block[0].offset
        byte_interval = target_code_block.byte_interval
        # 插入实体
        # 处理在函数开头插入的情况（此时没有上一个block可以用于参照）
        # 创造一个临时的jmp指令划分出一个块作为参照
        if basic_block_index == 0 and instruction_index == 0 and mode == 'pre':
            raise Exception("Cannot insert at the very beginning of the function directly. Please use insert_group_at_head_preparation and insert_group_at_head instead.")
        else:
            basic_block_str = ''
            for i_basic_block in i_group.i_basic_blocks:
                # 在插入junk_code时没有symbol，所以不会触发
                if i_basic_block.symbol:
                    basic_block_str += "{}:\n".format(i_basic_block.symbol)
                for i_instruction in i_basic_block.i_instructions:
                    basic_block_str += i_instruction.str
                    basic_block_str += '\n'
            ctx = self.get_context()
            ctx.insert_at(target_code_block, offset_in_code_block, literal_patch(basic_block_str))
            ctx.apply()
        # 同步引用
        if with_selfload:
            end_offset = self[-1].ir_basic_block[-1].offset + self[-1].ir_basic_block[-1].size
            code_blocks = self.collect_code_blocks(byte_interval, start_offset, end_offset)
            # self.show_code_blocks(code_blocks, decoder)
            basic_block_list = self.group_blocks_ida_style(code_blocks, decoder)
            self.selfload(basic_block_list, decoder)
        # 这要求每次插入basic_block产生全局ir范围内唯一的symbol
        #symbol = target_code_block.module.symbols_named(i_basic_block.symbol)
        #self[basic_block_index].symbol.append(symbol)
        #self[basic_block_index].mark = True
        #return self[basic_block_index]
    
    def get_entry_adress(self):
        return hex(self[0].ir_basic_block[0].address)
    
    # 识别一条边的跳转种类
    def edge_type(self, source_basic_block_index, target_basic_block_index, decoder):
        # 解析基本块
        source_basic_block = self[source_basic_block_index]
        target_basic_block = self[target_basic_block_index]
        
        type = None
        conditional = False
        direct = True
        # 起始基本块的最后一条指令是jmp，则该边为跳转边
        if source_basic_block[-1].jump_type(decoder) == Instruction.JMP:
            type = gtirb.EdgeType.Branch
        # 起始基本块的最后一条指令是cjmp，且目标基本块不是起始基本块的下一个基本块，则该边为跳转边
        elif source_basic_block[-1].jump_type(decoder) == Instruction.CJMP:
            conditional = True
            if target_basic_block.idx == source_basic_block.idx + 1:
                type = gtirb.EdgeType.Fallthrough       # 顺序边（F）
            else:
                type = gtirb.EdgeType.Branch            # 跳转边（T）
        # 其他情况一律为顺序边
        # Function这一层只有5个类型的边：顺序边、jmp跳转边、cjmp跳转边、cjmp顺序边以及call的顺序边
        else:
            type = gtirb.EdgeType.Fallthrough
        label = gtirb.EdgeLabel(type, conditional, direct)
        return label
    
    def _find_jmp_in_code_block(self, code_block, decoder):
        find_flag_at_end = False
        jmp_str = None
        instructions_capstone = list(decoder.get_instructions(code_block))
        jmp_instruction_offset = 0
        jmp_instruction_len = 0
        max_code_block_len = len(instructions_capstone)
        for idx, instruction_capstone in enumerate(instructions_capstone):
            jmp_instruction_len = instruction_capstone.size
            if instruction_capstone.group(CS_GRP_JUMP) and idx == max_code_block_len - 1:
                jmp_str = instruction_capstone.mnemonic
                find_flag_at_end = True
                break
            jmp_instruction_offset += instruction_capstone.size
        return find_flag_at_end, jmp_instruction_offset, jmp_instruction_len, jmp_str
    
    def clear(self):
        # 清空实体
        ctx = self.get_context()
        ctx.delete_function(self.function)
        # 清空引用
        super().clear()
        # 标记
        self.name = None

        # 实体
        self.function = None
    
    # 显示全部
    def show(self, decoder):
        print('Function: {}'.format(self.name))
        for basic_block in self:
            print('\tBlock_{}:'.format(basic_block.idx))
            basic_block.show(decoder)
        print('\n')
    
    def collect_code_blocks(self, byte_interval, start_offset, end_offset):
        code_blocks = []  
        for block in sorted(byte_interval.blocks, key=lambda b: b.offset):  
            if isinstance(block, gtirb.CodeBlock):  
                if start_offset <= block.offset < end_offset:
                    code_blocks.append(block)
        return code_blocks
    
    def show_code_blocks(self, code_blocks, decoder):
        function_name = self.function.get_name()
        sorted_code_blocks = sorted(code_blocks, key=lambda b: b.address if b.address else 0)
        print("Function: {}".format(function_name))
        
        for i, code_block in enumerate(sorted_code_blocks):
            # offset = 0
            print('symbol: {}'.format(list(code_block.references)))
            print('code_block {}: {}'.format(i, code_block))
            instructions_capstone = decoder.get_instructions(code_block)
            for instruction_capstone in instructions_capstone:
                print("\t[{}]: {} {}".format(hex(instruction_capstone.address), instruction_capstone.mnemonic, instruction_capstone.op_str))
            print('\n')
    
    # 在ir中查看
    def show_in_ir(self):           # TODO get_all_blocks不总是准确
        function_name = self.function.get_name()
        code_blocks = self.function.get_all_blocks()

        module = list(code_blocks)[0].ir.modules[0]
        decoder = GtirbInstructionDecoder(module.isa)

        sorted_code_blocks = sorted(code_blocks, key=lambda b: b.address if b.address else 0)
        print("Function: {}".format(function_name))
        
        for i, code_block in enumerate(sorted_code_blocks):
            # offset = 0
            print('symbol: {}'.format(list(code_block.references)))
            print('code_block {}: {}'.format(i, code_block))
            instructions_capstone = decoder.get_instructions(code_block)
            for instruction_capstone in instructions_capstone:
                print("\t[{}]: {} {}".format(hex(instruction_capstone.address), instruction_capstone.mnemonic, instruction_capstone.op_str))
            print('\n')
    
    # 整理为纯str
    def str(self, decoder):
        # 转化为字符串
        function_str = ""
        for basic_block in self:
            function_str += basic_block.str(decoder)
            function_str += '\n'
        return function_str

    # 整理为mix_embedder输入格式
    def to_mix_embedder_input(self, decoder):
        function_inst_str_list = []
        for basic_block in self:
            block_inst_str_list = []
            for instruction in basic_block:
                instruction_str = instruction.str(decoder)
                block_inst_str_list.append(instruction_str)
            function_inst_str_list.append(block_inst_str_list)
        return function_inst_str_list

    # 轻量级的影子函数
    def to_ShadowFunction(self, decoder):
        shadow_function = ShadowFunction(self.name)
        shadow_basic_block_list = []
        for basic_block in self:
            shadow_basic_block = ShadowBasicBlock(basic_block.idx)
            shadow_instruction_list = []
            basic_block_str_normalization = basic_block.normalization(decoder)
            for idx, instruction_str_normalization in enumerate(basic_block_str_normalization):
                shadow_instruction_list.append(ShadowInstruction(instruction_str_normalization, idx))
            shadow_basic_block.load(shadow_instruction_list)
            shadow_basic_block_list.append(shadow_basic_block)
        shadow_function.load(shadow_basic_block_list)
        
        # 同步后继关系
        for basic_block, shadow_basic_block in zip(self, shadow_function):
            for successor in basic_block.successors:
                shadow_basic_block.successors.append(shadow_function[successor.idx])
        
        return shadow_function
    
    def inst_count(self):
        """
        统计当前函数的指令数量
        """
        count = 0
        for basic_block in self:
            count += len(basic_block)
        return count
    
    def get_opcode_freq(self, decoder):
        """
        统计当前函数的操作码频率分布
        """
        import pandas as pd
        opcodes = []
        for basic_block in self:
            instructions_capstone = basic_block.disassemble(decoder)
            for inst in instructions_capstone:
                opcodes.append(inst.mnemonic.lower()) # 统一转小写
        
        if not opcodes:
            return pd.Series(dtype=float)
        return pd.Series(opcodes).value_counts(normalize=True)

    def calculate_kl_divergence(self, original_dist, decoder):
        """
        计算当前实体函数相对于原始分布的 KL 散度
        """
        from scipy.stats import entropy
        current_dist = self.get_opcode_freq(decoder)
        
        if current_dist.empty or original_dist.empty:
            return 0.0
            
        all_indices = original_dist.index.union(current_dist.index)
        p = original_dist.reindex(all_indices, fill_value=1e-6).values
        q = current_dist.reindex(all_indices, fill_value=1e-6).values
        
        return float(entropy(p, q, base=2))
    
    def __hash__(self):
        return hash(id(self))
    
    def __eq__(self, other):
        if not isinstance(other, Function):
            return False
        return id(self) == id(other)
    
    def __repr__(self):
        return f"Function(idx={self.idx}, name={self.name})"

# ============================================ 影子模式 ===================================================
class ShadowInstruction:
    def __init__(self, asm_str, idx):
        self.str = asm_str
        
        self.idx = idx
    
    def get_mnemonic(self):
        return self.str.split()[0].lower()
    
    def _is_jump(self):
        mnemonic = self.str.split()[0].lower()
        return mnemonic.startswith('j') or mnemonic == 'loop'
    
    def _is_indirect_jump(self):
        if self._is_jump():
            # 获取指令中操作数的部分（去掉助记符后的剩余字符串）
            parts = self.str.split(None, 1)
            if len(parts) < 2:
                return False
            
            operands = parts[1].lower()
            
            # 间接跳转的典型特征：
            # 1. 操作数包含方括号 (内存间接寻址，如 jmp [eax])
            # 2. 操作数不包含指令位置标签（通常是 0x 开头的地址或特定的 Label）
            # 这里提供一种基于操作数格式的通用判断逻辑：
            return '[' in operands or operands.startswith('qword ptr') or operands.startswith('dword ptr') or self._is_register(operands)
        else:
            return False

    def _is_register(self, operand):
        # 常见的 x86/x64 寄存器列表（可根据需求扩充）
        registers = {'eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'ebp', 'esp', 
                     'rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'rbp', 'rsp',
                     'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15'}
        # 去除前缀和多余空格后判断
        clean_op = operand.strip().split()[-1] 
        return clean_op in registers
    
    def _is_call(self):
        return self.str.lower().startswith('call')

    def get_asm2vec_token(self):
        """
        asm2vec 规则更新：
        1. 减号 '-' 替换为加号 '+'
        2. 偏移/立即数/调用目标 -> CONST
        3. 跳转目标 -> LABEL + 索引
        4. 前置空格缩进
        """
        raw_str = self.str.strip()
        parts = raw_str.split(maxsplit=1)
        mnemonic = parts[0]
        
        if len(parts) < 2:
            return f" {mnemonic}"
        
        # 核心修改：先将减号替换为加号
        operands = parts[1].replace('-', '+')
        
        if self._is_jump():
            block_match = re.search(r'Block_(\d+)', operands)
            target = f"LABEL{block_match.group(1)}{block_match.group(1)}" if block_match else "LABEL_UNKNOWN"
            return f" {mnemonic} {target}"
        
        if self._is_call():
            return f" {mnemonic} CONST"
        
        # 将所有数值或已有的 OFFSET 标记替换为 CONST
        norm_operands = re.sub(r'\b(OFFSET|0x[0-9a-fA-F]+|\d+)\b', 'CONST', operands)
        return f" {mnemonic} {norm_operands}"

    def get_safe_token(self):
        """
        SAFE 规则：
        - 格式：X_Mnemonic_Operand1,Operand2
        - 所有的数值/偏移/FUNCTION 统一替换为 HIMM
        - 指令末尾不带下划线
        """
        # 移除逗号后的空格以便统一处理操作数
        raw_str = self.str.strip()
        clean_str = raw_str.replace(', ', ',')
        parts = clean_str.split(maxsplit=1)
        mnemonic = parts[0].strip('_') # 防止 endbr64_ 这种情况
        
        if len(parts) < 2:
            return f"X_{mnemonic}"
            
        operands = parts[1]
        
        # 跳转目标处理：Block_5 -> Block_5
        if self._is_jump():
            # SAFE 的跳转目标通常保留 Block 标识
            pass 
        else:
            # 将所有立即数、偏移量标识、函数标识统一换成 HIMM
            operands = re.sub(r'\b(OFFSET|FUNCTION|0x[0-9a-fA-F]+|\d+)\b', 'HIMM', operands)
        
        # 移除空格以符合 X_mnemonic_op1,op2 格式
        operands = operands.replace(' ', '')
        return f"X_{mnemonic}_{operands}"

    def get_CLAP_token(self, addr_map=None):
        """
        addr_map: 一个字典，映射 {旧的跳转目标: 新的全局INSTR索引}
        """
        raw_str = self.str.strip()
        
        # 1. 处理跳转
        if self._is_jump():
            mnemonic = raw_str.split()[0]
            # 匹配目标，如 "Block_5" 或 "LABEL5"
            match = re.search(r'(?:Block_|LABEL)(\d+)', raw_str)
            if match and addr_map:
                old_block_idx = int(match.group(1))
                # 从映射表中找到该块第一条指令对应的新全局索引
                new_idx = addr_map.get(old_block_idx, "UNKNOWN")
                return f"{mnemonic:<7} short INSTR{new_idx}"
            return raw_str # 降级处理
            
        # 2. 处理调用（保持原本的 FUNCTION 标记或还原）
        if self._is_call():
            mnemonic = raw_str.split()[0]
            return f"{mnemonic:<7} set_program_name" if "FUNCTION" in raw_str else raw_str

        # 3. 普通指令格式化
        parts = raw_str.split(maxsplit=1)
        if len(parts) == 2:
            return f"{parts[0]:<7} {parts[1]}"
        return raw_str

class ShadowBasicBlock(list):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx
        self.successors = []
    
    def load(self, shadow_instruction_list):
        for shadow_instruction in shadow_instruction_list:
            super().append(shadow_instruction)
    
    def update_instructions_idx(self):
        for idx, instruction in enumerate(self):
            instruction.idx = idx
    
    def insert(self, index, shadow_instruction):
        super().insert(index, shadow_instruction)
        self.update_instructions_idx()
    
    def append(self, shadow_instruction):
        super().append(shadow_instruction)
        self.update_instructions_idx()
    
    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        # 必须使用 id 比较或 identity 比较
        return self is other

class ShadowEdge(NamedTuple):
    source: 'ShadowBasicBlock'
    target: 'ShadowBasicBlock'
    type: int
    
    class Type:
        Branch = 0         # 跳转边
        # Call = 1         # 调用边
        Fallthrough = 2    # 直接边

    def __repr__(self):
        type_str = "Branch" if self.type == 0 else "Fallthrough"
        return f"ShadowEdge(source={self.source.idx}, target={self.target.idx}, type={type_str})"

class ShadowCFG:
    True_color = 'forestgreen'   # Branch (T)
    False_color = 'tomato'       # Fallthrough (F)
    Normal_color = 'royalblue'   # 唯一后继边

    def __init__(self, edges=None):
        self._nxg = nx.DiGraph()
        if edges:
            for edge in edges:
                self.add_edge(edge)
    
    def nx(self):
        return self._nxg

    def add_edge(self, edge: ShadowEdge):
        self._nxg.add_edge(
            edge.source, 
            edge.target, 
            type=edge.type
        )
    
    def nodes(self):
        return self._nxg.nodes()

    def edges(self):
        for s, t, d in self._nxg.edges(data=True):
            yield ShadowEdge(s, t, d['type'])

    def out_edges(self, node: ShadowBasicBlock):
        """
        获取指定节点的出边列表
        """
        if node in self._nxg:
            # data=True 会返回存储在边上的 'type' 属性
            for s, t, data in self._nxg.out_edges(node, data=True):
                yield ShadowEdge(source=s, target=t, type=data['type'])

    def in_edges(self, node: ShadowBasicBlock):
        """
        获取指向指定节点的入边列表
        """
        if node in self._nxg:
            for s, t, data in self._nxg.in_edges(node, data=True):
                yield ShadowEdge(source=s, target=t, type=data['type'])
    
    def draw(self, png_name):
        """
        使用 PyGraphviz 绘制影子函数控制流图
        """
        if not self._nxg.nodes():
            return

        # 创建 PyGraphviz 图
        G = pgv.AGraph(strict=True, directed=True)
        
        # 1. 配置节点
        for node in self._nxg.nodes():
            # node 是 ShadowBasicBlock 实例
            # 构建 HTML 形式的 Label
            label = f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">'
            label += f'<tr><td align="left"><b>Block_{node.idx}</b></td></tr>'
            
            for inst in node:
                # 转义 HTML 特殊字符并换行显示指令
                inst_str = inst.str.replace('<', '&lt;').replace('>', '&gt;')
                label += f'<tr><td align="left">{inst_str}</td></tr>'
            
            label += f'</TABLE>>'
            
            G.add_node(node.idx, label=label, shape='box', style='filled', 
                       fillcolor='aquamarine', color='darkcyan', fontname='Courier')

        # 2. 配置边
        for u, v, data in self._nxg.edges(data=True):
            e_type = data['type']
            
            # 判定颜色：如果是条件跳转（有两个后继），渲染红绿边；否则渲染蓝色
            color = self.Normal_color
            edge_label = ''
            
            # 获取源块的后继数量来判断是否是条件分支
            if len(u.successors) == 2:
                if e_type == 0: # Branch
                    color = self.True_color
                    edge_label = 'T'
                else: # Fallthrough
                    color = self.False_color
                    edge_label = 'F'
            
            G.add_edge(u.idx, v.idx, color=color, label=edge_label, 
                       fontcolor=color, penwidth=1.5)

        # 布局并保存
        G.layout(prog='dot')
        G.draw(png_name)
        return G
    
    def __len__(self):
        return len(self._nxg.edges())

    def __repr__(self):
        return f"ShadowCFG(nodes={len(self._nxg.nodes)}, edges={len(self._nxg.edges)})"

class ShadowFunction(list):
    def __init__(self, name):
        super().__init__()
        self.name = name
    
    def load(self, shadow_basic_block_list):
        for shadow_basic_block in shadow_basic_block_list:
            super().append(shadow_basic_block)
    
    def update_basic_blocks_idx(self):
        """插入块后必须调用，用于刷新索引顺序"""
        for idx, block in enumerate(self):
            block.idx = idx
    
    def insert(self, index, shadow_basic_block):
        super().insert(index, shadow_basic_block)
        self.update_basic_blocks_idx()
    
    def append(self, shadow_basic_block):
        super().append(shadow_basic_block)
        self.update_basic_blocks_idx()
    
    def build_edge(self, source_idx, target_idx):
        if source_idx >= len(self) or target_idx >= len(self):
            return
        
        source_bb = self[source_idx]
        target_bb = self[target_idx]
        
        if target_bb in source_bb.successors:
            return
        
        assert len(source_bb.successors) < 2
        source_bb.successors.append(target_bb)

    def remove_edge(self, source_idx, target_idx):
        if source_idx >= len(self) and target_idx >= len(self):
            return
        source_bb = self[source_idx]
        target_bb = self[target_idx]
        # 过滤掉所有匹配的目标块对象引用
        source_bb.successors = [s for s in source_bb.successors if s != target_bb]
    
    # 更新跳转指令
    def update_jump(self):
        for shadow_basic_block in self:
            for shadow_instruction in shadow_basic_block:
                if shadow_instruction._is_jump():
                    jump_target = '?'
                    if len(shadow_basic_block.successors) == 2:
                        for successor in shadow_basic_block.successors:
                            if successor.idx != shadow_basic_block.idx + 1:
                                jump_target = successor.idx
                    elif len(shadow_basic_block.successors) == 1:
                        jump_target = shadow_basic_block.successors[0].idx
                    instruction_str = shadow_instruction.get_mnemonic() + f' Block_{jump_target}'
                    shadow_instruction.str = instruction_str
    
    def normalization(self, decoder):
        instructions_capstone = self.disassemble(decoder)
        instruction_str_normalization = []
        for instruction_capstone in instructions_capstone:
            # 处理跳转指令
            if instruction_capstone.group(CS_GRP_JUMP):
                jump_target = '?'
                if len(self.successors) == 2:
                    for successor in self.successors:
                        if successor.idx != self.idx + 1:
                            jump_target = successor.idx
                elif len(self.successors) == 1:
                    jump_target = self.successors[0].idx
                instruction_str = instruction_capstone.mnemonic + f' Block_{jump_target}'
            else:
                instruction_str = instruction_capstone.mnemonic + ' ' + instruction_capstone.op_str
            instruction_str_normalization.append(BPE_utils.normalization(instruction_str))
            
        return instruction_str_normalization
    
    def edge_type(self, source_idx: int, target_idx: int) -> int:
        """
        判断边类型：
        0: Branch (跳转边)
        2: Fallthrough (顺序边)
        """
        source_bb = self[source_idx]
        
        # 获取源基本块最后一条指令
        if not source_bb:
            return ShadowEdge.Type.Fallthrough # 默认顺序
            
        last_inst = source_bb[-1]
        mnemonic = last_inst.get_mnemonic()
        
        # 1. 如果是无条件跳转
        if mnemonic == 'jmp':
            return ShadowEdge.Type.Branch
            
        # 2. 如果是条件跳转
        if last_inst._is_jump(): # 指以 j 开头的指令
            # 如果跳转目标不是紧邻的下一个索引，则是 Branch (T)
            # 如果是紧邻的下一个索引，通常对应 CJMP 的 False 分支，设为 Fallthrough
            if target_idx == source_idx + 1:
                return ShadowEdge.Type.Fallthrough
            else:
                return ShadowEdge.Type.Branch
                
        # 3. 普通指令流
        return ShadowEdge.Type.Fallthrough

    def CFG(self) -> ShadowCFG:
        """
        构建并返回该影子函数的 ShadowCFG
        """
        cfg = ShadowCFG()
        
        # 如果只有一个块且没有后继
        if len(self) == 1 and not self[0].successors:
            cfg._nxg.add_node(self[0])
            return cfg

        for src_block in self:
            for tgt_block in src_block.successors:
                e_type = self.edge_type(src_block.idx, tgt_block.idx)
                edge = ShadowEdge(source=src_block, target=tgt_block, type=e_type)
                cfg.add_edge(edge)
                
        return cfg

    def str(self):
        # 转化为字符串
        shadow_function_str = ""
        for shadow_basic_block in self:
            for shadow_instruction in shadow_basic_block:
                shadow_function_str += shadow_instruction.str
                shadow_function_str += '\n'
            shadow_function_str += '\n'
        return shadow_function_str
    
    def to_expert_input(self):
        # 转化为字符串
        shadow_function_str = ""
        for block_idx, shadow_basic_block in enumerate(self):
            shadow_function_str += f"\n; --- [Block {block_idx}] --- (Max instructions: {len(shadow_basic_block)})\n"
            for inst_idx, shadow_instruction in enumerate(shadow_basic_block):
                shadow_function_str += f"inst_{inst_idx}: {shadow_instruction.str}\n"
            shadow_function_str += '\n'
        return shadow_function_str
    
    # 整理为mix_embedder输入格式
    def to_mix_embedder_input(self):
        shadow_function_inst_str_list = []
        for shadow_basic_block in self:
            shadow_block_inst_str_list = []
            for shadow_instruction in shadow_basic_block:
                shadow_instruction_str = shadow_instruction.str
                shadow_block_inst_str_list.append(shadow_instruction_str)
            shadow_function_inst_str_list.append(shadow_block_inst_str_list)
        return shadow_function_inst_str_list

    def to_shadow_mix_similarity_input(self, model_type):
        if model_type == 'asm2vec':
            return self.to_asm2vec_input()
        elif model_type == 'CLAP':
            return self.to_CLAP_input()
        elif model_type == 'safe':
            return self.to_safe_input()
        else:
            raise ValueError(f'Unsupported model_type {model_type}')

    def to_asm2vec_input(self):
        """
        精确渲染 asm2vec 文本格式：
        1. 预扫描所有跳转指令，记录哪些 Block 的索引是跳转目标。
        2. 仅在被引用的块上方打印 LABELx:
        3. 指令前保持一个空格缩进。
        """
        # --- 步骤 1: 统计哪些块是跳转目标 ---
        referenced_blocks = set()
        for block in self:
            for inst in block:
                if inst._is_jump():
                    # 从 "jne Block_5" 中提取数字 5
                    match = re.search(r'Block_(\d+)', inst.str.strip())
                    if match:
                        referenced_blocks.add(int(match.group(1)))

        # --- 步骤 2: 渲染文本 ---
        output = []
        for block in self:
            # 只有当该块被显式跳转引用时，才打印顶格标签
            if block.idx in referenced_blocks:
                output.append(f"LABEL{block.idx}{block.idx}:")
            
            for inst in block:
                # 渲染指令（内部已处理 jne Block_5 -> jne LABEL5）
                output.append(inst.get_asm2vec_token())
        
        return "\n".join(output)

    def to_safe_input(self):
        """
        返回一维 Token 列表
        """
        safe_tokens = []
        for block in self:
            for inst in block:
                safe_tokens.append(inst.get_safe_token())
        return safe_tokens

    def to_CLAP_input(self):
        """
        将影子系列转化为 CLAP 字典格式
        """
        # 建立 块ID 到 全局指令ID 的映射表
        block_to_global_instr_map = {}
        flat_instructions = []
        
        # 第一遍扫描：计算位置
        current_global_idx = 1
        for block in self:
            # 记录这个 Block 的起点对应的全局行号
            block_to_global_instr_map[block.idx] = current_global_idx
            for inst in block:
                flat_instructions.append(inst)
                current_global_idx += 1
        
        # 第二遍扫描：生成最终字典
        clap_input_dict = {}
        for i, inst in enumerate(flat_instructions):
            # i + 1 是当前的全局行号
            line_key = str(i + 1)
            # 传入映射表，让指令能正确找到跳转目标的新索引
            clap_input_dict[line_key] = inst.get_CLAP_token(addr_map=block_to_global_instr_map)
            
        return clap_input_dict

    def inst_count(self):
        """
        统计当前函数的指令数量
        """
        count = 0
        for basic_block in self:
            count += len(basic_block)
        return count

    def get_opcode_freq(self):
        """
        统计当前影子函数的操作码频率分布
        直接从存储的规范化字符串中提取，速度极快
        """
        import pandas as pd
        opcodes = []
        for block in self:
            for inst in block:
                mnemonic = inst.get_mnemonic()
                opcodes.append(mnemonic)
        
        if not opcodes:
            return pd.Series(dtype=float)
        return pd.Series(opcodes).value_counts(normalize=True)

    def calculate_kl_divergence(self, original_dist):
        """
        计算当前影子函数相对于原始分布的 KL 散度
        """
        from scipy.stats import entropy
        current_dist = self.get_opcode_freq()
        
        if current_dist.empty or original_dist.empty:
            return 0.0
            
        all_indices = original_dist.index.union(current_dist.index)
        p = original_dist.reindex(all_indices, fill_value=1e-6).values
        q = current_dist.reindex(all_indices, fill_value=1e-6).values
        
        return float(entropy(p, q, base=2))
    
    def show(self):
        for shadow_basic_block in self:
            print(f'Block_{shadow_basic_block.idx}:')
            for shadow_instruction in shadow_basic_block:
                print(f'{shadow_instruction.str}')
            
            successors_idx = [successor.idx for successor in shadow_basic_block.successors]
            print(f'Successors: {successors_idx}')
            print('\n')
        
# 控制流表示
class CFR:
    def __init__(self, ir):
        self.ir = ir
        self._raw_functions_dict = {}

        self.selfload()
    
    # 创建指令解码器
    def get_decoder(self):
        decoder = GtirbInstructionDecoder(self.ir.modules[0].isa)
        return decoder
    
    def get_context(self):
        module = self.ir.modules[0]
        functions = gtirb_functions.Function.build_functions(module)
        ctx = gtirb_rewriting.RewritingContext(module, functions)
        return ctx
    
    def selfload(self):
        # 清空当前缓存
        self._raw_functions_dict.clear()
        
        # 加载到symbol名:ir函数的字典模式
        functions = gtirb_functions.Function.build_functions(self.ir.modules[0])
        for function in functions:
            sorted_blocks = sorted(function.get_all_blocks(), key=lambda b: b.address if b.address else 0)
            # 这意味着CFR中可能出现多个symbol名对应同一个函数的情况
            keys = [s.name for s in sorted_blocks[0].references]
            for key in keys:
                self._raw_functions_dict[key] = function
    
    def find_all_function_by_name(self, name):        
        # 处理函数编辑后名称被添加后缀的情况
        true_symbols = []
        for symbol in self._raw_functions_dict.keys():
            if name in symbol:
                true_symbols.append(symbol)
        if not true_symbols:
            raise ValueError(f"Function {name} not found")

        functions = {}
        for true_symbol in true_symbols:
            raw_f = self._raw_functions_dict[true_symbol]
            
            # 只有在需要修改时，才将其加载为可编辑的 Function 对象
            function = Function(self.get_decoder(), raw_f, name, cfr=self)
            functions[true_symbol] = function
        return functions
    
    # 凭函数名称找到函数，必须使用未编辑情况下获得的完整、原始的函数名称
    def find_function_by_name(self, function_name, step_i=None, strict=False):
        # 严格模式
        if strict:
            raw_f = self._raw_functions_dict[function_name]
            function = Function(self.get_decoder(), raw_f, function_name, cfr=self)
            return function
        
        # 处理函数编辑后名称被添加后缀的情况
        # 这里会出现三种情况：
        true_symbol = []
        for symbol in self._raw_functions_dict.keys():    
            if function_name in symbol:
                true_symbol.append(symbol)
        
        raw_f = None
        raw_n = None
        
        # 1. 找不到函数，此时报错
        if len(true_symbol) == 0:
            raise ValueError(f"Function {function_name} not found")
        # 2. 找到唯一函数名的函数，此时symbol长度为1，取出来就是目标
        elif len(true_symbol) == 1:
            raw_f = self._raw_functions_dict[true_symbol[0]]
            raw_n = true_symbol[0]
        # 3. 找到多个相符的函数名
        # 此时可能是因为触发了pprinter的消岐机制
        # 进一步搜索所有code_block的references的name中带有'.entry_'+str(step_i)、'.splitting_'+str(step_i)的函数
        else:
            for ts in true_symbol:
                rf = self._raw_functions_dict[ts]
                code_blocks = rf.get_all_blocks()
                for code_block in code_blocks:
                    symbols = [s.name for s in code_block.references]
                    if '.entry_'+str(step_i) in symbols or '.splitting_'+str(step_i) in symbols:
                        raw_f = rf
                        raw_n = ts
        if not raw_f:
            raise ValueError(f"Found several functions, but no one have .entry_{str(step_i)} or .splitting_{str(step_i)}")
        # 只有在需要修改时，才将其加载为可编辑的 Function 对象
        function = Function(self.get_decoder(), raw_f, raw_n, cfr=self)
        return function
    
    # 凭函数地址找到函数，function_address形式为'0x....'的字符串
    def find_function_by_address(self, function_address):
        for symbol in self._raw_functions_dict.keys(): 
            raw_f = self._raw_functions_dict[symbol]
            raw_n = symbol
            sorted_blocks = sorted(raw_f.get_all_blocks(), key=lambda b: b.address if b.address else 0)
            if function_address == hex(sorted_blocks[0].address):
                function = Function(self.get_decoder(), raw_f, raw_n, cfr=self)
                return function
        raise ValueError(f"Address {function_address} not found")
    
    def new_function(self, i_function):
        ctx = self.get_context()
        function_str = ""
        # 插入实体
        for i_basic_block in i_function.i_basic_blocks:
            basic_block_str = "{}:\n".format(i_basic_block.symbol)
            for i_instruction in i_basic_block.i_instructions:
                basic_block_str += i_instruction.str
                basic_block_str += '\n'
            function_str += basic_block_str
        new_function_symbol = ctx.register_insert_function(i_function.label, literal_patch(function_str))     # 新函数名，函数指令串
        ctx.apply()

        # 更新引用
        self.selfload()
        return new_function_symbol
    
    def write(self, gtirb_file_name):
        self.ir.save_protobuf(gtirb_file_name)
    
    # 在IR层显示
    def show_ir(self):
        for module in self.ir.modules:
            print(module.name)      # 没有地址
            for section in module.sections:
                print('\t[{}] {}:'.format(section.address, section.name))
                print('\tsection size: {}'.format(section.size))
                for i, byte_interval in enumerate(section.byte_intervals):
                    print("\t\t[{}] byte_interval {}:".format(byte_interval.address, i))
                    print('\t\t\t{}'.format(byte_interval.contents))
                    for block in byte_interval.blocks:
                        print('\t\t\t[{}] {}'.format(block.address, block))
                        print('\t\t\tblock size: {} offset: {}'.format(block.size, block.offset))
                        for symbol in block.references:
                            print('\t\t\t\t{}'.format(symbol))
                    for key in byte_interval.symbolic_expressions.keys():
                        print('\t\t\t{}: {}'.format(key, byte_interval.symbolic_expressions[key]))
    
    # 在CFR层显示
    def show(self):
        decoder = self.get_decoder()
        for function in self:
            function.show(decoder)
    
    def __deepcopy__(self, memo):
        # Deepcopy the list part of the object using super().__deepcopy__
        cls = self.__class__
        ir_cp = deepcopy(self.ir)
        result = cls(ir_cp)
        memo[id(self)] = result
        return result
    
    def __repr__(self):
        return f"CFR Object at {hex(id(self))}"