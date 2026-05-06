import re
import random
import itertools
import torch
import json
import numpy as np

from scipy.spatial.distance import cdist
from typing import List, Any
from collections import Counter

from envs.binary_process_editor import BPE_CFR
from envs.binary_process_editor import BPE_utils

class RepetitionAnalyzer:
    def __init__(self):
        """
        初始化汇编代码分析器
        """
        pass
    
    @staticmethod
    def _normalize_assembly(asm_code: str) -> str:
        """
        标准化汇编代码：统一格式，移除不相关信息
        
        处理：
        1. 移除注释
        2. 移除地址信息
        3. 标准化大小写
        4. 处理标号
        """
        lines = asm_code.strip().split('\n')
        normalized_lines = []
        
        for line in lines:
            # 移除注释（; 或 # 开头的内容）
            line = re.sub(r'[;#].*$', '', line).strip()
            if not line:
                continue
                
            # 移除行号标签（如 .L1: 或 label:）
            line = re.sub(r'^\s*\.?[A-Za-z_][A-Za-z0-9_]*\s*:', '', line)
            
            # 移除内存地址中的偏移（保持寄存器）
            # 保留 push ebp, mov ebp, esp 这样的指令
            line = line.upper()  # 统一大写
            
            normalized_lines.append(line)
            
        return '\n'.join(normalized_lines)
    
    @staticmethod
    def _tokenize_assembly(asm_code: str) -> List[str]:
        """
        汇编代码分词
        
        策略：
        1. 分割为操作码和操作数
        2. 单独处理寄存器、立即数
        3. 保持内存访问格式
        """
        tokens = []
        lines = asm_code.split('\n')
        
        for line in lines:
            if not line.strip():
                continue
                
            # 分割指令和操作数
            parts = line.strip().split(None, 1)
            
            # 操作码总是作为一个token
            if parts:
                tokens.append(parts[0])
                
            # 处理操作数
            if len(parts) > 1:
                operands = parts[1]
                
                # 分割多个操作数（逗号分隔）
                for operand in re.split(r'\s*,\s*', operands):
                    operand = operand.strip()
                    if not operand:
                        continue
                    
                    # 进一步分解复杂操作数
                    # 例如：[EBP+8] 分解为 [ EBP + 8 ]
                    operand_parts = re.findall(r'\[|\]|[A-Z][A-Z0-9]+|\d+|[\+\-*/]', operand)
                    tokens.extend(operand_parts)
        
        return tokens
    
    @staticmethod
    def calculate_junk_repetition_rates(function_str, junk_code_blocks) -> List[float]:
        """
        计算每个垃圾指令组的重复率向量
        
        Args:
            function_str: 二进制函数指令表
            junk_groups: 垃圾指令组列表，每个元素是一个垃圾指令组字符串
            
        Returns:
            重复率向量，长度等于垃圾指令组数量
        """
        normalized_asm = RepetitionAnalyzer._normalize_assembly(function_str)
        tokens = RepetitionAnalyzer._tokenize_assembly(normalized_asm)
        token_freq = Counter(tokens)
        total_tokens = len(tokens)
        
        # 整理为整段形式
        junk_groups = []
        for junk_code_block in junk_code_blocks:
            # 获得basic_block中的指令
            junk_group = ''
            for bb_instruction_str in junk_code_block:
                junk_group += bb_instruction_str
                junk_group += '\n'
            junk_groups.append(junk_group)
        
        repetition_rates = []
        
        for junk_code in junk_groups:
            # 标准化并分词垃圾指令组
            normalized_junk = RepetitionAnalyzer._normalize_assembly(junk_code)
            junk_tokens = RepetitionAnalyzer._tokenize_assembly(normalized_junk)
            
            if not junk_tokens:
                repetition_rates.append(0.0)
                continue
                
            # 计算每个token在目标函数中的出现频率
            token_frequencies = []
            for token in junk_tokens:
                if token in token_freq:
                    # 频率 = token出现次数 / 总token数
                    frequency = token_freq[token] / total_tokens
                else:
                    frequency = 0.0
                token_frequencies.append(frequency)
            
            # 计算平均重复率
            if token_frequencies:
                avg_rate = np.mean(token_frequencies)
                repetition_rates.append(float(avg_rate))
            else:
                repetition_rates.append(0.0)
        
        repetition = np.array(repetition_rates, dtype=np.float32)
        return repetition

# x64: $1 (\text{nop}) + 14 \times 2 (\text{lea/push}) + 91 (\text{xchg}) = \mathbf{120}$ 个动作。
# x32: $1 + 6 \times 2 + 15 = \mathbf{28}$ 个动作。
class JunkBlockGenerator:
    def __init__(self, arch="x64"):
        self.arch = arch
        self.reg_pools = {
            "x64": ["rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"],
            "x32": ["eax", "ebx", "ecx", "edx", "esi", "edi"]
        }
        self.safe_regs = self.reg_pools.get(arch, self.reg_pools["x64"])

    def generate_junk_blocks(self):
        junk_blocks = []

        # 模式 1: NOP (Cost: 1)
        junk_blocks.append(["nop"])

        # 模式 2 & 3 & 4 (针对单个寄存器的操作)
        for reg in self.safe_regs:
            # 模式 2: LEA (Cost: 1)
            junk_blocks.append([f"lea {reg}, [{reg}]"])
            # 模式 3: NOTx2 (Cost: 2)
            #junk_blocks.append([f"not {reg}", f"not {reg}"])
            # 模式 4: PUSH/POP (Cost: 2)
            junk_blocks.append([f"push {reg}", f"pop {reg}"])

        # 模式 5: XCHGx2 (Cost: 2)
        # 使用 combinations 消除对称冗余：(rax, rbx) == (rbx, rax)
        reg_pairs = list(itertools.combinations(self.safe_regs, 2))
        for r1, r2 in reg_pairs:
            junk_blocks.append([f"xchg {r1}, {r2}", f"xchg {r1}, {r2}"])
            
        return junk_blocks

    # 保存为JSON
    @classmethod
    def save_junk_code_blocks_to_json(cls, data, filename="junk_blocks.json"):
        """保存垃圾指令列表到JSON文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # print(f"\n已保存 {len(data)} 个垃圾指令列表到 {filename}")

    # 加载JSON文件
    @classmethod
    def load_junk_code_blocks_from_json(cls, filename="junk_blocks.json"):
        """从JSON文件加载垃圾指令列表"""
        with open(filename, 'r', encoding='utf-8') as f:
            junk_code_blocks = json.load(f)
        
        # print(f"\n已从 {filename} 加载 {len(junk_code_blocks)} 个垃圾指令列表")
        return junk_code_blocks

class JunkCodeInserter:
    def __init__(self, junk_code_blocks):
        # self.function = function
        # self.decoder = decoder
        self.junk_code_blocks = junk_code_blocks

        self.junk_code_blocks_dict = self.to_i_group()
    
    def to_i_group(self):
        junk_code_blocks_dict = {}
        for idx, junk_code_block in enumerate(self.junk_code_blocks):
            # 获得basic_block中的指令
            iis_list = []
            for bb_instruction_str in junk_code_block:
                iis_list.append(BPE_CFR.IInstruction(bb_instruction_str))
                # 仅包含一个basic_block
            i_junk_code_bbs_list = [BPE_CFR.IBasicBlock(iis_list, symbol=None)]
            ig = BPE_CFR.IGroup(i_junk_code_bbs_list)
            junk_code_blocks_dict[idx] = ig
        return junk_code_blocks_dict
    
    def insert_junk_code(self, function, selected_basic_block_index, selected_instruction_index, selected_junk_code_index, decoder):
        # 解析选择节点
        #selected_basic_block = function[selected_basic_block_index]
        #selected_instruction = selected_basic_block[selected_instruction_index]

        # 解析插入指令
        selected_junk_code = self.junk_code_blocks_dict[selected_junk_code_index]

        # 插入指令
        function.insert_group(selected_basic_block_index, selected_instruction_index, selected_junk_code, decoder, mode='local')
        return len(self.junk_code_blocks[selected_junk_code_index]), 0   # 返回插入的指令数，供reward计算使用

# ============================================ 影子模式 ===================================================
class ShadowJunkCodeInserter:
    def __init__(self, junk_code_blocks):
        self.junk_code_blocks = junk_code_blocks

        self.junk_code_blocks_dict = self.to_shadow_instruction_list_dict()
    
    def to_shadow_instruction_list_dict(self):
        junk_blocks_dict = {}
        for idx, junk_code_block in enumerate(self.junk_code_blocks):
            # 获得basic_block中的指令
            junk_instruction_list = []
            for bb_instruction_str in junk_code_block:
                asm = BPE_utils.normalization(bb_instruction_str)
                junk_instruction_list.append(BPE_CFR.ShadowInstruction(asm, idx=None))
            junk_blocks_dict[idx] = junk_instruction_list
        return junk_blocks_dict
    
    def insert_junk_code(self, function, selected_basic_block_index, selected_instruction_index, selected_junk_code_index):
        # 解析选择节点
        selected_basic_block = function[selected_basic_block_index]
        selected_instruction = selected_basic_block[selected_instruction_index]

        # 解析插入指令
        selected_junk_code = self.junk_code_blocks_dict[selected_junk_code_index]

        # 插入指令
        for junk_instruction in selected_junk_code:
            selected_basic_block.insert(selected_instruction.idx, junk_instruction)
        return len(self.junk_code_blocks[selected_junk_code_index]), 0   # 返回插入的指令数，供reward计算使用
