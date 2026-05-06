import re
import os
import subprocess

from envs.binary_process_editor import BPE_CFR

def normalization(asm_str):
    # 分组 1: 指令名
    # 分组 2: 操作数部分
    # 注意：增加了一个匹配单指令（如 endbr64）的情况
    pattern = r'^(\s*([a-zA-Z0-9]+))(\s+[^;\n]+)?'

    def replace_func(match):
        full_line = match.group(0)
        instr = match.group(2).lower()
        operands = match.group(3)

        # 如果没有操作数（例如 endbr64, ret, nop）
        if not operands:
            return full_line
        
        # 1. 跳转指令：保持原样
        if instr.startswith('j') or instr == 'loop':
            return full_line
        
        # 2. 调用指令：替换目标为 FUNCTION
        if instr == 'call':
            return f"{match.group(1)} FUNCTION"
        
        # 3. 其他指令：替换数字偏移量
        # 匹配 0x... 或者 孤立的数字
        # 排除掉类似 xmm0, r12 这种带数字的寄存器名（通常寄存器名不只是数字）
        def offset_replacer(op_match):
            val = op_match.group(0)
            # 如果是纯数字或十六进制，且不是寄存器的一部分
            return "OFFSET"

        # 只对操作数部分进行数字替换
        # 匹配十六进制或十进制数，确保它是独立的单词
        new_operands = re.sub(r'\b(0x[0-9a-fA-F]+|\d+)\b', 'OFFSET', operands)
        
        return f"{match.group(1)}{new_operands}"

    # 使用 MULTILINE 模式逐行处理
    return re.sub(pattern, replace_func, asm_str, flags=re.MULTILINE)

def load_gtirb_to_cfr(gtirb_directory, gtirb_file_name):
    if not gtirb_directory.endswith('/'):
        gtirb_directory += '/'
    ir = BPE_CFR.gtirb.IR.load_protobuf(gtirb_directory + gtirb_file_name)
    cfr = BPE_CFR.CFR(ir)
    return cfr

def binary_read(binary_directory, gtirb_directory, binary_file_name):
    # 创建文件夹
    if not os.path.exists(gtirb_directory):
        os.mkdir(gtirb_directory)
    # bin -> gtirb
    gtirb_file_name = binary_file_name + '.gtirb'
    command = 'ddisasm {} --ir {}'.format(str(os.path.join(binary_directory, binary_file_name)),  str(os.path.join(gtirb_directory, gtirb_file_name)))
    # print('command:', command)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    # gtirb -> cfr
    cfr = load_gtirb_to_cfr(gtirb_directory, gtirb_file_name)
    return cfr

def binary_rewrite(rewritten_binary_directory, rewritten_gtirb_directory, cfr, rewritten_binary_file_name):
    # 创建文件夹
    if not os.path.exists(rewritten_gtirb_directory):
        os.mkdir(rewritten_gtirb_directory)
    # cfr -> gtirb
    rewritten_gtirb_file_name = rewritten_binary_file_name + '.gtirb'
    cfr.write(str(os.path.join(rewritten_gtirb_directory, rewritten_gtirb_file_name)))
    # gtirb -> bin
    command = 'gtirb-pprinter {} -b {}'.format(str(os.path.join(rewritten_gtirb_directory, rewritten_gtirb_file_name)), str(os.path.join(rewritten_binary_directory, rewritten_binary_file_name)))
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    return result