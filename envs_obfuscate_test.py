import warnings
warnings.filterwarnings("ignore")

import torch

from envs.binary_process_editor.BPE_utils import binary_read

from envs.obfuscate.junk_code import JunkBlockGenerator, RepetitionAnalyzer
from envs.obfuscate.mix_obfuscator import MixObfuscator

from arch.perceptor.embedder import MixEmbedder


if __name__ == '__main__':
    # 参数
    gtirb_directory = 'dataset/gtirb_test/'
    binary_directory = 'dataset/bin_test/'
    rewritten_binary_directory = 'dataset/rew_bin/'
    rewritten_gtirb_directory = 'dataset/rew_gtirb/'
    rewritten_gtirb_directory_r = rewritten_gtirb_directory + '/r/'
    rewritten_gtirb_directory_w = rewritten_gtirb_directory + '/w/'
    binary_file_name = 'false'
    rewritten_binary_file_name = 'false'
    # function_address = '0x29d8'

    LLM_embedder_type = None

    PalmTree_embedder_path = 'arch/perceptor/palmtree_pre_trained_model/palmtree/transformer.ep19'
    PalmTree_vocab_path = 'arch/perceptor/palmtree_pre_trained_model/palmtree/vocab'

    max_instructions = 30
    max_blocks = 100

    junk_num = 5


    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载二进制
    cfr = binary_read(binary_directory, gtirb_directory, binary_file_name)
    decoder = cfr.get_decoder()
    function = cfr.find_function_by_name('main', strict=True)
    function_address = function.get_entry_adress()
    function.show(decoder)
    function.CFG(decoder).draw('CFG_orignal.png', decoder=decoder)
    
    # 从json加载保存的生成结果
    junk_code_blocks = JunkBlockGenerator.load_junk_code_blocks_from_json(filename="envs/obfuscate/junk_blocks.json")
    print(f"\nload from junk_blocks.json")

    print(f"select {len(junk_code_blocks)} junk blocks:")
    for i, junk_code_block in enumerate(junk_code_blocks):
        print(f"{i}: {junk_code_block}")

    # 计算重复率向量
    function_str = function.str(decoder)
    repetition = RepetitionAnalyzer.calculate_junk_repetition_rates(function_str, junk_code_blocks)
    print(f"repetition: {repetition}")

    # 加载混合混淆器
    mix_obfuscator = MixObfuscator(rewritten_binary_directory, rewritten_gtirb_directory_w, rewritten_gtirb_directory_r, junk_code_blocks, rank=0, debug=False, draw_cfg=False)
    mix_obfuscator.reset(function, function_address, 'main', binary_file_name)
    # 混淆测试
    # action = {
    #     'action_type': 0,
    #     'selected_basic_block': 1,
    #     'selected_instruction': 5,
    #     'predicate': 3,
    #     'junk': 2
    # }
    
    action = {
        'action_type': 1,
        'selected_basic_block': 2,
        'selected_instruction': 0,
        'predicate': 2,
        'junk': 0
    }
    i = 0
    mix_obfuscator.action_execute(action, step_i=i)
    mix_obfuscator.function.CFG(mix_obfuscator.decoder).draw(f'CFG_step_{i}.png', decoder=mix_obfuscator.decoder)
    
    functions = cfr.find_all_function_by_name('main')
    for key in functions.keys():
        print(f'find function {functions[key].name}: {key}')
