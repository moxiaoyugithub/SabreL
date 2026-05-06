import warnings
warnings.filterwarnings("ignore")

from envs.obfuscate.junk_code import JunkBlockGenerator

if __name__ == '__main__':
    # 垃圾指令生成
    junk_code_generator = JunkBlockGenerator()
    junk_code_blocks = junk_code_generator.generate_junk_blocks()
    print(f"generate {len(junk_code_blocks)} junk_blocks")
    
    # 储存垃圾指令生成结果到json
    JunkBlockGenerator.save_junk_code_blocks_to_json(junk_code_blocks, filename="envs/obfuscate/junk_blocks.json")
    print(f"\nsave to junk_blocks.json")
    # 从json加载保存的生成结果
    junk_code_blocks = JunkBlockGenerator.load_junk_code_blocks_from_json(filename="envs/obfuscate/junk_blocks.json")
    print(f"\nload from junk_blocks.json")

    print(f"select {len(junk_code_blocks)} junk blocks:")
    for i, junk_code_block in enumerate(junk_code_blocks):
        print(f"{i}: {junk_code_block}")