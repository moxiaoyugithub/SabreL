import warnings
warnings.filterwarnings("ignore")

import yaml

from envs.obfuscate.junk_code import JunkBlockGenerator
from envs.obfuscate.opaque_predicate import OpaquePredicateInserter

class ArchArgs:
    def __init__(self, arguments_yaml_path):
        # 从yaml加载参数
        args = None
        with open(arguments_yaml_path, 'r', encoding='utf-8') as f:
            args = yaml.safe_load(f)
        
        args_env = args['env']
        args_arch = args['arch']
            
        self.hidden_dim = args_arch['hidden_dim']
        self.autoregressive_embedding_dim = args_arch['autoregressive_embedding_dim']
        
        self.max_blocks = args_env['max_blocks']
        self.max_instructions = args_env['max_instructions']
        
        # 加载预先生成的垃圾代码块
        junk_blocks = JunkBlockGenerator.load_junk_code_blocks_from_json(filename=args_env['junk_blocks_path'])

        self.predicate_num = OpaquePredicateInserter.opaque_predicate_function_num  # 混淆参数-预设的不透明谓词模板数目
        self.junk_num = len(junk_blocks)                                            # 混淆参数-预设的垃圾指令组数目

        # 运算设备
        self.decider_device = args_arch['decider_device']

        self.debug = args_arch['debug']
