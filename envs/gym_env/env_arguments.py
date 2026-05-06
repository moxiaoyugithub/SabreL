import warnings
warnings.filterwarnings("ignore")

import yaml

from envs.obfuscate.junk_code import JunkBlockGenerator
from envs.obfuscate.opaque_predicate import OpaquePredicateInserter

from envs.gym_env import env_utils

from dataset.sampler import SABREDataSet, ShadowSABREDataSet

class EnvArgs:
    def __init__(self, arguments_yaml_path):
        # 从yaml加载参数
        args = None
        with open(arguments_yaml_path, 'r', encoding='utf-8') as f:
            args = yaml.safe_load(f)
            # 从json加载训练数据目录
            if args['train']['shadow_mode']:
                envs_train_dataset = ShadowSABREDataSet(args['train']['train_data_index_json_path'])
                envs_test_dataset = ShadowSABREDataSet(args['train']['test_data_index_json_path'])
            else:
                envs_train_dataset = SABREDataSet(args['train']['train_data_index_json_path'])
                envs_test_dataset = SABREDataSet(args['train']['test_data_index_json_path'])
            
            args = args['env']
        
        # 基础环境名称
        self.base_env_name = args['base_env_name']
        
        # 源数据：通过平衡采样生成的训练函数列表，其中保存了读取被混淆函数所需的信息（二进制路径, gtirb路径, 二进制名, 函数地址）,在每次reset时切换到下一个以保证模型泛化
        self.env_train_data = envs_train_dataset.gen_env_data(env_num=args['num_processes'], shuffle=True)
        self.env_test_data = envs_test_dataset.gen_env_data(env_num=args['num_processes'], shuffle=True)
        # 源数据路径由采样器生成并提前保存在json中，被读进self.env_train_data

        # 临时文件的读写路径
        self.rewritten_binary_directory = args['rew_binary_dir']
        self.rewritten_gtirb_directory_r = args['rew_gtirbs_dir'] + 'r/'
        self.rewritten_gtirb_directory_w = args['rew_gtirbs_dir'] + 'w/'
        # 清理上次运行的文件
        env_utils.delete_and_recreate_folder(self.rewritten_gtirb_directory_r)
        env_utils.delete_and_recreate_folder(self.rewritten_gtirb_directory_w)

        # 观测模块参数
        self.mix_embedder_server_host = args['mix_embedder_server_host']
        self.mix_embedder_server_port = args['mix_embedder_server_port']
        
        self.PalmTree_embedder_path = args['PalmTree_embedder_path']
        self.PalmTree_vocab_path = args['PalmTree_vocab_path']

        if args['LLM_embedder_type']:
            self.LLM_embedder_type = args['LLM_embedder_type']  # 如果启用大模型辅助，则需要提供其类型和路径
        else:
            self.LLM_embedder_type = None   # 不加载大模型使用单PalmTree模型

        # 决策模块参数
        self.action_type_num = 3    # 混淆类型数

        self.max_blocks = args['max_blocks']                # 混淆参数-指针网络最大观测/选择基本块数量
        self.max_instructions = args['max_instructions']    # 混淆参数-指针网络最大观测/选择指令数量

        # 加载预先生成的垃圾代码块
        self.junk_blocks = JunkBlockGenerator.load_junk_code_blocks_from_json(filename=args['junk_blocks_path'])

        self.predicate_num = OpaquePredicateInserter.opaque_predicate_function_num  # 混淆参数-预设的不透明谓词模板数目
        self.junk_num = len(self.junk_blocks)                                       # 混淆参数-预设的垃圾指令组数目

        # 相似度模型参数
        self.similarity_mode = args['similarity_mode']
        self.target_model_types = args['target_model_types']
        self.mix_similarity_server_host = args['mix_similarity_server_host']
        self.mix_similarity_server_port = args['mix_similarity_server_port']
        
        # 回合成功指标
        self.obf_success_similarity = args['obf_success_similarity']
        self.obf_max_inst_growth = args['obf_max_inst_growth']

        # 并行参数
        self.num_processes = args['num_processes']
        self.num_frame_stack = args['num_frame_stack']

        # 各模块的运算设备
        self.sabre_device = args['sabre_device']
        self.embedder_device = args['embedder_device']
        self.differ_device = args['differ_device']

        self.debug = args['debug']
        self.draw_cfg = args['draw_cfg']
