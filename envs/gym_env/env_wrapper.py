import warnings
warnings.filterwarnings("ignore")

import asyncio
import numpy as np
from copy import deepcopy

import gym
from gym import spaces
from gym.utils import seeding

from envs.binary_process_editor import BPE_utils

from envs.obfuscate.junk_code import RepetitionAnalyzer
from envs.obfuscate.mix_obfuscator import MixObfuscator, ShadowMixObfuscator
from envs.obfuscate.CDManager import ObfuscationActionCDManager

from arch.perceptor.embedder import MixEmbedder, MixEmbedderClient

from envs.score.mix_differ import MixSimilarityClient, ShadowMixSimilarityClient
from envs.score.reward import Obfuscation_step_evaluator

from dataset.sampler import SABREEnvDataLoader

from logs.logger import logger

class SABREWrapper(object):
    def __init__(self, args, rank):
        self.args = args
        self.rank = rank
        self.wrapped_env = gym.make(self.args.base_env_name)
        self._loop = None
        self.eval_mode = self.args.eval_mode
        self.debug = self.args.debug
        self.draw_cfg = self.args.draw_cfg
        
        logger.setup_rank(self.rank)
        
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: ========== env.init ==========', file_only=True)

        # -------------被混淆函数的信息---------------
        # 被混淆的二进制和其中的函数地址
        self.wrapped_env.binary_directory = None
        self.wrapped_env.binary_name = None
        self.wrapped_env.function_address = None
        self.wrapped_env.function_name = None

        # 被混淆的函数，从env_train_data中读取，在每次回合结束reset的时候切换
        self.wrapped_env.function = None                     # 被混淆的代码（对比依据）
        self.wrapped_env.function_hat = None

        # -------------编辑模块---------------
        # 高级修改器需要使用动态值function_hat，不能在初始化时定义，需要在reset()中重置初始化
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Initializing mix_obfuscator...', file_only=True)
        
        self.wrapped_env.mix_obfuscator = MixObfuscator(
            self.args.rewritten_binary_directory, 
            self.args.rewritten_gtirb_directory_w, 
            self.args.rewritten_gtirb_directory_r, 
            self.args.junk_blocks, 
            self.rank, 
            self.debug, 
            self.draw_cfg
            )
        # 动作CD管理器
        self.wrapped_env.cd_config = {0: 0, 1: 0, 2: 0}  # Split无冷却, Opaque冷却4步, Junk无冷却
        self.wrapped_env.cd_manager = None

        #--------------观测模块---------------
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Initializing mix_embedder...', file_only=True)
        
        self.wrapped_env.mix_embedder = MixEmbedderClient(self.rank, self.args.mix_embedder_server_host, self.args.mix_embedder_server_port)
        output_dim = MixEmbedder.get_output_dim(LLM_embedder_type=self.args.LLM_embedder_type)

        # -------------评价模块---------------
        # 多模型综合的代码相似度模型
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Initializing mix_similar_calculator...', file_only=True)
        
        self.wrapped_env.mix_similar_calculator = MixSimilarityClient(self.rank, self.args.mix_similarity_server_host, self.args.mix_similarity_server_port)
        # 奖励计算器
        self.wrapped_env.obfuscation_step_evaluator = None
        
        # -------------环境数据集---------------
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Initializing env_dataloader...', file_only=True)
        if self.eval_mode:
            self.wrapped_env.env_dataloader = SABREEnvDataLoader(self.args.env_test_data, self.rank)
        else:
            self.wrapped_env.env_dataloader = SABREEnvDataLoader(self.args.env_train_data, self.rank)

        # -------------观测/动作空间---------------
        # 观测空间
        if self.args.LLM_embedder_type:
            self.wrapped_env.observation_space = spaces.Dict({
                # 1. function_PalmTree_embedding
                "function_PalmTree_embedding": spaces.Box(
                    low=-np.inf, 
                    high=np.inf, 
                    shape=(self.args.max_blocks, self.args.max_instructions, output_dim['function_PalmTree_embedding_dim']), 
                    dtype=np.float32
                ),
                # 2. function_PalmTree_mask
                "function_PalmTree_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.args.max_blocks, self.args.max_instructions),
                    dtype=np.int8
                ),
                # 3. function_LLM_embedding
                "function_LLM_embedding": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(output_dim['function_LLM_embedding_dim'],),
                    dtype=np.float32
                ), 
                # 4. junk_repeat_ratio
                "junk_repeat_ratio": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.args.junk_num,),
                    dtype=np.float32
                ), 
                # 5. available_actions_mask
                "available_actions_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.args.action_type_num,),
                    dtype=np.int8
                )
            })
        else:
            self.wrapped_env.observation_space = spaces.Dict({
                # 1. function_PalmTree_embedding
                "function_PalmTree_embedding": spaces.Box(
                    low=-np.inf, 
                    high=np.inf, 
                    shape=(self.args.max_blocks, self.args.max_instructions, output_dim['function_PalmTree_embedding_dim']), 
                    dtype=np.float32
                ),
                # 2. function_PalmTree_mask
                "function_PalmTree_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.args.max_blocks, self.args.max_instructions),
                    dtype=np.int8
                ),
                # 3. junk_repeat_ratio
                "junk_repeat_ratio": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.args.junk_num,),
                    dtype=np.float32
                ), 
                # 4. available_actions_mask
                "available_actions_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.args.action_type_num,),
                    dtype=np.int8
                )
            })

        # 动作空间
        self.wrapped_env.action_space = spaces.Dict({
            'action_type': spaces.Discrete(self.args.action_type_num),             # 混淆类型数
            'selected_basic_block': spaces.Discrete(self.args.max_blocks),         # 混淆参数-指针网络最大观测/选择基本块数量
            'selected_instruction': spaces.Discrete(self.args.max_instructions+1), # 混淆参数-指针网络最大观测/选择指令数量（多一位NO_OP）
            'predicate': spaces.Discrete(self.args.predicate_num+1),               # 混淆参数-预设的不透明谓词模板数目（多一位NO_OP）
            'junk': spaces.Discrete(self.args.junk_num+1)                          # 混淆参数-预设的垃圾指令组数目（多一位NO_OP）
        })

        # -------------记录数据---------------
        self.wrapped_env.step_i = 0
        self.wrapped_env.available_actions_mask = None
        self.wrapped_env.state = 0
        self.wrapped_env.beyond_state = 0
        self.wrapped_env.episode = None
        self.wrapped_env.original_opcode_dist = None
        self.wrapped_env.original_similarity = None

        # 控制类数据
        self.wrapped_env.similarity_mode = self.args.similarity_mode
        self.wrapped_env.obf_success_similarity = self.args.obf_success_similarity  # 如果相似度小于0.2则视为混淆成功
        self.wrapped_env.obf_max_inst_growth = self.args.obf_max_inst_growth        # 允许的最大指令增量
        
        # 奖励构成数据
        self.wrapped_env.similarity = None
        self.wrapped_env.similarity_details = None
        self.wrapped_env.beyond_similarity = None
        self.wrapped_env.beyond_similarity_details = None
        self.wrapped_env.instruction_count = None
        self.wrapped_env.beyond_instruction_count = None
        self.wrapped_env.step_inst_growth = None
        self.wrapped_env.step_vcp_growth = None
        self.wrapped_env.accumulated_inst_growth = 0
        self.wrapped_env.accumulated_vcp_growth = 0
        self.wrapped_env.min_similarity = None
        self.wrapped_env.min_similarity_details = None
        self.wrapped_env.action_history = None

        # 自有空间
        self.action_space = self.wrapped_env.action_space
        self.observation_space = self.wrapped_env.observation_space
        
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: ==========================', file_only=True)
        
        self.auto_refresh = True
        self.stop_loop = False
    
    def seed(self, seed=None):
        self.wrapped_env.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_loop(self):
        """获取或创建当前进程的事件循环"""
        if self._loop is None:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                # 如果当前线程没有循环，创建一个新的
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop
    
    def step(self, action):
        loop = self._get_loop()
        return loop.run_until_complete(self._async_step(action))
    
    async def _async_step(self, action):
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: ========== env.step ==========', file_only=True)
        
        # 记录beyond_state
        self.wrapped_env.beyond_state = self.wrapped_env.state

        # 加入混淆
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Executing action...', file_only=True)
        try:
            self.wrapped_env.step_inst_growth, self.wrapped_env.step_vcp_growth, self.wrapped_env.function_hat, action_type = self.wrapped_env.mix_obfuscator.action_execute(action, step_i=self.wrapped_env.step_i)
            self.wrapped_env.action_history[action_type] += 1
            # 动作进入冷却
            self.wrapped_env.cd_manager.update_usage(action_type, self.wrapped_env.step_i)
            self.wrapped_env.available_actions_mask = self.wrapped_env.cd_manager.get_available_mask(self.wrapped_env.step_i + 1)
        
        except Exception:
            # 混淆函数丢失，立刻结束回合，混淆失败
            if self.debug:
                logger.info(f'[Env Rank {self.rank}]: CFR - Function lost, episode end.', file_only=True)
            
            reward = 0
            done = True
            if self.wrapped_env.beyond_similarity:
                beyond_similarity = self.wrapped_env.beyond_similarity
            else:
                beyond_similarity = None
            
            return self.wrapped_env.beyond_state, reward, done, {
                'episode': self.wrapped_env.episode, 
                'step_i': self.wrapped_env.step_i+1, 
                'binary_name': self.wrapped_env.binary_name, 
                'function_name': self.wrapped_env.function_name, 
                'source_binary_path': self.wrapped_env.binary_directory + '/' + self.wrapped_env.binary_name, 
                'rewritten_binary_path': self.args.rewritten_binary_directory + self.wrapped_env.binary_name + '_' + str(self.rank), 
                'beyond_similarity': beyond_similarity, 
                'similarity': None, 
                'original_similarity': self.wrapped_env.original_similarity, 
                'operation_record': self.wrapped_env.mix_obfuscator.operation_records, 
                'similarity_details': None, 
                'edge_count': None, 
                'node_count': None, 
                'stealthiness_kl': None, 
                'accumulated_inst_growth': self.wrapped_env.accumulated_inst_growth, 
                'accumulated_vcp_growth': self.wrapped_env.accumulated_vcp_growth
                }
        
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Action Executed.', file_only=True)
        
        # 计算混淆后的相似度
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Calculating similarity - function_hat {self.args.rewritten_binary_directory}{self.wrapped_env.binary_name}_{str(self.rank)} address {self.wrapped_env.mix_obfuscator.function_address} name {self.wrapped_env.mix_obfuscator.function_name} and {self.wrapped_env.binary_directory}/{self.wrapped_env.binary_name} address {self.wrapped_env.function_address} name {self.wrapped_env.function_name}...', file_only=True)
        
        try:
            cosine_similarity, similarity_details = await self.wrapped_env.mix_similar_calculator.compare(
                self.args.rewritten_binary_directory + self.wrapped_env.binary_name + '_' + str(self.rank), self.wrapped_env.mix_obfuscator.function_address, self.wrapped_env.mix_obfuscator.function_name, 
                self.wrapped_env.binary_directory + '/' + self.wrapped_env.binary_name, self.wrapped_env.function_address, self.wrapped_env.function_name, 
                mode=self.wrapped_env.similarity_mode
                )
        except Exception:
            # 相似度模型无法识别函数，视为达到相似度阈值，混淆成功
            if self.debug:
                logger.info(f'[Env Rank {self.rank}]: SIM - Function lost, episode done.', file_only=True)
            
            cosine_similarity = self.wrapped_env.obf_success_similarity
            similarity_details = {}
            for target_model_type in self.args.target_model_types:
                similarity_details[target_model_type] = self.wrapped_env.obf_success_similarity
        
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Got cosine_similarity - {cosine_similarity} {similarity_details}', file_only=True)
        
        # 记录beyond_similarity
        self.wrapped_env.beyond_similarity = deepcopy(self.wrapped_env.similarity)
        self.wrapped_env.beyond_similarity_details = {}
        for target_model_type in self.args.target_model_types:
            if self.wrapped_env.similarity_details[target_model_type]:
                self.wrapped_env.beyond_similarity_details[target_model_type] = self.wrapped_env.similarity_details[target_model_type]

        for target_model_type in self.args.target_model_types:
            if similarity_details[target_model_type] == None:
                similarity_details[target_model_type] = self.wrapped_env.beyond_similarity_details[target_model_type]
        
        # 更新similarity
        self.wrapped_env.similarity = deepcopy(cosine_similarity)
        self.wrapped_env.similarity_details = deepcopy(similarity_details)

        # 记录min_similarity
        if self.wrapped_env.similarity < self.wrapped_env.min_similarity:
            self.wrapped_env.min_similarity = self.wrapped_env.similarity
        
        for target_model_type in self.args.target_model_types:
            if self.wrapped_env.similarity_details[target_model_type]:
                if self.wrapped_env.min_similarity_details[target_model_type] < self.wrapped_env.similarity_details[target_model_type]:
                    self.wrapped_env.min_similarity_details[target_model_type] = self.wrapped_env.similarity_details[target_model_type]
        
        # 记录beyond_instruction_count
        # self.wrapped_env.beyond_instruction_count = deepcopy(self.wrapped_env.instruction_count)
        
        # 更新instruction_count
        # self.wrapped_env.instruction_count = self.wrapped_env.function_hat.inst_count()
        
        # 计算本步增加的指令数
        # self.wrapped_env.step_inst_growth = self.wrapped_env.instruction_count - self.wrapped_env.beyond_instruction_count

        # 计算累积增加的指令数
        self.wrapped_env.accumulated_inst_growth += self.wrapped_env.step_inst_growth
        self.wrapped_env.accumulated_vcp_growth += self.wrapped_env.step_vcp_growth
        
        # 计算奖励，判断回合是否结束
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Calculating reward...', file_only=True)
        
        reward, done = self.wrapped_env.obfuscation_step_evaluator.calculate_reward_and_done(
            self.wrapped_env.accumulated_inst_growth, 
            self.wrapped_env.step_inst_growth, 
            # self.wrapped_env.similarity, 
            # self.wrapped_env.beyond_similarity, 
            # self.wrapped_env.min_similarity
            self.wrapped_env.similarity_details, 
            self.wrapped_env.beyond_similarity_details, 
            self.wrapped_env.min_similarity_details
            )

        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Got reward - {reward}', file_only=True)

        decoder = self.wrapped_env.function_hat.cfr.get_decoder()
        
        # 计算新的垃圾指令重复率
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Calculating junk_repeat_ratio...', file_only=True)
        
        junk_repeat_ratio = RepetitionAnalyzer.calculate_junk_repetition_rates(self.wrapped_env.function_hat.str(decoder), self.args.junk_blocks)
        
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Got junk_repeat_ratio - {junk_repeat_ratio}', file_only=True)

        # 获得混淆后的新state
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Calculating state...', file_only=True)
        
        state = await self.wrapped_env.mix_embedder.embedding(
            function_mix_embedder_input=self.wrapped_env.function_hat.to_mix_embedder_input(decoder), 
            remaining_budget=self.args.obf_max_inst_growth - self.wrapped_env.accumulated_inst_growth, 
            similarity_details=self.wrapped_env.similarity_details, 
            max_instructions=self.args.max_instructions, 
            max_blocks=self.args.max_blocks
        )
        function_PalmTree_embedding = np.array(state["function_PalmTree_embedding"], dtype=np.float32)
        function_PalmTree_mask = np.array(state["function_PalmTree_mask"], dtype=np.int8)
        if self.args.LLM_embedder_type:
            function_LLM_embedding = np.array(state["function_LLM_embedding"], dtype=np.float32)
            self.wrapped_env.state = {
                "function_PalmTree_embedding": function_PalmTree_embedding, 
                "function_PalmTree_mask": function_PalmTree_mask, 
                "function_LLM_embedding": function_LLM_embedding, 
                "junk_repeat_ratio": junk_repeat_ratio, 
                'available_actions_mask': self.wrapped_env.available_actions_mask
            }
        else:
            self.wrapped_env.state = {
                "function_PalmTree_embedding": function_PalmTree_embedding, 
                "function_PalmTree_mask": function_PalmTree_mask, 
                "junk_repeat_ratio": junk_repeat_ratio, 
                'available_actions_mask': self.wrapped_env.available_actions_mask
            }
        
        # 计算和原始函数的KL散度
        if self.eval_mode:
            kl_val = self.wrapped_env.function_hat.calculate_kl_divergence(self.wrapped_env.original_opcode_dist, decoder)
        
        # 更新步数记录
        self.wrapped_env.step_i += 1
        
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Got state - {type(self.wrapped_env.state)}', file_only=True)
        
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: ==========================', file_only=True)
        
        return self.wrapped_env.state, reward, done, {
            'episode': self.wrapped_env.episode, 
            'step_i': self.wrapped_env.step_i, 
            'binary_name': self.wrapped_env.binary_name,
            'function_name': self.wrapped_env.function_name,
            'source_binary_path': self.wrapped_env.binary_directory + '/' + self.wrapped_env.binary_name, 
            'rewritten_binary_path': self.args.rewritten_binary_directory + self.wrapped_env.binary_name + '_' + str(self.rank), 
            'beyond_similarity': self.wrapped_env.beyond_similarity, 
            'similarity': self.wrapped_env.similarity, 
            'original_similarity': self.wrapped_env.original_similarity, 
            'operation_record': self.wrapped_env.mix_obfuscator.operation_records, 
            'similarity_details': self.wrapped_env.similarity_details, 
            'edge_count': len(list(self.wrapped_env.function_hat.CFG(decoder).edges())), 
            'node_count': len(list(self.wrapped_env.function_hat.CFG(decoder).nodes())), 
            'stealthiness_kl': kl_val if self.eval_mode else None, 
            'accumulated_inst_growth': self.wrapped_env.accumulated_inst_growth, 
            'accumulated_vcp_growth': self.wrapped_env.accumulated_vcp_growth
            }

    def reset(self):
        # 统一使用持有的 loop 运行异步任务
        loop = self._get_loop()
        return loop.run_until_complete(self._async_reset())
    
    async def _async_reset(self):
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: ========== env.reset ==========', file_only=True)
        
        # 更新回合数记录
        if self.wrapped_env.episode != None:
            self.wrapped_env.episode += 1
        else:
            self.wrapped_env.episode = 0
        
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Episode {self.wrapped_env.episode}', file_only=True)
        
        # 重新加载函数（从env_train_data中获得gtirb路径和函数地址，加载到ir->cfr，找到函数）
        if not self.auto_refresh and self.wrapped_env.env_dataloader.reset:
            self.stop_loop = True
            
        episode_data = self.wrapped_env.env_dataloader.sample()
        
        self.wrapped_env.binary_directory = 'dataset/' + episode_data['binary_directory']
        self.wrapped_env.binary_name = episode_data['binary_name']
        self.wrapped_env.gtirb_directory = 'dataset/' + episode_data['gtirb_directory']
        self.wrapped_env.function_address = episode_data['function_address']
        self.wrapped_env.function_name = episode_data['function_name']
        
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Got function from env_train_data - function_name {self.wrapped_env.function_name} function_address {self.wrapped_env.function_address} in bianry {self.wrapped_env.binary_directory}/{self.wrapped_env.binary_name}', file_only=True)
        
        cfr_obf = BPE_utils.load_gtirb_to_cfr(self.wrapped_env.gtirb_directory, self.wrapped_env.binary_name + '.gtirb')
        cfr_obf_hat = deepcopy(cfr_obf)
        
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: CFR ready', file_only=True)
        
        self.wrapped_env.function = cfr_obf.find_function_by_address(self.wrapped_env.function_address)        # 被混淆的函数，作为对比依据存在，不会改变
        self.wrapped_env.function_hat = cfr_obf_hat.find_function_by_address(self.wrapped_env.function_address)# 在本回合中参与迭代的函数
        self.wrapped_env.function_name = self.wrapped_env.function_hat.name
        
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Find function {self.wrapped_env.function_name} by address {self.wrapped_env.function_address}', file_only=True)
        
        # 重置函数修改器
        self.wrapped_env.mix_obfuscator.reset(
            self.wrapped_env.function_hat, 
            self.wrapped_env.function_address, 
            self.wrapped_env.function_name, 
            self.wrapped_env.binary_name
            )
        # 动作CD管理器
        self.wrapped_env.cd_manager = ObfuscationActionCDManager(self.args.action_type_num, self.wrapped_env.cd_config)
        
        # 获取初始掩码
        self.wrapped_env.available_actions_mask = self.wrapped_env.cd_manager.get_available_mask(self.wrapped_env.step_i)

        # 计算出原始相似度（自己和自己）
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Calculating similarity...', file_only=True)
            
        cosine_similarity, similarity_details = await self.wrapped_env.mix_similar_calculator.compare(
            self.wrapped_env.binary_directory + '/' + self.wrapped_env.binary_name, self.wrapped_env.function_address, self.wrapped_env.function_name, 
            self.wrapped_env.binary_directory + '/' + self.wrapped_env.binary_name, self.wrapped_env.function_address, self.wrapped_env.function_name, 
            mode=self.wrapped_env.similarity_mode
            )
        for target_model_type in self.args.target_model_types:
            if similarity_details[target_model_type] == None:
                similarity_details[target_model_type] = 1.0
        
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Got cosine_similarity - {cosine_similarity} {similarity_details}', file_only=True)
        
        decoder = self.wrapped_env.function_hat.cfr.get_decoder()
        
        # 计算垃圾指令重复率
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Calculating junk_repeat_ratio...', file_only=True)
            
        junk_repeat_ratio = RepetitionAnalyzer.calculate_junk_repetition_rates(self.wrapped_env.function_hat.str(decoder), self.args.junk_blocks)
        
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Got junk_repeat_ratio - {junk_repeat_ratio}', file_only=True)
        
        # 计算state
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Calculating state...', file_only=True)
        
        state = await self.wrapped_env.mix_embedder.embedding(
            function_mix_embedder_input=self.wrapped_env.function_hat.to_mix_embedder_input(decoder), 
            remaining_budget=self.args.obf_max_inst_growth - self.wrapped_env.accumulated_inst_growth, 
            similarity_details=similarity_details, 
            max_instructions=self.args.max_instructions, 
            max_blocks=self.args.max_blocks
        )
        function_PalmTree_embedding = np.array(state["function_PalmTree_embedding"], dtype=np.float32)
        function_PalmTree_mask = np.array(state["function_PalmTree_mask"], dtype=np.int8)
        # 可用动作掩码
        # self.wrapped_env.available_actions_mask = np.ones(self.args.action_type_num, dtype=int)
        if self.args.LLM_embedder_type:
            function_LLM_embedding = np.array(state["function_LLM_embedding"], dtype=np.float32)
            self.wrapped_env.state = {
                "function_PalmTree_embedding": function_PalmTree_embedding, 
                "function_PalmTree_mask": function_PalmTree_mask, 
                "function_LLM_embedding": function_LLM_embedding, 
                "junk_repeat_ratio": junk_repeat_ratio, 
                "available_actions_mask": self.wrapped_env.available_actions_mask
            }
        else:
            self.wrapped_env.state = {
                "function_PalmTree_embedding": function_PalmTree_embedding, 
                "function_PalmTree_mask": function_PalmTree_mask, 
                "junk_repeat_ratio": junk_repeat_ratio, 
                "available_actions_mask": self.wrapped_env.available_actions_mask
            }
        
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Got state - {type(self.wrapped_env.state)}', file_only=True)
        
        self.wrapped_env.beyond_state = 0

        # 记录用数据
        self.wrapped_env.step_i = 0
        self.wrapped_env.similarity = cosine_similarity
        self.wrapped_env.similarity_details = similarity_details
        self.wrapped_env.original_similarity = deepcopy(self.wrapped_env.similarity)
        self.wrapped_env.beyond_similarity = None
        self.wrapped_env.beyond_similarity_details = None
        self.wrapped_env.min_similarity = deepcopy(self.wrapped_env.similarity)
        self.wrapped_env.min_similarity_details = deepcopy(self.wrapped_env.similarity_details)
        self.wrapped_env.instruction_count = self.wrapped_env.function.inst_count()
        self.wrapped_env.beyond_instruction_count = None
        self.wrapped_env.step_inst_growth = None
        self.wrapped_env.step_vcp_growth = None
        self.wrapped_env.accumulated_inst_growth = 0
        self.wrapped_env.accumulated_vcp_growth = 0
        self.wrapped_env.action_history = {0:0, 1:0, 2:0}
        
        # 计算和原始函数的KL散度
        if self.eval_mode:
            self.wrapped_env.original_opcode_dist = self.wrapped_env.function.get_opcode_freq(decoder)
            kl_val = self.wrapped_env.function_hat.calculate_kl_divergence(self.wrapped_env.original_opcode_dist, decoder)

        # 加载混淆评价器
        self.wrapped_env.obfuscation_step_evaluator = Obfuscation_step_evaluator(
            self.wrapped_env.obf_max_inst_growth,        # 允许的最大指令增量
            self.wrapped_env.obf_success_similarity      # 判定攻击成功的相似度阈值
        )
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: ==========================', file_only=True)
        
        return self.wrapped_env.state, {
            'episode': self.wrapped_env.episode, 
            'step_i': self.wrapped_env.step_i, 
            'binary_name': self.wrapped_env.binary_name,
            'function_name': self.wrapped_env.function_name,
            'source_binary_path': self.wrapped_env.binary_directory + '/' + self.wrapped_env.binary_name, 
            'rewritten_binary_path': self.args.rewritten_binary_directory + self.wrapped_env.binary_name + '_' + str(self.rank), 
            'beyond_similarity': None, 
            'similarity': self.wrapped_env.similarity, 
            'original_similarity': self.wrapped_env.original_similarity, 
            'operation_record': self.wrapped_env.mix_obfuscator.operation_records, 
            'similarity_details': self.wrapped_env.similarity_details, 
            'edge_count': len(list(self.wrapped_env.function_hat.CFG(decoder).edges())), 
            'node_count': len(list(self.wrapped_env.function_hat.CFG(decoder).nodes())), 
            'stealthiness_kl': kl_val if self.eval_mode else None, 
            'accumulated_inst_growth': self.wrapped_env.accumulated_inst_growth, 
            'accumulated_vcp_growth': self.wrapped_env.accumulated_vcp_growth
            }
    
    def close(self):
        # 只在最后一个环境关闭时清理资源
        if hasattr(self, 'wrapped_env') and hasattr(self.wrapped_env, 'mix_embedder'):
            # 这里可以添加一些逻辑来决定何时真正关闭
            pass

# ============================================ 影子模式 ===================================================
class ShadowSABREWrapper(object):
    def __init__(self, args, rank):
        self.args = args
        self.rank = rank
        self.wrapped_env = gym.make(self.args.base_env_name)
        self._loop = None
        self.eval_mode = self.args.eval_mode
        self.debug = self.args.debug
        self.draw_cfg = self.args.draw_cfg
        
        logger.setup_rank(self.rank)
        
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: ========== env.init ==========', file_only=True)

        # -------------被混淆函数的信息---------------
        # 被混淆的二进制和其中的函数地址
        self.wrapped_env.binary_directory = None
        self.wrapped_env.binary_name = None
        self.wrapped_env.function_address = None
        self.wrapped_env.function_name = None

        # 被混淆的函数，从env_train_data中读取，在每次回合结束reset的时候切换
        self.wrapped_env.function = None                     # 被混淆的代码（对比依据）
        self.wrapped_env.function_hat = None

        # -------------编辑模块---------------
        # 高级修改器需要使用动态值function_hat，不能在初始化时定义，需要在reset()中重置初始化
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Initializing mix_obfuscator...', file_only=True)
        
        self.wrapped_env.mix_obfuscator = ShadowMixObfuscator(
            self.args.junk_blocks, 
            self.rank, 
            self.debug, 
            self.draw_cfg
            )
        # 动作CD管理器
        self.wrapped_env.cd_config = {0: 0, 1: 0, 2: 0}  # Split无冷却, Opaque冷却4步, Junk无冷却
        self.wrapped_env.cd_manager = None

        #--------------观测模块---------------
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Initializing mix_embedder...', file_only=True)
        
        self.wrapped_env.mix_embedder = MixEmbedderClient(self.rank, self.args.mix_embedder_server_host, self.args.mix_embedder_server_port)
        output_dim = MixEmbedder.get_output_dim(LLM_embedder_type=self.args.LLM_embedder_type)

        # -------------评价模块---------------
        # 多模型综合的代码相似度模型
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Initializing mix_similar_calculator...', file_only=True)
        
        self.wrapped_env.mix_similar_calculator = ShadowMixSimilarityClient(self.rank, self.args.mix_similarity_server_host, self.args.mix_similarity_server_port)
        # 奖励计算器
        self.wrapped_env.obfuscation_step_evaluator = None
        
        # -------------环境数据集---------------
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Initializing env_dataloader...', file_only=True)
        if self.eval_mode:
            self.wrapped_env.env_dataloader = SABREEnvDataLoader(self.args.env_test_data, self.rank)
        else:
            self.wrapped_env.env_dataloader = SABREEnvDataLoader(self.args.env_train_data, self.rank)

        # -------------观测/动作空间---------------
        # 观测空间
        if self.args.LLM_embedder_type:
            self.wrapped_env.observation_space = spaces.Dict({
                # 1. function_PalmTree_embedding
                "function_PalmTree_embedding": spaces.Box(
                    low=-np.inf, 
                    high=np.inf, 
                    shape=(self.args.max_blocks, self.args.max_instructions, output_dim['function_PalmTree_embedding_dim']), 
                    dtype=np.float32
                ),
                # 2. function_PalmTree_mask
                "function_PalmTree_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.args.max_blocks, self.args.max_instructions),
                    dtype=np.int8
                ),
                # 3. function_LLM_embedding
                "function_LLM_embedding": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(output_dim['function_LLM_embedding_dim'],),
                    dtype=np.float32
                ), 
                # 4. junk_repeat_ratio
                "junk_repeat_ratio": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.args.junk_num,),
                    dtype=np.float32
                ), 
                # 5. available_actions_mask
                "available_actions_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.args.action_type_num,),
                    dtype=np.int8
                )
            })
        else:
            self.wrapped_env.observation_space = spaces.Dict({
                # 1. function_PalmTree_embedding
                "function_PalmTree_embedding": spaces.Box(
                    low=-np.inf, 
                    high=np.inf, 
                    shape=(self.args.max_blocks, self.args.max_instructions, output_dim['function_PalmTree_embedding_dim']), 
                    dtype=np.float32
                ),
                # 2. function_PalmTree_mask
                "function_PalmTree_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.args.max_blocks, self.args.max_instructions),
                    dtype=np.int8
                ),
                # 3. junk_repeat_ratio
                "junk_repeat_ratio": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.args.junk_num,),
                    dtype=np.float32
                ), 
                # 4. available_actions_mask
                "available_actions_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.args.action_type_num,),
                    dtype=np.int8
                )
            })

        # 动作空间
        self.wrapped_env.action_space = spaces.Dict({
            'action_type': spaces.Discrete(self.args.action_type_num),             # 混淆类型数
            'selected_basic_block': spaces.Discrete(self.args.max_blocks),         # 混淆参数-指针网络最大观测/选择基本块数量
            'selected_instruction': spaces.Discrete(self.args.max_instructions+1), # 混淆参数-指针网络最大观测/选择指令数量（多一位NO_OP）
            'predicate': spaces.Discrete(self.args.predicate_num+1),               # 混淆参数-预设的不透明谓词模板数目（多一位NO_OP）
            'junk': spaces.Discrete(self.args.junk_num+1)                          # 混淆参数-预设的垃圾指令组数目（多一位NO_OP）
        })

        # -------------记录数据---------------
        self.wrapped_env.step_i = 0
        self.wrapped_env.available_actions_mask = None
        self.wrapped_env.state = 0
        self.wrapped_env.beyond_state = 0
        self.wrapped_env.episode = None
        self.wrapped_env.original_opcode_dist = None
        self.wrapped_env.original_similarity = None

        # 控制类数据
        self.wrapped_env.similarity_mode = self.args.similarity_mode
        self.wrapped_env.obf_success_similarity = self.args.obf_success_similarity  # 如果相似度小于0.2则视为混淆成功
        self.wrapped_env.obf_max_inst_growth = self.args.obf_max_inst_growth        # 允许的最大指令增量
        
        # 奖励构成数据
        self.wrapped_env.similarity = None
        self.wrapped_env.similarity_details = None
        self.wrapped_env.beyond_similarity = None
        self.wrapped_env.beyond_similarity_details = None
        self.wrapped_env.instruction_count = None
        self.wrapped_env.beyond_instruction_count = None
        self.wrapped_env.step_inst_growth = None
        self.wrapped_env.step_vcp_growth = None
        self.wrapped_env.accumulated_inst_growth = 0
        self.wrapped_env.accumulated_vcp_growth = 0
        self.wrapped_env.min_similarity = None
        self.wrapped_env.min_similarity_details = None
        self.wrapped_env.action_history = None

        # 自有空间
        self.action_space = self.wrapped_env.action_space
        self.observation_space = self.wrapped_env.observation_space
        
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: ==========================', file_only=True)
        
        self.auto_refresh = True
        self.stop_loop = False
    
    def seed(self, seed=None):
        self.wrapped_env.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_loop(self):
        """获取或创建当前进程的事件循环"""
        if self._loop is None:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                # 如果当前线程没有循环，创建一个新的
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop
    
    def step(self, action):
        loop = self._get_loop()
        return loop.run_until_complete(self._async_step(action))
    
    async def _async_step(self, action):
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: ========== env.step ==========', file_only=True)
        
        # 记录beyond_state
        self.wrapped_env.beyond_state = self.wrapped_env.state

        # 加入混淆
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Executing action...', file_only=True)
        try:
            self.wrapped_env.step_inst_growth, self.wrapped_env.step_vcp_growth, self.wrapped_env.function_hat, action_type = self.wrapped_env.mix_obfuscator.action_execute(action)
            self.wrapped_env.action_history[action_type] += 1
            # 动作进入冷却
            self.wrapped_env.cd_manager.update_usage(action_type, self.wrapped_env.step_i)
            self.wrapped_env.available_actions_mask = self.wrapped_env.cd_manager.get_available_mask(self.wrapped_env.step_i + 1)
            
        except Exception:
            # 混淆函数丢失，立刻结束回合，混淆失败
            if self.debug:
                logger.info(f'[Env Rank {self.rank}]: CFR - Function lost, episode end.', file_only=True)
            
            reward = 0
            done = True
            if self.wrapped_env.beyond_similarity:
                beyond_similarity = self.wrapped_env.beyond_similarity
            else:
                beyond_similarity = None
            
            return self.wrapped_env.beyond_state, reward, done, {
                'episode': self.wrapped_env.episode, 
                'step_i': self.wrapped_env.step_i+1, 
                'binary_name': self.wrapped_env.binary_name, 
                'function_name': self.wrapped_env.function_name, 
                'source_binary_path': self.wrapped_env.binary_directory + '/' + self.wrapped_env.binary_name, 
                'rewritten_binary_path': self.args.rewritten_binary_directory + self.wrapped_env.binary_name + '_' + str(self.rank), 
                'beyond_similarity': beyond_similarity, 
                'similarity': None, 
                'original_similarity': self.wrapped_env.original_similarity, 
                'operation_record': self.wrapped_env.mix_obfuscator.operation_records, 
                'similarity_details': None, 
                'edge_count': None, 
                'node_count': None, 
                'stealthiness_kl': None, 
                'accumulated_inst_growth': self.wrapped_env.accumulated_inst_growth, 
                'accumulated_vcp_growth': self.wrapped_env.accumulated_vcp_growth
                }
        
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Action Executed.', file_only=True)
        
        # 计算混淆后的相似度
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Calculating similarity - function_hat {self.args.rewritten_binary_directory}{self.wrapped_env.binary_name}_{str(self.rank)} address {self.wrapped_env.mix_obfuscator.function_address} name {self.wrapped_env.mix_obfuscator.function_name} and {self.wrapped_env.binary_directory}/{self.wrapped_env.binary_name} address {self.wrapped_env.function_address} name {self.wrapped_env.function_name}...', file_only=True)
        
        try:
            cosine_similarity, similarity_details = await self.wrapped_env.mix_similar_calculator.compare(
                self.wrapped_env.function_hat, 
                self.wrapped_env.function, 
                mode=self.wrapped_env.similarity_mode
                )
        except Exception:
            # 相似度模型无法识别函数，视为达到相似度阈值，混淆成功
            if self.debug:
                logger.info(f'[Env Rank {self.rank}]: SIM - Function lost, episode done.', file_only=True)
            
            cosine_similarity = self.wrapped_env.obf_success_similarity
            similarity_details = {}
            for target_model_type in self.args.target_model_types:
                similarity_details[target_model_type] = self.wrapped_env.obf_success_similarity
        
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Got cosine_similarity - {cosine_similarity} {similarity_details}', file_only=True)
        
        # 记录beyond_similarity
        self.wrapped_env.beyond_similarity = deepcopy(self.wrapped_env.similarity)
        self.wrapped_env.beyond_similarity_details = deepcopy(self.wrapped_env.similarity_details)      # 上一步一定有值

        # 处理本步的缺失值，将缺失值填充为上一步的值
        for target_model_type in self.args.target_model_types:
            if similarity_details[target_model_type] == None:
                similarity_details[target_model_type] = self.wrapped_env.beyond_similarity_details[target_model_type]
        
        # 更新similarity
        self.wrapped_env.similarity = deepcopy(cosine_similarity)
        self.wrapped_env.similarity_details = deepcopy(similarity_details)

        # 记录min_similarity
        if self.wrapped_env.similarity < self.wrapped_env.min_similarity:
            self.wrapped_env.min_similarity = self.wrapped_env.similarity
        
        for target_model_type in self.args.target_model_types:
            if self.wrapped_env.min_similarity_details[target_model_type] < self.wrapped_env.similarity_details[target_model_type]:
                self.wrapped_env.min_similarity_details[target_model_type] = self.wrapped_env.similarity_details[target_model_type]
            
        # 记录beyond_instruction_count
        # self.wrapped_env.beyond_instruction_count = deepcopy(self.wrapped_env.instruction_count)
        
        # 更新instruction_count
        # self.wrapped_env.instruction_count = self.wrapped_env.function_hat.inst_count()
        
        # 计算本步增加的指令数
        # self.wrapped_env.step_inst_growth = self.wrapped_env.instruction_count - self.wrapped_env.beyond_instruction_count

        # 计算累积增加的指令数
        self.wrapped_env.accumulated_inst_growth += self.wrapped_env.step_inst_growth
        self.wrapped_env.accumulated_vcp_growth += self.wrapped_env.step_vcp_growth

        # 计算奖励，判断回合是否结束
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Calculating reward...', file_only=True)
        
        reward, done = self.wrapped_env.obfuscation_step_evaluator.calculate_reward_and_done(
            self.wrapped_env.accumulated_inst_growth, 
            self.wrapped_env.step_inst_growth, 
            # self.wrapped_env.similarity, 
            # self.wrapped_env.beyond_similarity, 
            # self.wrapped_env.min_similarity
            self.wrapped_env.similarity_details, 
            self.wrapped_env.beyond_similarity_details, 
            self.wrapped_env.min_similarity_details
            )

        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Got reward - {reward}', file_only=True)
        
        # 计算新的垃圾指令重复率
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Calculating junk_repeat_ratio...', file_only=True)
        
        junk_repeat_ratio = RepetitionAnalyzer.calculate_junk_repetition_rates(self.wrapped_env.function_hat.str(), self.args.junk_blocks)
        
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Got junk_repeat_ratio - {junk_repeat_ratio}', file_only=True)

        # 获得混淆后的新state
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Calculating state...', file_only=True)
        
        state = await self.wrapped_env.mix_embedder.embedding(
            function_mix_embedder_input=self.wrapped_env.function_hat.to_mix_embedder_input(), 
            remaining_budget=self.args.obf_max_inst_growth - self.wrapped_env.accumulated_inst_growth, 
            similarity_details=self.wrapped_env.similarity_details,  
            max_instructions=self.args.max_instructions, 
            max_blocks=self.args.max_blocks
        )
        function_PalmTree_embedding = np.array(state["function_PalmTree_embedding"], dtype=np.float32)
        function_PalmTree_mask = np.array(state["function_PalmTree_mask"], dtype=np.int8)
        if self.args.LLM_embedder_type:
            function_LLM_embedding = np.array(state["function_LLM_embedding"], dtype=np.float32)
            self.wrapped_env.state = {
                "function_PalmTree_embedding": function_PalmTree_embedding, 
                "function_PalmTree_mask": function_PalmTree_mask, 
                "function_LLM_embedding": function_LLM_embedding, 
                "junk_repeat_ratio": junk_repeat_ratio, 
                "available_actions_mask": self.wrapped_env.available_actions_mask
            }
        else:
            self.wrapped_env.state = {
                "function_PalmTree_embedding": function_PalmTree_embedding, 
                "function_PalmTree_mask": function_PalmTree_mask, 
                "junk_repeat_ratio": junk_repeat_ratio, 
                "available_actions_mask": self.wrapped_env.available_actions_mask
            }
        
        # 计算和原始函数的KL散度
        if self.eval_mode:
            kl_val = self.wrapped_env.function_hat.calculate_kl_divergence(self.wrapped_env.original_opcode_dist)
        
        # 更新步数记录
        self.wrapped_env.step_i += 1
        
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Got state - {type(self.wrapped_env.state)}', file_only=True)
        
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: ==========================', file_only=True)
        
        return self.wrapped_env.state, reward, done, {
            'episode': self.wrapped_env.episode, 
            'step_i': self.wrapped_env.step_i, 
            'binary_name': self.wrapped_env.binary_name, 
            'function_name': self.wrapped_env.function_name, 
            'source_binary_path': self.wrapped_env.binary_directory + '/' + self.wrapped_env.binary_name, 
            'rewritten_binary_path': self.args.rewritten_binary_directory + self.wrapped_env.binary_name + '_' + str(self.rank), 
            'beyond_similarity': self.wrapped_env.beyond_similarity, 
            'similarity': self.wrapped_env.similarity, 
            'original_similarity': self.wrapped_env.original_similarity, 
            'operation_record': self.wrapped_env.mix_obfuscator.operation_records, 
            'similarity_details': self.wrapped_env.similarity_details, 
            'edge_count': len(list(self.wrapped_env.function_hat.CFG().edges())), 
            'node_count': len(list(self.wrapped_env.function_hat.CFG().nodes())), 
            'stealthiness_kl': kl_val if self.eval_mode else None, 
            'accumulated_inst_growth': self.wrapped_env.accumulated_inst_growth, 
            'accumulated_vcp_growth': self.wrapped_env.accumulated_vcp_growth
            }

    def reset(self):
        # 统一使用持有的 loop 运行异步任务
        loop = self._get_loop()
        return loop.run_until_complete(self._async_reset())
    
    async def _async_reset(self):
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: ========== env.reset ==========', file_only=True)
        
        # 更新回合数记录
        if self.wrapped_env.episode != None:
            self.wrapped_env.episode += 1
        else:
            self.wrapped_env.episode = 0
        
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Episode {self.wrapped_env.episode}', file_only=True)
        
        # 重新加载函数（从env_train_data中获得gtirb路径和函数地址）
        if not self.auto_refresh and self.wrapped_env.env_dataloader.reset:
            self.stop_loop = True
        
        episode_data = self.wrapped_env.env_dataloader.sample()

        self.wrapped_env.binary_directory = 'dataset/' + episode_data['binary_directory']
        self.wrapped_env.binary_name = episode_data['binary_name']
        self.wrapped_env.gtirb_directory = 'dataset/' + episode_data['gtirb_directory']
        self.wrapped_env.function_address = episode_data['function_address']
        self.wrapped_env.function_name = episode_data['function_name']
        
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Got function from env_train_data - function_name {self.wrapped_env.function_name} function_address {self.wrapped_env.function_address} in bianry {self.wrapped_env.binary_directory}/{self.wrapped_env.binary_name}', file_only=True)
        
        self.wrapped_env.function = episode_data['data']                            # 被混淆的函数，作为对比依据存在，不会改变
        self.wrapped_env.function_hat = deepcopy(self.wrapped_env.function)         # 在本回合中参与迭代的函数
        
        # 重置函数修改器
        self.wrapped_env.mix_obfuscator.reset(
            self.wrapped_env.function_hat, 
            self.wrapped_env.function_address, 
            self.wrapped_env.function_name, 
            self.wrapped_env.binary_name
            )
        # 动作CD管理器
        self.wrapped_env.cd_manager = ObfuscationActionCDManager(self.args.action_type_num, self.wrapped_env.cd_config)
        
        # 获取初始掩码
        self.wrapped_env.available_actions_mask = self.wrapped_env.cd_manager.get_available_mask(self.wrapped_env.step_i)

        # 计算出原始相似度（自己和自己）
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Calculating similarity...', file_only=True)
            
        cosine_similarity, similarity_details = await self.wrapped_env.mix_similar_calculator.compare(
            self.wrapped_env.function_hat, 
            self.wrapped_env.function, 
            mode=self.wrapped_env.similarity_mode
            )
        for target_model_type in self.args.target_model_types:
            if similarity_details[target_model_type] == None:
                similarity_details[target_model_type] = 1.0
        
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Got cosine_similarity - {cosine_similarity} {similarity_details}', file_only=True)
        
        # 计算垃圾指令重复率
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Calculating junk_repeat_ratio...', file_only=True)
            
        junk_repeat_ratio = RepetitionAnalyzer.calculate_junk_repetition_rates(self.wrapped_env.function_hat.str(), self.args.junk_blocks)
        
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Got junk_repeat_ratio - {junk_repeat_ratio}', file_only=True)
        
        # 计算state
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Calculating state...', file_only=True)
        
        state = await self.wrapped_env.mix_embedder.embedding(
            function_mix_embedder_input=self.wrapped_env.function_hat.to_mix_embedder_input(), 
            remaining_budget=self.args.obf_max_inst_growth - self.wrapped_env.accumulated_inst_growth, 
            similarity_details=similarity_details, 
            max_instructions=self.args.max_instructions, 
            max_blocks=self.args.max_blocks
        )
        function_PalmTree_embedding = np.array(state["function_PalmTree_embedding"], dtype=np.float32)
        function_PalmTree_mask = np.array(state["function_PalmTree_mask"], dtype=np.int8)
        # 可用动作掩码
        # self.wrapped_env.available_actions_mask = np.ones(self.args.action_type_num, dtype=int)
        if self.args.LLM_embedder_type:
            function_LLM_embedding = np.array(state["function_LLM_embedding"], dtype=np.float32)
            self.wrapped_env.state = {
                "function_PalmTree_embedding": function_PalmTree_embedding, 
                "function_PalmTree_mask": function_PalmTree_mask, 
                "function_LLM_embedding": function_LLM_embedding, 
                "junk_repeat_ratio": junk_repeat_ratio, 
                "available_actions_mask": self.wrapped_env.available_actions_mask
            }
        else:
            self.wrapped_env.state = {
                "function_PalmTree_embedding": function_PalmTree_embedding, 
                "function_PalmTree_mask": function_PalmTree_mask, 
                "junk_repeat_ratio": junk_repeat_ratio, 
                "available_actions_mask": self.wrapped_env.available_actions_mask
            }
        
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: Got state - {type(self.wrapped_env.state)}', file_only=True)
        
        self.wrapped_env.beyond_state = 0

        # 记录用数据
        self.wrapped_env.step_i = 0
        self.wrapped_env.similarity = cosine_similarity
        self.wrapped_env.similarity_details = similarity_details
        self.wrapped_env.original_similarity = deepcopy(self.wrapped_env.similarity)
        self.wrapped_env.beyond_similarity = None
        self.wrapped_env.beyond_similarity_details = None
        self.wrapped_env.min_similarity = deepcopy(self.wrapped_env.similarity)
        self.wrapped_env.min_similarity_details = deepcopy(self.wrapped_env.similarity_details)
        self.wrapped_env.instruction_count = self.wrapped_env.function.inst_count()
        self.wrapped_env.beyond_instruction_count = None
        self.wrapped_env.step_inst_growth = None
        self.wrapped_env.step_vcp_growth = None
        self.wrapped_env.accumulated_inst_growth = 0
        self.wrapped_env.accumulated_vcp_growth = 0
        self.wrapped_env.action_history = {0:0, 1:0, 2:0}
        
        # 计算和原始函数的KL散度
        if self.eval_mode:
            self.wrapped_env.original_opcode_dist = self.wrapped_env.function.get_opcode_freq()
            kl_val = self.wrapped_env.function_hat.calculate_kl_divergence(self.wrapped_env.original_opcode_dist)

        # 加载混淆评价器
        self.wrapped_env.obfuscation_step_evaluator = Obfuscation_step_evaluator(
            self.wrapped_env.obf_max_inst_growth,        # 允许的最大指令增量
            self.wrapped_env.obf_success_similarity      # 判定攻击成功的相似度阈值
        )
        if self.debug:
            logger.info(f'[Env Rank {self.rank}]: ==========================', file_only=True)
        
        return self.wrapped_env.state, {
            'episode': self.wrapped_env.episode, 
            'step_i': self.wrapped_env.step_i, 
            'binary_name': self.wrapped_env.binary_name,
            'function_name': self.wrapped_env.function_name,
            'source_binary_path': self.wrapped_env.binary_directory + '/' + self.wrapped_env.binary_name, 
            'rewritten_binary_path': self.args.rewritten_binary_directory + self.wrapped_env.binary_name + '_' + str(self.rank), 
            'beyond_similarity': None, 
            'similarity': self.wrapped_env.similarity, 
            'original_similarity': self.wrapped_env.original_similarity, 
            'operation_record': self.wrapped_env.mix_obfuscator.operation_records, 
            'similarity_details': self.wrapped_env.similarity_details, 
            'edge_count': len(list(self.wrapped_env.function_hat.CFG().edges())), 
            'node_count': len(list(self.wrapped_env.function_hat.CFG().nodes())), 
            'stealthiness_kl': kl_val if self.eval_mode else None, 
            'accumulated_inst_growth': self.wrapped_env.accumulated_inst_growth, 
            'accumulated_vcp_growth': self.wrapped_env.accumulated_vcp_growth
            }
    
    def close(self):
        # 只在最后一个环境关闭时清理资源
        if hasattr(self, 'wrapped_env') and hasattr(self.wrapped_env, 'mix_embedder'):
            # 这里可以添加一些逻辑来决定何时真正关闭
            pass
