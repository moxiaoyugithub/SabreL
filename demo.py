import os
import sys
sys.setrecursionlimit(10000)
import warnings

warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import time
import asyncio
import multiprocessing
import torch
import numpy as np

from envs.gym_env.env_arguments import EnvArgs
from envs.gym_env.env_wrapper import SABREWrapper, ShadowSABREWrapper

from envs.score.mix_differ import MixSimilarityServer, ShadowMixSimilarityServer

from arch.perceptor.embedder import MixEmbedder, MixEmbedderServer
from arch.agent import FunctionObfuscationAgent_AI
from arch.arch_arguments import ArchArgs

from train_arguments import TrainArgs

# --- 服务器启动辅助函数 ---
def run_mix_embedder_server_process(PalmTree_path, PalmTree_vocab, LLM_type, device, env_num,host, port):
    server = MixEmbedderServer(PalmTree_path, PalmTree_vocab, LLM_type, device, env_num, host, port)
    asyncio.run(server.run())

def run_mix_similarity_server_process(device, env_num, target_model_types, host, port, shadow_mode):
    server = (ShadowMixSimilarityServer if shadow_mode else MixSimilarityServer)(device, env_num, target_model_types, host, port)
    asyncio.run(server.run())

def run_single_obfuscation_demo(arguments_yaml_path, checkpoint_name='random'):
    # 加载参数
    env_args = EnvArgs(arguments_yaml_path)
    arch_args = ArchArgs(arguments_yaml_path)
    train_args = TrainArgs(arguments_yaml_path)
    
    env_args.eval_mode = True  # 演示模式下启用eval_mode
    env_args.num_processes = 1  # 单进程运行
    
    multiprocessing.set_start_method('spawn', force=True)
    
    # 启动分布式计算服务器
    print(f'[Demo]: ============= Starting test =============')
    print(f'[Demo]: Starting servers...')
    procs = []
    procs.append(multiprocessing.Process(target=run_mix_embedder_server_process, args=(
        env_args.PalmTree_embedder_path, 
        env_args.PalmTree_vocab_path, 
        env_args.LLM_embedder_type, 
        env_args.embedder_device, 
        env_args.num_processes, 
        env_args.mix_embedder_server_host, 
        env_args.mix_embedder_server_port)))
    
    procs.append(multiprocessing.Process(target=run_mix_similarity_server_process, args=(
        env_args.differ_device, 
        env_args.num_processes, 
        env_args.target_model_types, 
        env_args.mix_similarity_server_host, 
        env_args.mix_similarity_server_port, 
        train_args.shadow_mode)))

    for p in procs: p.start()
    time.sleep(5) # 等待服务器初始化
    
    # 1. 环境初始化 (根据模式选择实体或影子)
    print(f"[Demo]: Initializing {'Shadow' if train_args.shadow_mode else 'Real'} Environment...")
    if train_args.shadow_mode:
        # 影子模式：仅在文本修改，不改写磁盘二进制
        env = ShadowSABREWrapper(env_args, rank=0)
    else:
        # 实体模式：会产生重写后的二进制文件
        env = SABREWrapper(env_args, rank=0)
    # 2. 模型初始化并加载权重
    output_dim = MixEmbedder.get_output_dim(LLM_embedder_type=env_args.LLM_embedder_type)
    sabre_agent = FunctionObfuscationAgent_AI(
        autoregressive_embedding_dim=arch_args.autoregressive_embedding_dim,
        function_PalmTree_embedding_dim=output_dim['function_PalmTree_embedding_dim'], 
        function_LLM_embedding_dim=output_dim['function_LLM_embedding_dim'], 
        hidden_dim=arch_args.hidden_dim,
        max_blocks=arch_args.max_blocks,
        max_instructions=arch_args.max_instructions,
        predicate_num=arch_args.predicate_num,
        junk_num=arch_args.junk_num
    ).to(env_args.sabre_device)

    if checkpoint_name == 'random':
        print(f"[Demo]: Using random weights.")
    else:
        checkpoint_path = os.path.join(train_args.checkpoints_save_path, checkpoint_name)
        if os.path.exists(checkpoint_path):
            sabre_agent.load_state_dict(torch.load(checkpoint_path, map_location=env_args.sabre_device))
            print(f"[Demo]: Loaded model from {checkpoint_path}")
        else:
            print(f"[Demo]: No checkpoint found, using random weights.")
    
    sabre_agent.eval()

    # 3. 开始回合
    obs, info = env.reset()
    done = False
    
    # 手动维护一个帧堆叠缓冲区 (L=4)
    # 模拟 VecPyTorchFrameStackDict 的行为
    n_stack = env_args.num_frame_stack
    stack_buffers = {}
    for key in obs.keys():
        if isinstance(obs[key], np.ndarray):
            buf_shape = (1, n_stack) + obs[key].shape
            stack_buffers[key] = torch.zeros(buf_shape).to(env_args.sabre_device)
            # 填充初始帧到序列末尾
            stack_buffers[key][0, -1] = torch.from_numpy(obs[key]).to(env_args.sabre_device)

    # 初始化 LSTM 隐藏状态
    hxs = torch.zeros(sabre_agent.core.lstm_num_layers, 1, arch_args.hidden_dim).to(env_args.sabre_device)
    cxs = torch.zeros(sabre_agent.core.lstm_num_layers, 1, arch_args.hidden_dim).to(env_args.sabre_device)
    recurrent_hidden_states = (hxs, cxs)
    masks = torch.ones(1, 1).to(env_args.sabre_device)

    print(f"\n[Demo]: Start - Target Binary: {env_args.rewritten_binary_directory}{env.wrapped_env.binary_name}_{str(env.rank)} | Target Function: {env.wrapped_env.function_name}")
    print(f"Initial Similarity: {info['similarity']}")
    print("-" * 60)

    step_idx = 0
    total_reward = 0

    # 4. 混淆主循环
    while not done:
        step_idx += 1
        
        # 构造 Agent 输入
        input_obs = {k: v for k, v in stack_buffers.items()}
        # 注意：这里需要根据你 Core 的 forward 结构传入参数
        # 假设核心需要的键是 pt_emb, pt_mask, llm_emb 等
        
        with torch.no_grad():
            recurrent_hidden_states, _, actions, _, _, _ = sabre_agent.act(
                function_PalmTree_embedding=input_obs['function_PalmTree_embedding'],
                function_PalmTree_mask=input_obs['function_PalmTree_mask'],
                function_LLM_embedding=input_obs.get('function_LLM_embedding'),
                recurrent_hidden_states_in=recurrent_hidden_states,
                recurrent_mask_in=masks,
                junk_repeat_ratio=input_obs.get('junk_repeat_ratio'),
                available_actions_mask=input_obs.get('available_actions_mask')
            )

        # 转换动作以便环境执行
        # actions 是 Tensor 字典，需要转为 numpy 标量
        env_action = {k: v.cpu().numpy().item() for k, v in actions.items()}
        
        # 环境执行一步
        obs, reward, done, info = env.step(env_action)
        total_reward += reward

        # 更新帧堆叠缓冲区
        for key in stack_buffers.keys():
            stack_buffers[key][:, :-1] = stack_buffers[key][:, 1:].clone()
            stack_buffers[key][0, -1] = torch.from_numpy(obs[key]).to(env_args.sabre_device)

        # 提取混淆动作的可读描述
        act_type_map = {0: "Block Split", 1: "Opaque Predicate", 2: "Junk Code"}
        act_name = act_type_map.get(env_action['action_type'], "Unknown")
        
        # 打印当前步详细信息
        print(f"STEP {step_idx}:")
        print(f"  > Action Type: {act_name}")
        print(f"  > Target BB:   {env_action['selected_basic_block']}")
        print(f"  > Similarity:  {info['similarity']} (Raw: {info.get('beyond_similarity', 'N/A')} -> {info['similarity']})")
        print(f"  > similarity_details:  {info['similarity_details']}")
        print(f"  > Inst Growth:   {info['accumulated_inst_growth']}")
        print(f"  > Step Reward: {reward:.4f}")
        
        # 记录具体的修改记录 (如果环境支持)
        if 'operation_record' in info:
            last_op = info['operation_record'][-1] if info['operation_record'] else "N/A"
            print(f"  > Op Record:   {last_op}")
        
        print("-" * 30)

    print(f"\n[Demo]: End - Total Steps: {step_idx} | Cumulative Reward: {total_reward:.4f}")
    print(f"Final Similarity: {info['similarity']}")
    env.close()
    
    # 演示结束，关闭服务器
    print(f'[Demo]: ============= Test over, servers shutdown =============')
    for p in procs: p.terminate()

if __name__ == "__main__":
    arguments_yaml_path = 'arguments_palmtree_demo.yaml'
    # arguments_yaml_path = 'arguments_palmtree_valid_test.yaml'
    checkpoint_name = "sabre_agent_update_200.pth"
    # checkpoint_name = "random"  # 使用随机权重演示
    
    # 运行演示
    run_single_obfuscation_demo(arguments_yaml_path, checkpoint_name=checkpoint_name)