import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import shutil
import torch
import numpy as np
import multiprocessing
import time
import asyncio

from pathlib import Path
from envs.gym_env.env_arguments import EnvArgs
from envs.gym_env.env_wrapper import SABREWrapper
from envs.score.mix_differ import MixSimilarityServer
from arch.perceptor.embedder import MixEmbedder, MixEmbedderServer
from arch.agent import FunctionObfuscationAgent_AI
from arch.arch_arguments import ArchArgs
from train_arguments import TrainArgs

# --- 服务器辅助函数 ---
def run_mix_embedder_server_process(PalmTree_path, PalmTree_vocab, LLM_type, device, env_num, host, port):
    server = MixEmbedderServer(PalmTree_path, PalmTree_vocab, LLM_type, device, env_num, host, port)
    asyncio.run(server.run())

def run_mix_similarity_server_process(device, env_num, target_model_types, host, port):
    server = MixSimilarityServer(device, env_num, target_model_types, host, port)
    asyncio.run(server.run())

def batch_obfuscate(output_dir, arguments_yaml_path, checkpoint_name, ex_lib):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    env_args = EnvArgs(arguments_yaml_path)
    arch_args = ArchArgs(arguments_yaml_path)
    train_args = TrainArgs(arguments_yaml_path)
    
    env_args.eval_mode = True
    env_args.num_processes = 1
    train_args.shadow_mode = False 

    multiprocessing.set_start_method('spawn', force=True)

    print(f"[*] Starting background servers...")
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
        env_args.mix_similarity_server_port)))

    for p in procs: p.start()
    time.sleep(8) 

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

    checkpoint_path = os.path.join(train_args.checkpoints_save_path, checkpoint_name)
    if os.path.exists(checkpoint_path):
        sabre_agent.load_state_dict(torch.load(checkpoint_path, map_location=env_args.sabre_device))
        print(f"[*] Loaded agent weights from {checkpoint_path}")
    sabre_agent.eval()

    print(f"[*] Initializing Batch Processing Loop...")

    # 初始化一次环境，之后通过 reset 切换内部二进制
    env = SABREWrapper(env_args, rank=0)
    env.auto_refresh = False 

    while True:
        try:
            obs, info = env.reset()
            if env.stop_loop:
                print(f"[*] Task queue exhausted. Stopping loop.")
                break
            
            if ex_lib:
                original_path = os.environ.get("LIBRARY_PATH", "")
                binary_directory = info["source_binary_path"].replace(info["binary_name"], '')
                print("LIBRARY_PATH:", binary_directory, ':', original_path)
                os.environ["LIBRARY_PATH"] = f"{binary_directory}:{original_path}"
            
            done = False
            # 从 info 中获取当前处理的基础文件名（用于日志打印）
            current_bin_name = info.get('binary_name', 'unknown_bin')
            print(f"[*] Processing: {current_bin_name} | Function: {info.get('function_name')}")

            n_stack = env_args.num_frame_stack
            stack_buffers = {k: torch.zeros((1, n_stack) + v.shape).to(env_args.sabre_device) 
                             for k, v in obs.items() if isinstance(v, np.ndarray)}
            for k in stack_buffers:
                stack_buffers[k][0, -1] = torch.from_numpy(obs[k]).to(env_args.sabre_device)

            hxs = torch.zeros(sabre_agent.core.lstm_num_layers, 1, arch_args.hidden_dim).to(env_args.sabre_device)
            cxs = torch.zeros(sabre_agent.core.lstm_num_layers, 1, arch_args.hidden_dim).to(env_args.sabre_device)
            recurrent_hidden_states = (hxs, cxs)
            masks = torch.ones(1, 1).to(env_args.sabre_device)

            while not done:
                input_obs = {k: v for k, v in stack_buffers.items()}
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

                env_action = {k: v.cpu().numpy().item() for k, v in actions.items()}
                obs, reward, done, info = env.step(env_action)

                for k in stack_buffers:
                    stack_buffers[k][:, :-1] = stack_buffers[k][:, 1:].clone()
                    stack_buffers[k][0, -1] = torch.from_numpy(obs[k]).to(env_args.sabre_device)

            # --- 修改后的核心提取部分 ---
            if 'rewritten_binary_path' in info:
                gen_path = info['rewritten_binary_path']
                source_path = info['source_binary_path']
                path_obj = Path(source_path)
                target_env = path_obj.parent.name

                # 自动提取文件名，包含后缀
                gen_filename = os.path.basename(gen_path)
                final_out_path = os.path.join(output_dir, target_env + '_' + gen_filename.replace('_0', ''))
                
                shutil.copy(gen_path, final_out_path)
                print(f"[+] Saved obfuscated binary to: {final_out_path}")
            # ---------------------------

        except Exception as e:
            print(f"[!] Error during obfuscation: {e}")
            continue

    env.close()
    print(f"[*] Cleaning up servers...")
    for p in procs: p.terminate()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch Binary Obfuscation Tool")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("-l", "--ex_lib", action='store_true', default=False, help="extra lib for rewritten")
    # parser.add_argument("-c", "--config", default="arguments_palmtree_valid_test_small.yaml", help="Config path")
    # parser.add_argument("-c", "--config", default="arguments_palmtree_valid_test_medium.yaml", help="Config path")
    # parser.add_argument("-c", "--config", default="arguments_palmtree_valid_test_large.yaml", help="Config path")
    parser.add_argument("-c", "--config", default="arguments_palmtree_obfuscate_main.yaml", help="Config path")
    # parser.add_argument("-c", "--config", default="arguments_palmtree_obfuscate_heartbleed.yaml", help="Config path")
    # parser.add_argument("-c", "--config", default="arguments_palmtree_obfuscate_spec.yaml", help="Config path")
    # parser.add_argument("-c", "--config", default="arguments_palmtree_obfuscate_sort.yaml", help="Config path")
    parser.add_argument("-m", "--model", default="sabre_agent_update_200.pth", help="Model name")
    args = parser.parse_args()
    
    batch_obfuscate(args.output, args.config, args.model, args.ex_lib)