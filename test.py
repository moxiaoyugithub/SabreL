import os
import warnings

warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import shutil
import torch
import asyncio
import time
import multiprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from datetime import datetime
from sklearn.manifold import TSNE

# 环境与模型组件
from envs.gym_env.env_utils import delete_and_recreate_folder
from envs.gym_env.env_arguments import EnvArgs
from envs.gym_env.env_wrapper import SABREWrapper, ShadowSABREWrapper
from envs.score.mix_differ import MixSimilarityServer, ShadowMixSimilarityServer
from arch.perceptor.embedder import MixEmbedder, MixEmbedderServer
from envs.gym_env.env_parallel import EnvMaker
from arch.agent import FunctionObfuscationAgent_AI
from arch.arch_arguments import ArchArgs
from train_arguments import TrainArgs

# --- Server Process Helpers ---
def run_mix_embedder_server_process(PalmTree_path, PalmTree_vocab, LLM_type, device, env_num, host, port):
    server = MixEmbedderServer(PalmTree_path, PalmTree_vocab, LLM_type, device, env_num, host, port)
    asyncio.run(server.run())

def run_mix_similarity_server_process(device, env_num, target_model_types, host, port, shadow_mode):
    server = (ShadowMixSimilarityServer if shadow_mode else MixSimilarityServer)(device, env_num, target_model_types, host, port)
    asyncio.run(server.run())

class SABREEvaluator:
    def __init__(self, arguments_path, checkpoints_name=None, label="Agent_LLM"):
        # 1. 基础配置
        self.env_args = EnvArgs(arguments_path)
        self.arch_args = ArchArgs(arguments_path)
        self.train_args = TrainArgs(arguments_path)
        self.device = self.env_args.sabre_device
        self.is_shadow = self.train_args.shadow_mode
        self.env_args.eval_mode = True
        self.label = label
        
        # 2. 目录初始化
        self._init_directories()
        
        # 3. 启动基础设施
        multiprocessing.set_start_method('spawn', force=True)
        self._start_servers()
        
        # 4. 加载环境与模型
        wrapper_class = ShadowSABREWrapper if self.is_shadow else SABREWrapper
        self.envs = EnvMaker(wrapper_class).make_vec_envs(self.env_args)
        self._load_agent(checkpoints_name)

    def _init_directories(self):
        delete_and_recreate_folder('dataset/rew_bin/')
        delete_and_recreate_folder('dataset/rew_gtirb/r/')
        delete_and_recreate_folder('dataset/rew_gtirb/w/')
        
        ts = datetime.now().strftime('%m%dH%M')
        mode = "Shadow" if self.is_shadow else "Entity"
        self.root_dir = f"evaluation_results/{self.label}_{mode}_{ts}"
        self.sub_dirs = {
            "metrics": os.path.join(self.root_dir, "metrics"),
            "plots": os.path.join(self.root_dir, "analysis_plots"),
            "bins": os.path.join(self.root_dir, "binaries_collection") if not self.is_shadow else None
        }
        for d in self.sub_dirs.values():
            if d: os.makedirs(d, exist_ok=True)

    def _start_servers(self):
        print(f'[SABREEvaluator]: Starting Servers...')
        self.procs = []
        self.procs.append(multiprocessing.Process(target=run_mix_embedder_server_process, args=(
            self.env_args.PalmTree_embedder_path, self.env_args.PalmTree_vocab_path, 
            self.env_args.LLM_embedder_type, 
            self.env_args.embedder_device, self.env_args.num_processes, 
            self.env_args.mix_embedder_server_host, self.env_args.mix_embedder_server_port)))
        
        self.procs.append(multiprocessing.Process(target=run_mix_similarity_server_process, args=(
            self.env_args.differ_device, self.env_args.num_processes, 
            self.env_args.target_model_types, 
            self.env_args.mix_similarity_server_host, self.env_args.mix_similarity_server_port, 
            self.is_shadow)))
        for p in self.procs: p.start()
        time.sleep(5)

    def _load_agent(self, checkpoints_name):
        output_dim = MixEmbedder.get_output_dim(LLM_embedder_type=self.env_args.LLM_embedder_type)
        self.agent = FunctionObfuscationAgent_AI(
            autoregressive_embedding_dim=self.arch_args.autoregressive_embedding_dim,
            function_PalmTree_embedding_dim=output_dim['function_PalmTree_embedding_dim'],
            function_LLM_embedding_dim=output_dim['function_LLM_embedding_dim'],
            hidden_dim=self.arch_args.hidden_dim,
            max_blocks=self.arch_args.max_blocks,
            max_instructions=self.arch_args.max_instructions,
            predicate_num=self.arch_args.predicate_num,
            junk_num=self.arch_args.junk_num
        ).to(self.device)

        cp_path = os.path.join(self.train_args.checkpoints_save_path, checkpoints_name) if checkpoints_name else None
        if cp_path and os.path.exists(cp_path):
            self.agent.load_state_dict(torch.load(cp_path, map_location=self.device))
            print(f"[SABREEvaluator]: Loaded weights from {checkpoints_name}")
        self.agent.eval()
    
    def _shutdown_servers(self):
        print("[SABREEvaluator]: Cleaning up resource and servers...")
        for p in self.procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2) # 必须 join，等待进程完全释放资源
                if p.is_alive():
                    p.kill() # 如果还不退，强制 kill
        
        # 强制清理可能残留的 perf 进程，这是导致延迟的元凶
        os.system("pkill -9 perf > /dev/null 2>&1")
        # 释放并行环境
        self.envs.close()

    # ==========================================
    # 核心流程模块
    # ==========================================
    def run_evaluation(self, num_episodes=50):
        process_buffers = [[] for _ in range(self.env_args.num_processes)]
        episode_counts = 0
        obs, infos = self.envs.reset()
        
        # Step 0 初始化
        for i in range(self.env_args.num_processes):
            process_buffers[i].append(self._create_step_log(0, infos[i], done=False))

        # LSTM States
        hxs = torch.zeros(self.agent.core.lstm_num_layers, self.env_args.num_processes, self.agent.core.hidden_dim).to(self.device)
        cxs = torch.zeros(self.agent.core.lstm_num_layers, self.env_args.num_processes, self.agent.core.hidden_dim).to(self.device)
        recurrent_hidden_states = (hxs, cxs)
        masks = torch.ones(self.env_args.num_processes, 1).to(self.device)

        print(f"[SABREEvaluator]: Running Evaluation Pipeline...")
        function_count = 0
        while episode_counts < num_episodes:
            with torch.no_grad():
                recurrent_hidden_states, actions, _, _, _ = self.agent.act(
                    obs['function_PalmTree_embedding'], obs['function_PalmTree_mask'], 
                    obs.get('function_LLM_embedding'), recurrent_hidden_states, masks,
                    obs['junk_repeat_ratio'], obs['available_actions_mask'], 
                    return_logits=False
                )

            obs, rewards, dones, infos = self.envs.step(actions)
            
            for i in range(self.env_args.num_processes):
                # 注入元数据用于对齐
                log = self._create_step_log(infos[i].get('step_i', 0), infos[i], dones[i])
                log.update({
                    'action_type': actions['action_type'][i].item(),
                    'source_path': infos[i].get('source_binary_path'),
                    'rewritten_path': infos[i].get('rewritten_binary_path'),
                    'bin_name': infos[i].get('binary_name'),
                    'func_name': infos[i].get('function_name'),
                    'rank': i
                })
                process_buffers[i].append(log)

                if dones[i]:
                    if not self.is_shadow: self._save_physical_binary(episode_counts, infos[i])
                    episode_counts += 1
                    process_buffers[i].append(self._create_step_log(0, infos[i], done=False))
                    function_count += 1

            masks = torch.FloatTensor(1.0 - dones).view(-1, 1).to(self.device)
        print(f"[SABREEvaluator]: Processed {function_count} functions")

        # 1. 数据对齐与分发
        df, episodes_data = self._process_rollout_data(process_buffers)
        
        # 2. 只有实体模式执行性能测试
        if not self.is_shadow:
            self._run_physical_analysis(episodes_data)
        
        # 3. 统一绘图
        self._visualize_all(df, episodes_data)
        
        # 4. 关闭服务器
        self._shutdown_servers()

    def _create_step_log(self, step_idx, info, done):
        # 隐蔽度降低定义为 KL 散度
        kl = info.get('stealthiness_kl', 1e-6)
        return {
            "step": step_idx,
            "mix_sim": info['similarity'],
            "sub_sims": info.get('similarity_details', {}),
            "stealthiness_cost": kl,  # 隐蔽度指标
            "node_count": info.get('node_count', 1),
            "edge_count": info.get('edge_count', 0),
            "cyclomatic": info.get('edge_count', 0) - info.get('node_count', 1) + 2,
            "cfg_density": info.get('edge_count', 0) / max(1, info.get('node_count', 1)),
            "done": done
        }

    # ==========================================
    # 模块：数据对齐 (Data Aligner)
    # ==========================================
    def _process_rollout_data(self, process_buffers):
        all_ep_steps = []
        for buf in process_buffers:
            curr = []
            for entry in buf:
                curr.append(entry)
                if entry['done']:
                    if len(curr) > 1: all_ep_steps.append(deepcopy(curr))
                    curr = []

        rows = []
        episodes_metadata = []
        for ep_idx, steps in enumerate(all_ep_steps):
            s_step, e_step = steps[0], steps[-1]
            
            # 记录该 Episode 的宏观指标用于帕累托和热力图
            episodes_metadata.append({
                "episode": ep_idx,
                "func": e_step['func_name'],
                "initial_sim": s_step['mix_sim'],
                "final_sim": e_step['mix_sim'],
                "sim_reduction": s_step['mix_sim'] - e_step['mix_sim'],
                "complexity": e_step['cyclomatic'],
                "actions": [s['action_type'] for s in steps if 'action_type' in s],
                "source_path": e_step['source_path'],
                "rewritten_path": e_step['rewritten_path']
            })

            for s in steps:
                row = {
                    "episode": ep_idx, "step": s['step'], "mix_sim": s['mix_sim'],
                    "stealthiness_cost": s['stealthiness_cost'], "cyclomatic": s['cyclomatic'], "cfg_density": s['cfg_density']
                }
                for k, v in s['sub_sims'].items(): row[f"sub_{k}"] = v
                rows.append(row)
        
        df = pd.DataFrame(rows).sort_values(['episode', 'step']).reset_index(drop=True)
        df.to_csv(os.path.join(self.sub_dirs['metrics'], "full_rollout_history.csv"), index=False)
        return df, episodes_metadata

    # ==========================================
    # 模块：性能分析 (Physical Analyzer)
    # ==========================================
    def _run_physical_analysis(self, episodes):
        print(f"[SABREEvaluator]: Running Physical Overhead Benchmarking...")
        for ep in episodes:
            try:
                # 空间开销
                s_orig, s_obf = os.path.getsize(ep['source_path']), os.path.getsize(ep['rewritten_path'])
                ep['storage_growth'] = ((s_obf - s_orig) / s_orig) * 100
                
                # 时间开销 (Perf)
                t_orig = self._measure_perf(ep['source_path'])
                t_obf = self._measure_perf(ep['rewritten_path'])
                
                if t_orig > 0:
                    growth = ((t_obf - t_orig) / t_orig) * 100
                    ep['perf_growth'] = max(0.0, growth) # 过滤负向噪音
                else:
                    ep['perf_growth'] = float('inf')

            except Exception as e:
                print(f"Benchmarking error on {ep['func']}: {e}")

    def _measure_perf(self, path, iters=10):
        import subprocess, re
        os.chmod(path, 0o755)
        cycles = []
        for _ in range(iters):
            try:
                res = subprocess.run(["perf", "stat", "-x", ",", "-e", "cycles", path, "--help"], 
                                     stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, timeout=2, text=True)
                m = re.search(r"(\d+),,cycles", res.stderr)
                # print(f"Perf output for {path}: {res.stderr.strip()}")
                if m: cycles.append(int(m.group(1)))
            except: continue
        return np.mean(cycles) if cycles else 0

    # ==========================================
    # 模块：可视化 (Visualizer)
    # ==========================================
    def _visualize_all(self, df, episodes_data):
        # 1. 相似度降低曲线 (Mix + Subs)
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='step', y='mix_sim', color='red', linewidth=3, label='Mix (Avg)', errorbar='ci')
        for col in [c for c in df.columns if c.startswith('sub_')]:
            sns.lineplot(data=df, x='step', y=col, linestyle='--', alpha=0.6, label=col.replace('sub_',''), errorbar=None)
        plt.title('Similarity Decay: Cross-Episode Average')
        plt.ylim(0, 1.05)
        plt.savefig(os.path.join(self.sub_dirs['plots'], "similarity_decay.png"))
        
        # 2. 代价分析曲线 (Shadow/Entity 共用指标)
        plt.figure(figsize=(15, 5))
        # 隐蔽度（KL 散度）
        plt.subplot(1, 3, 1)
        sns.lineplot(data=df, x='step', y='stealthiness_cost', color='blue')
        plt.title('Stealthiness cost (KL Divergence)')
        
        # 圈复杂度
        plt.subplot(1, 3, 2)
        sns.boxplot(data=df, x='step', y='cyclomatic', color='purple')
        plt.title('Logic Complexity Growth (Cyclomatic)')

        # 边点比
        plt.subplot(1, 3, 3)
        sns.lineplot(data=df, x='step', y='cfg_density', color='green')
        plt.title('CFG Density Analysis (Edge/Node)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.sub_dirs['plots'], "aligned_metrics.png"))

        # 3. 策略热力图 (圈复杂度 vs 动作分布)
        ep_df = pd.DataFrame(episodes_data)
        ep_df.to_csv(os.path.join(self.sub_dirs['metrics'], "episodes_history.csv"), index=False)
        act_names = {0: "Block Split", 1: "Opaque Pred", 2: "Junk Code"}
        
        act_dists = []
        for acts in ep_df['actions']:
            dist = np.bincount(acts, minlength=3) / len(acts) if acts else [0,0,0]
            act_dists.append(dist)
        
        dist_df = pd.DataFrame(act_dists, columns=[act_names[i] for i in range(3)])
        
        unique_complexities = ep_df['complexity'].nunique()
        if unique_complexities >= 4:
            dist_df['complexity_q'] = pd.qcut(ep_df['complexity'], q=4, 
                                              labels=['Low','Med-L','Med-H','High'], 
                                              duplicates='drop')
        elif unique_complexities > 1:
            # 如果唯一值不足 4 个，则按实际唯一值数量分 bin
            labels = ['Low', 'High'][:unique_complexities]
            dist_df['complexity_q'] = pd.cut(ep_df['complexity'], bins=unique_complexities, 
                                             labels=labels, duplicates='drop')
        else:
            # 如果复杂度全是一样的，直接归为一类
            dist_df['complexity_q'] = 'Uniform'
        
        plt.figure(figsize=(10, 6))
        heatmap_data = dist_df.groupby('complexity_q').mean()
        sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f")
        plt.title(f'Action Preference vs. Initial Complexity ({self.label})')
        plt.xlabel('Obfuscation Action Type')
        plt.ylabel('Function Complexity Group')
        plt.tight_layout()
        plt.savefig(os.path.join(self.sub_dirs['plots'], "strategy_heatmap.png"), dpi=300)

        # 4. 实体模式专属：帕累托分析 (相似度降低 vs Perf 开销)
        if not self.is_shadow:
            clean_ep = ep_df[ep_df['perf_growth'] != float('inf')]
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=clean_ep, x='perf_growth', y='sim_reduction', size='storage_growth', hue='complexity', alpha=0.7)
            plt.title('Pareto Analysis: Sim Reduction vs. Performance Overhead')
            plt.xlabel('Perf Cycles Growth (%)')
            plt.ylabel('Sim Reduction (%)')
            plt.savefig(os.path.join(self.sub_dirs['plots'], "pareto_analysis.png"))

    def _save_physical_binary(self, ep, info):
        source = info['source_binary_path']
        if os.path.exists(source):
            target = f"ep{ep}_{info['binary_name']}_{info['function_name']}"
            shutil.copy(source, os.path.join(self.sub_dirs['bins'], target))

# --- 使用示例：对比实验脚本 ---
if __name__ == "__main__":
    # checkpoints_name = "sabre_agent_update_400.pth"
    # checkpoints_name = "random"
    
    # eval_llm = SABREEvaluator(arguments_path='arguments_llm_test.yaml', 
    #                         checkpoints_name=checkpoints_name, 
    #                         label="Agent_llm_400")
    # eval_llm.run_evaluation(num_episodes=100)
    
    checkpoints_name = "sabre_agent_update_200.pth"
    eval_no_llm = SABREEvaluator(arguments_path='arguments_palmtree_test.yaml', 
                               checkpoints_name=checkpoints_name, 
                               label="Agent_palmtree_200")
    eval_no_llm.run_evaluation(num_episodes=100)