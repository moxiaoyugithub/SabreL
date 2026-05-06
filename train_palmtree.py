import os
import warnings
warnings.filterwarnings("ignore")

import time
import asyncio
import multiprocessing
import torch

from envs.gym_env.env_utils import delete_and_recreate_folder

from envs.gym_env.env_arguments import EnvArgs
from envs.gym_env.env_wrapper import SABREWrapper, ShadowSABREWrapper
from envs.gym_env.env_parallel import EnvMaker

from envs.score.mix_differ import MixSimilarityServer, ShadowMixSimilarityServer

from arch.perceptor.embedder import MixEmbedder, MixEmbedderServer
from arch.agent import FunctionObfuscationAgent_AI
from arch.arch_arguments import ArchArgs

from buffer.rollout import SABRERolloutStorage
from algo.ppo import PPO

from train_arguments import TrainArgs

from torch.utils.tensorboard import SummaryWriter
from logs.logger import logger

# --- 服务器启动辅助函数 ---
def run_mix_embedder_server_process(PalmTree_path, PalmTree_vocab, LLM_type, device, env_num,host, port):
    server = MixEmbedderServer(PalmTree_path, PalmTree_vocab, LLM_type, device, env_num, host, port)
    asyncio.run(server.run())

def run_mix_similarity_server_process(device, env_num, target_model_types, host, port, shadow_mode):
    server = (ShadowMixSimilarityServer if shadow_mode else MixSimilarityServer)(device, env_num, target_model_types, host, port)
    asyncio.run(server.run())

def main():
    # 1. 环境准备
    logger.info(f'[Main]: Preparing...')
    delete_and_recreate_folder('dataset/rew_bin/')
    delete_and_recreate_folder('dataset/rew_gtirb/r/')
    delete_and_recreate_folder('dataset/rew_gtirb/w/')
    delete_and_recreate_folder('logs/log_files/')
    delete_and_recreate_folder('logs/tensorboard_logs/palmtree/')
    
    logger.info(f'[Main]: Reading configuration...')
    arguments_yaml_path = 'arguments_palmtree_train.yaml'
    env_args = EnvArgs(arguments_yaml_path)
    arch_args = ArchArgs(arguments_yaml_path)
    train_args = TrainArgs(arguments_yaml_path)
    
    env_args.eval_mode = False  # 训练模式下关闭eval_mode

    multiprocessing.set_start_method('spawn', force=True)
    
    # 2. 启动分布式计算服务器
    logger.info(f'[Main]: Starting servers...')
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

    # 3. 初始化环境与智能体
    logger.info(f'[Main]: Starting SABRE environment...')
    env_maker = EnvMaker(ShadowSABREWrapper if train_args.shadow_mode else SABREWrapper)
    envs = env_maker.make_vec_envs(env_args)
    
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

    # 4. 初始化 PPO 学习器
    logger.info(f'[Main]: Initializing leaner...')
    leaner = PPO(
        agent=sabre_agent,
        ppo_clip_param=train_args.ppo_clip_param,
        ppo_epoch=train_args.ppo_epoch,
        ppo_num_mini_batch=train_args.ppo_num_mini_batch,
        ppo_value_loss_coef=train_args.ppo_value_loss_coef,
        ppo_entropy_coef=train_args.ppo_entropy_coef,
        ppo_adam_learning_rate=float(train_args.ppo_adam_learning_rate),
        ppo_adam_epsilon=float(train_args.ppo_adam_epsilon),
        ppo_max_grad_norm=train_args.ppo_max_grad_norm
    )
    
    # 4. 初始化经验存储器
    logger.info(f'[Main]: Initializing rollouts...')
    rollouts = SABRERolloutStorage(
        storage_num_steps=train_args.storage_num_steps,
        num_processes=env_args.num_processes,
        observation_space=envs.observation_space,
        action_space=envs.action_space,
        hidden_dim=arch_args.hidden_dim,
        lstm_num_layers=sabre_agent.core.lstm_num_layers,
        num_frame_stack=env_args.num_frame_stack,
        device=env_args.sabre_device
    )
    
    # 初始化 TensorBoard 记录器
    writer = SummaryWriter(log_dir='logs/tensorboard_logs/palmtree/')

    # 5. 训练主循环
    logger.info(f'[Main]: ============= Starting main loop =============')
    obs, _ = envs.reset()
    rollouts.obs['function_PalmTree_embedding'][0].copy_(obs['function_PalmTree_embedding'])
    rollouts.obs['function_PalmTree_mask'][0].copy_(obs['function_PalmTree_mask'])
    if 'function_LLM_embedding' in obs:
        rollouts.obs['function_LLM_embedding'][0].copy_(obs['function_LLM_embedding'])
    rollouts.obs['junk_repeat_ratio'][0].copy_(obs['junk_repeat_ratio'])
    rollouts.obs['available_actions_mask'][0].copy_(obs['available_actions_mask'])

    num_updates = train_args.num_updates    # 总更新轮数
    start = time.time()

    for j in range(num_updates):
        # --- A. 采集 Rollout 数据 ---
        for step in range(train_args.storage_num_steps):
            with torch.no_grad():
                # 获取初始隐藏状态
                recurrent_hidden_states_in = (rollouts.recurrent_hidden_states[step], rollouts.recurrent_cell_states[step])
                masks = rollouts.recurrent_masks[step]
                
                # 智能体决策
                res_act = sabre_agent.act(
                    function_PalmTree_embedding=rollouts.obs['function_PalmTree_embedding'][step],
                    function_PalmTree_mask=rollouts.obs['function_PalmTree_mask'][step],
                    function_LLM_embedding=None,
                    recurrent_hidden_states_in=recurrent_hidden_states_in,
                    recurrent_mask_in=masks,
                    junk_repeat_ratio=rollouts.obs['junk_repeat_ratio'][step],
                    available_actions_mask=rollouts.obs['available_actions_mask'][step],
                    return_logits=True
                )
                recurrent_hidden_states_out, action_logits, actions, total_log_probs, avg_entropy, values = res_act

            # 环境交互
            obs, reward, done, infos = envs.step(actions)
            
            # 记录这一批次的平均奖励
            mean_reward = reward.mean().item()
            writer.add_scalar('Analysis/Mean_Reward', mean_reward, j)

            # 存入存储器
            rollouts.insert(obs, recurrent_hidden_states_out, actions, total_log_probs, values, reward, done)

        # --- B. 更新模型参数 ---
        val_loss, act_loss, ent = leaner.update(rollouts)
        rollouts.after_update()
        
        # --- C. 实时记录数据到 TensorBoard ---
        total_num_steps = (j + 1) * env_args.num_processes * train_args.storage_num_steps
        
        # 将损失值写入 TensorBoard
        writer.add_scalar('Loss/Value_Loss', val_loss, j)
        writer.add_scalar('Loss/Action_Loss', act_loss, j)
        writer.add_scalar('Analysis/Entropy', ent, j)
        
        # 记录每步的平均消耗时间（吞吐量监控）
        if j > 0:
            avg_cost_per_step = (time.time() - start) / total_num_steps
            writer.add_scalar('Performance/Cost_per_Step', avg_cost_per_step, j)
        
        # --- D. 日志与保存逻辑 ---
        if j % 10 == 0:
            end = time.time()
            logger.info(f"[Main]: Update {j}, Steps {total_num_steps}, Losses: Val {val_loss:.3f}, Act {act_loss:.3f}, Entropy {ent:.3f}, Cost {end - start}")
            torch.save(sabre_agent.state_dict(), os.path.join(train_args.checkpoints_save_path, f"sabre_agent_update_{j}.pth"))

    # 训练结束，关闭 writer
    writer.close()
    logger.info(f'[Main]: ============= Training over, servers shutdown =============')
    for p in procs: p.terminate()

if __name__ == "__main__":
    main()