import os
import warnings

warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import csv
import time
import asyncio
import pandas as pd
import multiprocessing
import numpy as np
from tqdm.asyncio import tqdm

from envs.gym_env.env_arguments import EnvArgs
from envs.score.mix_differ import MixSimilarityServer, MixSimilarityClient
from envs.binary_process_editor.BPE_utils import binary_read
from logs.logger import logger

# --- 配置 ---
INPUT_CSV = "heartbleed_detailed_results.csv"
OUTPUT_CSV = "heartbleed_detailed_results_REPAIRED.csv"
BASE_DIR = "dataset/bin_openssl-1.0.1c"
REW_DIR = "dataset/rew_bin_openssl-1.0.1c"
TARGET_FUNC = "tls1_process_heartbeat"

def run_mix_similarity_server_process(device, env_num, target_model_types, host, port):
    server = MixSimilarityServer(device, env_num, target_model_types, host, port)
    asyncio.run(server.run())

async def repair_flow(host, port):
    if not os.path.exists(INPUT_CSV):
        logger.error(f"❌ 找不到原始文件 {INPUT_CSV}")
        return

    # 1. 读取数据
    df = pd.read_csv(INPUT_CSV)
    df['similarity'] = pd.to_numeric(df['similarity'], errors='coerce')
    
    normal_df = df[df['similarity'] != 0.0]
    zero_df = df[df['similarity'] == 0.0]
    
    logger.info(f"📊 数据状态 -> 正常: {len(normal_df)} | 待修复: {len(zero_df)}")

    # 初始化新文件
    normal_df.to_csv(OUTPUT_CSV, index=False)
    
    # 2. 预启动客户端
    logger.info("🔗 正在建立与模型后端的连接...")
    clients = [MixSimilarityClient(rank=i, host=host, port=port) for i in range(8)]
    
    # 3. 按组修复
    grouped = zero_df.groupby(['phase', 'config'])
    total_groups = len(grouped)
    
    logger.info(f"🚀 开始修复任务，共 {total_groups} 个二进制配置组")

    with open(OUTPUT_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        
        group_idx = 1
        for (phase, config), g_data in grouped:
            is_obf = (phase == 'obfuscated')
            orig_so = os.path.join(BASE_DIR, config, "libssl.so")
            search_so = os.path.join(REW_DIR if is_obf else BASE_DIR, config, "libssl.so")
            
            logger.info(f"[{group_idx}/{total_groups}] 正在解析组: {config} ({phase})")
            
            try:
                cfr_orig = binary_read(os.path.dirname(orig_so), os.path.dirname(orig_so), "libssl.so")
                query_addr = cfr_orig.find_function_by_name(TARGET_FUNC, strict=True).get_entry_adress()
                print('query_addr:', query_addr)
                cfr_search = binary_read(os.path.dirname(search_so), os.path.dirname(search_so), "libssl.so")
            except Exception as e:
                logger.error(f"   ⚠️ 解析二进制失败: {e}")
                continue

            # 遍历组内需要修复的条目
            for i, (_, row) in enumerate(g_data.iterrows()):
                candidate_func = row['candidate_func']
                model_name = row['model']
                
                # 简单的负载均衡：根据索引分配 client
                rank = i % 8
                client = clients[rank] 
                
                try:
                    target_func_obj = cfr_search.find_function_by_name(candidate_func, strict=True)
                    target_addr = target_func_obj.get_entry_adress()
                    
                    # 调试打印：发送请求前
                    # print(f"   [DEBUG] 请求对比: {TARGET_FUNC} vs {candidate_func} (Model: {model_name})")
                    
                    _, sim_dict = await client.compare(
                        orig_so, query_addr, TARGET_FUNC,
                        search_so, target_addr, candidate_func,
                        mode='avg'
                    )
                    
                    new_sim = sim_dict.get(model_name, 0.0)
                    
                    # 实时调试输出
                    if new_sim > 0:
                        print(f"   ✅ 修复成功: {candidate_func} | {model_name} -> {new_sim:.6f}")
                    else:
                        print(f"   ❓ 修复结果仍为0: {candidate_func} | {model_name}")

                    writer.writerow([
                        phase, config, row['compiler'], row['opt'], 
                        TARGET_FUNC, candidate_func, model_name, f"{new_sim:.6f}"
                    ])
                    
                except Exception as e:
                    print(f"   ❌ 运行时错误 ({candidate_func}): {e}")
                    continue
            
            group_idx += 1
            # 每个配置组完成后刷新磁盘缓存
            f.flush()

if __name__ == "__main__":
    arguments_yaml_path = "arguments_palmtree_obfuscate_heartbleed.yaml"
    env_args = EnvArgs(arguments_yaml_path)
    
    host = env_args.mix_similarity_server_host
    port = env_args.mix_similarity_server_port
    num_parallel = 8 

    logger.info("🛠️ 正在启动模型服务端进程...")
    server_process = multiprocessing.Process(
        target=run_mix_similarity_server_process, 
        args=(env_args.differ_device, num_parallel, env_args.target_model_types, host, port)
    )
    server_process.start()
    
    # 给模型加载预留时间
    logger.info("⏳ 等待模型加载 (30s)...")
    time.sleep(30)

    try:
        asyncio.run(repair_flow(host, port))
    finally:
        server_process.terminate()
        server_process.join()
        logger.info("🏁 修复脚本执行结束。")