import os
import warnings

warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import sys
import json
import time
import pickle
import asyncio
import multiprocessing
import numpy as np
from tqdm import tqdm

# 确保能找到环境相关的包
sys.path.append("..")
from envs.score.mix_embedder import MixEmbedderServer, MixEmbedderClient
from envs.binary_process_editor.BPE_utils import binary_read
from logs.logger import logger

# --- 1. 配置路径与参数 ---
BIN_ORIG_ROOT = 'dataset/bin_sort'
BIN_REW_ROOT = 'dataset/rew_bin_sort'
GTIRB_ROOT = 'gtirb_sort/'  # 假设 GTIRB 文件统一存放在此

EMB_BASE_DIR = 'dataset/emb_sort'
REW_EMB_BASE_DIR = 'dataset/rew_emb_sort'

# 10 个目标排序函数（分类标签）
TARGET_FUNCTIONS = [
    'bubble_sort', 'selection_sort', 'insertion_sort', 'shell_sort', 
    'partition', 'quick_sort_recursive', 'heapify', 'heap_sort', 
    'comb_sort', 'gnome_sort'
]

# 待测试的相似度模型
MODELS = ['asm2vec', 'BinCola', 'CLAP', 'jTrans', 'safe']

# --- 2. 辅助函数：创建目录结构 ---
def prepare_dirs(base_path, models, functions):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    for m in models:
        model_path = os.path.join(base_path, m)
        for f in functions:
            final_path = os.path.join(model_path, f)
            os.makedirs(final_path, exist_ok=True)

# --- 3. 后端服务进程 ---
def run_server_process(device, env_num, target_model_types, host, port):
    server = MixEmbedderServer(device, env_num, target_model_types, host, port)
    asyncio.run(server.run())

# --- 4. 核心提取逻辑 ---
async def extract_embeddings_flow(bin_root, output_base_dir, is_rew=False):
    """
    遍历二进制目录，提取所有函数的 Embedding 并按规则保存
    """
    # 建立与后端的长连接 (使用 rank 0 即可，因为是顺序批处理)
    client = MixEmbedderClient(rank=0, port=6000)
    
    bin_files = [f for f in os.listdir(bin_root) if os.path.isfile(os.path.join(bin_root, f))]
    bin_files = [f for f in bin_files if not f.endswith(('.c', '.sh'))]

    logger.info(f"[*] Found {len(bin_files)} binaries in {bin_root}. Starting extraction...")

    for bin_name in tqdm(bin_files, desc=f"Processing {'Obfuscated' if is_rew else 'Original'}"):
        bin_path = os.path.join(bin_root, bin_name)
        
        try:
            # 1. 使用 BPE 解析二进制获取地址
            cfr = binary_read(bin_root, GTIRB_ROOT, bin_name)
            
            for func_name in TARGET_FUNCTIONS:
                func_obj = cfr.find_function_by_name(func_name, strict=True)
                if not func_obj:
                    continue
                
                addr = func_obj.get_entry_adress()
                
                # 2. 调用客户端获取混合 Embedding 字典
                # 返回格式: {'asm2vec': np.array, 'CLAP': np.array, ...}
                embeddings = await client.get_embeddings(bin_path, addr, func_name)
                
                # 3. 按规则保存文件: output_base_dir/模型名/函数名/sort-xxx.pkl
                for model_name, vector in embeddings.items():
                    if vector is None: continue
                    
                    save_dir = os.path.join(output_base_dir, model_name, func_name)
                    save_path = os.path.join(save_dir, f"{bin_name}.pkl")
                    
                    with open(save_path, 'wb') as f:
                        pickle.dump(vector, f)
                        
        except Exception as e:
            logger.error(f"Error processing {bin_name}: {e}")

async def main():
    # A. 准备目录
    prepare_dirs(EMB_BASE_DIR, MODELS, TARGET_FUNCTIONS)
    prepare_dirs(REW_EMB_BASE_DIR, MODELS, TARGET_FUNCTIONS)

    # B. 启动长效后端 (多进程)
    # env_num 设置为 1 即可，因为我们采用顺序 Client 调用
    num_parallel = 1 
    server_proc = multiprocessing.Process(target=run_server_process, args=(
        'cuda', num_parallel, MODELS, '127.0.0.1', 6000))
    server_proc.start()
    
    logger.info("Waiting for models to load (30s)...")
    time.sleep(30) 

    try:
        # C. 提取原始版本嵌入
        logger.info("=== Phase 1: Original Binaries ===")
        await extract_embeddings_flow(BIN_ORIG_ROOT, EMB_BASE_DIR, is_rew=False)

        # D. 提取混淆版本嵌入
        logger.info("=== Phase 2: Obfuscated Binaries ===")
        await extract_embeddings_flow(BIN_REW_ROOT, REW_EMB_BASE_DIR, is_rew=True)

    finally:
        server_proc.terminate()
        server_proc.join()
        logger.info("[SUCCESS] All embeddings extracted and saved.")

if __name__ == "__main__":
    asyncio.run(main())