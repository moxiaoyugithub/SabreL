import os
import csv
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import asyncio
import time
import multiprocessing
import numpy as np
from tqdm.asyncio import tqdm

from envs.gym_env.env_arguments import EnvArgs
from envs.score.mix_differ import MixSimilarityServer, MixSimilarityClient
from envs.binary_process_editor.BPE_utils import binary_read
from logs.logger import logger

# --- 配置路径 ---
BASE_DIR = "dataset/bin_openssl-1.0.1c"
REW_DIR = "dataset/rew_bin_openssl-1.0.1c"
TARGET_FUNC = "tls1_process_heartbeat"
COMPILERS = ['gcc', 'clang']
OPT_LEVELS = ['o0', 'o1', 'o2', 'o3']
MODELS = ['asm2vec', 'BinCola', 'CLAP', 'jTrans', 'safe']

def run_mix_similarity_server_process(device, env_num, target_model_types, host, port):
    server = MixSimilarityServer(device, env_num, target_model_types, host, port)
    asyncio.run(server.run())

# 新增：定义详细结果 CSV 的表头
CSV_HEADER = [
    "phase", "config", "compiler", "opt", "target_func", 
    "candidate_func", "model", "similarity"
]
CSV_FILE = "heartbleed_detailed_results.csv"

def save_to_csv(row):
    """将单条对比结果追加到 CSV 文件"""
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_HEADER)
        writer.writerow(row)

async def evaluate_single_config(client, rank, comp, opt, is_obfuscated=True):
    phase_label = "obfuscated" if is_obfuscated else "baseline"
    config_name = f"openssl-1.0.1c-{comp}-{opt}"
    
    orig_so_path = os.path.join(BASE_DIR, config_name, "libssl.so")
    search_so_path = os.path.join(REW_DIR if is_obfuscated else BASE_DIR, config_name, "libssl.so")

    try:
        cfr_orig = binary_read(os.path.dirname(orig_so_path), os.path.dirname(orig_so_path), "libssl.so")
        query_func = cfr_orig.find_function_by_name(TARGET_FUNC, strict=True)
        query_addr = query_func.get_entry_adress()

        cfr_search = binary_read(os.path.dirname(search_so_path), os.path.dirname(search_so_path), "libssl.so")
        search_functions = list(cfr_search._raw_functions_dict.keys())
    except Exception as e:
        logger.error(f"[{config_name}] BPE Read Error: {e}")
        return config_name, None

    model_scores = {m: [] for m in MODELS}
    target_sims = {m: 0.0 for m in MODELS}

    with tqdm(total=len(search_functions), desc=f"[{phase_label}] {config_name}", unit="func", leave=False) as pbar:
        for func_name in search_functions:
            try:
                target_func_obj = cfr_search.find_function_by_name(func_name, strict=True)
                target_addr = target_func_obj.get_entry_adress()
                
                _, sim_dict = await client.compare(
                    orig_so_path, query_addr, TARGET_FUNC,
                    search_so_path, target_addr, func_name,
                    mode='avg'
                )

                for m in MODELS:
                    score = sim_dict.get(m, 0.0) or 0.0
                    model_scores[m].append((func_name, score))
                    if func_name == TARGET_FUNC:
                        target_sims[m] = score
                    
                    # 实时保存到 CSV 以供复盘
                    save_to_csv([phase_label, config_name, comp, opt, TARGET_FUNC, func_name, m, f"{score:.6f}"])

            except Exception:
                continue
            pbar.update(1)

    # 计算该配置的结果
    config_results = {}
    print(f"\n>>> Interim Result for {phase_label} - {config_name}:")
    for m in MODELS:
        sorted_list = sorted(model_scores[m], key=lambda x: x[1], reverse=True)
        top_1_names = [x[0] for x in sorted_list[:1]]
        top_5_names = [x[0] for x in sorted_list[:5]]
        
        res = {
            "top1": 1.0 if TARGET_FUNC in top_1_names else 0.0,
            "top5": 1.0 if TARGET_FUNC in top_5_names else 0.0,
            "sim": target_sims[m]
        }
        config_results[m] = res
        # 实时打印中间得分情况，方便观察模型是否“挂了”或者分数异常
        print(f"  [{m:8s}] Top1: {res['top1']}, Sim: {res['sim']:.4f}")
    
    return config_name, config_results

async def main_flow(host, port):
    # 1. 预先启动好所有 Client
    logger.info("=== Pre-connecting Clients to Server Workers ===")
    clients = [MixSimilarityClient(rank=i, host=host, port=port) for i in range(8)]
    
    # 2. 构造任务参数 (确保包含 rank 索引 i)
    tasks_params = []
    idx = 0
    for comp in COMPILERS:
        for opt in OPT_LEVELS:
            # 【修复点】这里必须存入 4 个值，以对应后面的拆包
            tasks_params.append((clients[idx], idx, comp, opt)) 
            idx += 1

    # 3. 执行 Phase 1
    logger.info("=== Phase 1: Calculating Baseline ===")
    # 这里的 cli, r, c, o 分别对应 client, rank, comp, opt
    baseline_tasks = [evaluate_single_config(cli, r, c, o, is_obfuscated=False) 
                      for cli, r, c, o in tasks_params]
    baseline_raw = await asyncio.gather(*baseline_tasks)
    baseline_dict = dict(baseline_raw)

    # 4. 执行 Phase 2
    logger.info("=== Phase 2: Calculating SabreL Impact ===")
    obf_tasks = [evaluate_single_config(cli, r, c, o, is_obfuscated=True) 
                 for cli, r, c, o in tasks_params]
    obf_raw = await asyncio.gather(*obf_tasks)
    obf_dict = dict(obf_raw)

    return baseline_dict, obf_dict

def generate_heartbleed_latex_tables(base_dict, obf_dict):
    """
    base_dict: { config_name: { model_name: { 'top1': val, 'top5': val, 'sim': val } } }
    obf_dict:  { config_name: { model_name: { 'top1': val, 'top5': val, 'sim': val } } }
    """
    models = ['asm2vec', 'BinCola', 'CLAP', 'jTrans', 'safe']
    compilers = ['gcc', 'clang']
    opts = ['o0', 'o1', 'o2', 'o3']
    configs = [f"openssl-1.0.1c-{c}-{o}" for c in compilers for o in opts]

    # --- Table 1: Search Accuracy Degradation (Delta) ---
    print("\n% ====== TABLE 1: ACCURACY DEGRADATION ======")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Impact on Heartbleed Search Success Rate ($\Delta$ Accuracy)}")
    print(r"\resizebox{\textwidth}{!}{")
    print(r"\begin{tabular}{llcccccccc}")
    print(r"\toprule")
    print(r"\multirow{2}{*}{\textbf{Model}} & \multirow{2}{*}{\textbf{Metric}} & \multicolumn{4}{c}{\textbf{GCC}} & \multicolumn{4}{c}{\textbf{Clang}} \\")
    print(r"\cmidrule(lr){3-6} \cmidrule(lr){7-10}")
    print(r"& & \textbf{O0} & \textbf{O1} & \textbf{O2} & \textbf{O3} & \textbf{O0} & \textbf{O1} & \textbf{O2} & \textbf{O3} \\")
    print(r"\midrule")

    for m in models:
        # 第一行打印模型名占位，第二行留空
        for i, metric in enumerate(['top1', 'top5']):
            row_label = r"$\Delta$Top-1" if metric == 'top1' else r"$\Delta$Top-5"
            
            # 关键点：使用双大括号 {{ }} 转义 LaTeX 的大括号
            model_cell = f"\\multirow{{2Lower}}{{*}}{{{m}}}" if i == 0 else ""
            
            values = []
            for cfg in configs:
                try:
                    delta = obf_dict[cfg][m][metric] - base_dict[cfg][m][metric]
                    values.append(f"{delta:+.2f}")
                except KeyError:
                    values.append("N/A")
            
            print(f"{model_cell} & {row_label} & {' & '.join(values)} \\\\")
        print(r"\midrule")
    
    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{table}")

    # --- Table 2: Similarity Scores (Direct) ---
    print("\n% ====== TABLE 2: DIRECT SIMILARITY SCORES ======")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Direct Similarity Scores (Original vs. Obfuscated)}")
    print(r"\resizebox{\textwidth}{!}{")
    print(r"\begin{tabular}{lcccccccc}")
    print(r"\toprule")
    print(r"\multirow{2}{*}{\textbf{Model}} & \multicolumn{4}{c}{\textbf{GCC}} & \multicolumn{4}{c}{\textbf{Clang}} \\")
    print(r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}")
    print(r"& \textbf{O0} & \textbf{O1} & \textbf{O2} & \textbf{O3} & \textbf{O0} & \textbf{O1} & \textbf{O2} & \textbf{O3} \\")
    print(r"\midrule")

    for m in models:
        values = []
        for cfg in configs:
            try:
                sim = obf_dict[cfg][m]['sim']
                values.append(f"{sim:.3f}")
            except KeyError:
                values.append("N/A")
        print(f"{m} & {' & '.join(values)} \\\\")
    
    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{table}")

if __name__ == "__main__":
    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)
        
    arguments_yaml_path = "arguments_palmtree_obfuscate_heartbleed.yaml"
    env_args = EnvArgs(arguments_yaml_path)
    
    num_parallel = 8 
    host = env_args.mix_similarity_server_host
    port = env_args.mix_similarity_server_port

    # 启动 Server 进程
    server_process = multiprocessing.Process(
        target=run_mix_similarity_server_process, 
        args=(env_args.differ_device, num_parallel, env_args.target_model_types, host, port)
    )
    server_process.start()
    
    # 5个模型加载较慢，且有8个并发 Worker，建议给足 30s+ 加载时间
    logger.info("Waiting for models to fully initialize in server workers...")
    time.sleep(30) 

    try:
        loop = asyncio.get_event_loop()
        # 传入 host 和 port 用于 Client 初始化
        base_res, obf_res = loop.run_until_complete(main_flow(host, port))
        
        # 生成 LaTeX 表格
        generate_heartbleed_latex_tables(base_res, obf_res)
    finally:
        server_process.terminate()
        server_process.join()
    
    logger.info("Evaluation pipeline finished.")