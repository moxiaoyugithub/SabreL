import pandas as pd
import os

# --- 配置 ---
CSV_FILE = "heartbleed_detailed_results.csv"
TARGET_FUNC = "tls1_process_heartbeat"
MODELS = ['asm2vec', 'BinCola', 'CLAP', 'jTrans', 'safe']
TOP_K_LIST = [1, 5, 15, 30, 50, 100]

def get_accuracies(df_clean, model, phase):
    model_df = df_clean[(df_clean['model'] == model) & (df_clean['phase'] == phase)]
    valid_configs = model_df['config'].unique()
    num_configs = len(valid_configs)
    
    if num_configs == 0:
        return {k: 0.0 for k in TOP_K_LIST}

    accs = {}
    for k in TOP_K_LIST:
        success = 0
        for config in valid_configs:
            config_data = model_df[model_df['config'] == config]
            top_k = config_data.sort_values(by='similarity', ascending=False).head(k)
            if TARGET_FUNC in top_k['candidate_func'].values:
                success += 1
        accs[k] = success / num_configs
    return accs

def main():
    if not os.path.exists(CSV_FILE):
        print("% Error: CSV file not found")
        return

    df = pd.read_csv(CSV_FILE)
    df['similarity'] = pd.to_numeric(df['similarity'], errors='coerce')
    df_clean = df[df['similarity'] > 0].copy()

    # --- 开始打印 LaTeX 代码 ---
    print("\n% ------ LaTeX Table Code Starts Here ------")
    print(r"\begin{table}[htbp]")
    print(r"  \centering")
    print(r"  \caption{Comparison of Search Accuracy (Recall@K) for Heartbleed Vulnerability on Baseline and Obfuscated Datasets}")
    print(r"  \label{tab:vulnerability_search}")
    print(r"  \resizebox{\textwidth}{!}{")
    print(r"  \begin{tabular}{ll" + "c" * len(TOP_K_LIST) + "}")
    print(r"    \toprule")
    print(r"    \textbf{Model} & \textbf{Scenario} & " + " & ".join([f"\\textbf{{T-{k}}}" for k in TOP_K_LIST]) + r" \\")
    print(r"    \midrule")

    for i, model in enumerate(MODELS):
        b_accs = get_accuracies(df_clean, model, 'baseline')
        o_accs = get_accuracies(df_clean, model, 'obfuscated')
        
        # 使用 multirow 增强排版效果
        print(f"    \\multirow{{2Lower}}{{*}}{{{model}}}")
        
        # Baseline 行
        b_vals = " & ".join([f"{b_accs[k]:.3f}" for k in TOP_K_LIST])
        print(f"    & Baseline & {b_vals} \\\\")
        
        # Obfuscated 行
        o_vals = " & ".join([f"{o_accs[k]:.3f}" for k in TOP_K_LIST])
        print(f"    & Obfuscated & \\textbf{{{o_vals}}} \\\\")
        
        # 在模型间添加微弱分割线，最后一个模型后不加
        if i < len(MODELS) - 1:
            print(r"    \cmidrule(lr){1-8}")

    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"  }")
    print(r"\end{table}")
    print("% ------ LaTeX Table Code Ends Here ------\n")

if __name__ == "__main__":
    main()