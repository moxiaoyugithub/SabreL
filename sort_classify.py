import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- 配置 ---
MODELS = ['asm2vec', 'BinCola', 'CLAP', 'jTrans', 'safe']
BASE_EMB_DIR = "dataset/emb_sort"
REW_EMB_DIR = "dataset/rew_emb_sort"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
BATCH_SIZE = 16
NUM_RUNS = 50 

LABEL_MAP = {
    'bubble_sort': 0, 'insertion_sort': 1, 'selection_sort': 2,
    'shell_sort': 3, 'quick_sort': 4, 'merge_sort': 5,
    'heap_sort': 6, 'radix_sort': 7, 'bucket_sort': 8, 'counting_sort': 9
}

def load_dataset(root_dir, model_name):
    X, y = [], []
    model_path = os.path.join(root_dir, model_name)
    if not os.path.exists(model_path):
        return None, None
    for func_name in os.listdir(model_path):
        if func_name in LABEL_MAP:
            func_dir = os.path.join(model_path, func_name)
            for file in os.listdir(func_dir):
                if file.endswith(('.pkl', '.npy')):
                    with open(os.path.join(func_dir, file), 'rb') as f:
                        X.append(pickle.load(f))
                    y.append(LABEL_MAP[func_name])
    return np.array(X), np.array(y)

def single_run(X_train, y_train, X_test, y_test):
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_tensor = torch.FloatTensor(X_test).to(DEVICE)
    test_labels = torch.LongTensor(y_test).to(DEVICE)

    input_dim = X_train.shape[1]
    classifier = nn.Linear(input_dim, 10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        classifier.train()
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = classifier(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    classifier.eval()
    with torch.no_grad():
        train_out = classifier(torch.FloatTensor(X_train).to(DEVICE))
        orig_acc = (train_out.argmax(1) == torch.LongTensor(y_train).to(DEVICE)).float().mean().item()
        test_out = classifier(test_tensor)
        obf_acc = (test_out.argmax(1) == test_labels).float().mean().item()
        
    return orig_acc, obf_acc

def main():
    results = []
    print(f"Starting Evaluation: {NUM_RUNS} runs per model...")

    for m in MODELS:
        X_tr, y_tr = load_dataset(BASE_EMB_DIR, m)
        X_te, y_te = load_dataset(REW_EMB_DIR, m)
        if X_tr is None or X_te is None: continue

        orig_scores, obf_scores = [], []
        print(f"[Evaluating {m}] ", end="", flush=True)
        for i in range(NUM_RUNS):
            oa, ra = single_run(X_tr, y_tr, X_te, y_te)
            orig_scores.append(oa); obf_scores.append(ra)
            print(".", end="", flush=True)

        results.append({
            'model': m,
            'orig_mean': np.mean(orig_scores), 'orig_std': np.std(orig_scores),
            'obf_mean': np.mean(obf_scores), 'obf_std': np.std(obf_scores)
        })

    # --- 输出 LaTeX 代码 ---
    print("\n\n" + "% " + "="*20 + " LaTeX Code Starts " + "="*20)
    print(r"\begin{table}[htbp]")
    print(r"  \centering")
    print(r"  \caption{Comparison of Classification Accuracy on Original and Obfuscated Embeddings (Average of " + str(NUM_RUNS) + r" Runs)}")
    print(r"  \label{tab:sort_classification}")
    print(r"  \begin{tabular}{lcc}")
    print(r"    \toprule")
    print(r"    \textbf{Model} & \textbf{Original Acc (\%)} & \textbf{Obfuscated Acc (\%)} \\")
    print(r"    \midrule")
    
    for res in results:
        m_name = res['model']
        orig = f"{res['orig_mean']*100:.2f} \pm {res['orig_std']*100:.2f}"
        obf = f"{res['obf_mean']*100:.2f} \pm {res['obf_std']*100:.2f}"
        # 将混淆后的结果加粗，突出对比
        print(f"    {m_name} & ${orig}$ & \\textbf{{${obf}$}} \\\\")
    
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")
    print("% " + "="*20 + " LaTeX Code Ends " + "="*20 + "\n")

if __name__ == "__main__":
    main()