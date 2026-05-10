# SabreL

SabreL is a reinforcement learning framework for similarity-aware binary obfuscation against deep learning-based binary code similarity detection (BCSD) models. It learns obfuscation policies that jointly decide:

- what transformation to apply,
- where to apply it, and
- how to instantiate its parameters.

The current implementation supports both:

- `shadow` training, which uses a lightweight text-based approximation environment for efficient policy learning, and
- `entity` evaluation / deployment, which performs real binary rewriting through GTIRB-based infrastructure.

The repository contains training scripts, evaluation pipelines, batch obfuscation utilities, dataset preparation scripts, and figure-generation code used in the paper.

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/moxiaoyugithub/SabreL.git
   cd SabreL
   ```

2. Download the excluded files archive from Google Drive[https://drive.google.com/file/d/1tVtO7fGTvrTn5fyHgMMdk4Jl8yhIyyRJ/view?usp=sharing] and extract it:
   ```bash
   # Download sabrel_excluded.tar.gz (19GB compressed)
   tar -xzf sabrel_excluded.tar.gz
   ```
   
   This archive contains large datasets, model checkpoints, and other files excluded from the Git repository for size reasons. After extraction, the project structure will be complete and ready for use.

3. Follow the detailed setup instructions in:
   - `GTIRB_tools_prepare.md` — for GTIRB, GTIRB pretty printer, and ddisasm
   - `envs/score/BCSD_prepare.md` — for BCSD model environments

## Features

- Multi-head RL policy for binary obfuscation
- Hierarchical decision-making over action type, basic block, instruction location, and action-specific parameters
- Support for multiple BCSD backends, including `asm2vec`, `SAFE`, `CLAP`, `jTrans`, and `BinCola`
- Approximate training mode for low-cost learning
- Real rewriting mode for executable binary generation
- Evaluation pipelines for similarity reduction, vulnerability retrieval, and function classification

## Repository Structure

```text
.
├── algo/                      # PPO optimizer
├── arch/                      # Perception, core network, decision heads, critic
│   ├── center/
│   ├── critic/
│   ├── decider/
│   └── perceptor/
├── buffer/                    # Rollout storage
├── dataset/                   # Dataset builders, samplers, and prepared index files
├── envs/                      # RL environment, obfuscation primitives, similarity backends
├── logs/                      # TensorBoard and runtime logs
├── paper_figures/             # Exported figures for the paper
├── train_llm.py               # Main training entry for LLM-assisted SabreL
├── train_palmtree.py          # Main training entry for PalmTree-only SabreL
├── test.py                    # Evaluation pipeline
├── demo.py                    # Single-sample interactive demo
├── run_obfuscate.py           # Batch obfuscation entry point
├── plot_rollout_analysis.py   # Academic-style rollout analysis plots
├── make_loss_curves.py        # TensorBoard loss-curve plotting
└── compare_shadow_entity.py   # Real-vs-approximate comparison plot
```

## Core Workflow

SabreL follows a closed-loop workflow:

1. Parse a target binary into an editable representation.
2. Encode the function into multi-grained embeddings.
3. Use the policy network to select an obfuscation action.
4. Apply the action in either shadow mode or entity mode.
5. Query BCSD models to obtain similarity-based reward.
6. Optimize the policy with PPO.

For training efficiency, shadow mode approximates rewriting on tokenized assembly sequences. For final testing and deployment, entity mode performs actual binary rewriting.

## Requirements

The project assumes a Linux environment with CUDA-capable GPUs. The codebase was developed and evaluated with:

- Python 3.10
- PyTorch
- TensorBoard
- pandas
- seaborn
- matplotlib
- GTIRB-related tooling
- GCC / Clang toolchains

The project also depends on local similarity-model components under `envs/score/`.

For detailed setup instructions, see:

- `GTIRB_tools_prepare.md` — build and install GTIRB, the GTIRB pretty printer, and ddisasm.
- `envs/score/BCSD_prepare.md` — prepare the BCSD model environments and use the repository `ida_docker` helper scripts.

In the current workspace, the scripts were typically executed in conda environments such as:

- `GBO-main`
- `asm2vec`
- `CLAP`
- `jTrans`
- `safe`
- `BinCola`

If you are reproducing the full pipeline, make sure the required model checkpoints and GTIRB-based rewriting dependencies are installed and accessible.

## Configuration

SabreL is configured through YAML files. The main configuration files include:

- `arguments_llm_train.yaml`
- `arguments_llm_test.yaml`
- `arguments_palmtree_train.yaml`
- `arguments_palmtree_test.yaml`
- `arguments_palmtree_demo.yaml`
- `arguments_palmtree_obfuscate_main.yaml`

The configuration structure is split into:

- `env`: environment, dataset, similarity-model, and device settings
- `arch`: model architecture hyperparameters
- `train`: PPO and training-loop hyperparameters

Important fields include:

- `shadow_mode`: whether to use approximate training
- `target_model_types`: BCSD models used by the reward/evaluation module
- `max_blocks`, `max_instructions`: observation truncation limits
- `obf_success_similarity`: success threshold
- `obf_max_inst_growth`: perturbation budget

## Dataset Preparation

The repository already contains multiple prepared dataset index files, such as:

- `dataset/train_5-128.sabredataset`
- `dataset/test_5-128.sabredataset`
- `dataset/shadow_dataset/train_5-128.ssabredataset`
- `dataset/shadow_dataset/test_5-128.ssabredataset`
- `dataset/heartbleed_tls1_process_heartbeat.sabredataset`
- `dataset/sort.sabredataset`

If you want to rebuild datasets from source binaries, start from:

- `dataset/build_json_dataset.py`
- `dataset/sampler.py`
- `dataset/extract_main.py`
- `dataset/gen_shadow_dataset.py`

There is also an existing dataset note at:

- [dataset/readme.md](/data/mxy/GBO/dataset/readme.md)

The paper experiments mainly use binaries compiled from:

- GNU coreutils
- GNU binutils
- OpenSSL

under multiple compilers and optimization levels.

## Training

### LLM-assisted training

```bash
conda run -n GBO-main python train_llm.py
```

This entry:

- starts the embedding server,
- starts the similarity server,
- launches vectorized environments,
- trains the policy with PPO, and
- writes TensorBoard logs to `logs/tensorboard_logs/llm/`.

### PalmTree-only training

```bash
conda run -n GBO-main python train_palmtree.py
```

Use this when you want to disable the LLM branch and train only with PalmTree features.

### Checkpoints

By default, checkpoints are saved under paths specified in the YAML config, such as:

- `checkpoints/with_llm/`
- `checkpoints/only_palmtree/`

## Evaluation

The main evaluation entry is:

```bash
conda run -n GBO-main python test.py
```

This pipeline:

- loads a trained policy,
- evaluates rollouts under shadow or entity mode,
- saves per-step rollout histories,
- exports analysis plots, and
- optionally benchmarks physical overhead in entity mode.

Evaluation results are written to timestamped directories under:

```text
evaluation_results/
```

Typical outputs include:

- `metrics/full_rollout_history.csv`
- `metrics/episodes_history.csv`
- `analysis_plots/`
- copied rewritten binaries in entity mode

## Interactive Demo

For a single-sample step-by-step rollout:

```bash
conda run -n GBO-main python demo.py
```

This is useful for:

- inspecting action decisions,
- checking similarity changes after each step,
- debugging rewritten outputs, and
- understanding the learned policy qualitatively.

## Batch Obfuscation

To apply a trained model to a batch of binaries:

```bash
conda run -n GBO-main python run_obfuscate.py \
  -o output_binaries \
  -c arguments_palmtree_obfuscate_main.yaml \
  -m sabre_agent_update_200.pth
```

Optional flag:

```bash
-l / --ex_lib
```

Use this when rewritten binaries need additional runtime library resolution.

## Analysis and Plotting

Several scripts are provided for generating publication-style figures:

- `make_loss_curves.py`
  - plots training loss curves from TensorBoard logs
- `plot_rollout_analysis.py`
  - produces rollout-level academic plots
- `compare_shadow_entity.py`
  - compares approximate and real execution modes

These scripts usually write outputs to:

- `paper_figures/`
- `evaluation_results/.../analysis_plots/`

## Current Obfuscation Primitives

SabreL currently supports three semantics-preserving primitives:

- junk code insertion
- basic block splitting
- opaque predicate insertion

These are implemented in the obfuscation environment and executed through the rewriting pipeline.

## Similarity Models

The project contains integration code for multiple BCSD models:

- `asm2vec`
- `SAFE`
- `CLAP`
- `jTrans`
- `BinCola`

Training-time reward commonly uses a subset of these models for efficiency, while evaluation can use a larger model set.

## Reproducibility Notes

To reproduce the paper as closely as possible:

1. Use the provided YAML configurations instead of rewriting them from scratch.
2. Train in `shadow_mode=true` for efficiency.
3. Evaluate in real rewriting mode for final metrics.
4. Keep compiler versions, optimization levels, and dataset partitions consistent with the paper setup.
5. Use the same target model set when comparing to reported numbers.

The paper experiments were run on multi-GPU Linux servers with GCC/Clang toolchains and CUDA devices dedicated to:

- the policy network,
- the embedding service, and
- the similarity service.

## Known Limitations

- The full system assumes access to local model backends and rewriting infrastructure.
- Training and evaluation are currently tailored to Linux `x86/64` binaries.
- Completely stripped binaries may be hard to process if function boundaries cannot be recovered reliably.
- Shadow-mode training is an approximation and does not fully reproduce all low-level binary effects.

## Citation

If you use this repository, please cite the SabreL paper once the bibliographic information is finalized.

## License

No repository-wide license file is currently included at the root of this workspace. Add an explicit license before public release if redistribution is intended.
