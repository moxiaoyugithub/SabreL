# BCSD Environment Preparation for SabreL

This document describes the similarity model environment setup used by SabreL in `envs/score/`.
It also documents the repository-provided IDA docker helper scripts and how to use them.

## Purpose

SabreL evaluates obfuscation quality against binary code similarity detection (BCSD) models.
These models are managed under `envs/score/` and typically run in dedicated Conda environments.

## Main SabreL Environment

Create and activate the main environment used by the repository:

```bash
conda create -n GBO-main python=3.10.19 -y
conda activate GBO-main
```

Install the core dependencies used by SabreL:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install torch-geometric==2.5.3 scikit-learn==1.3.2 sentencepiece==0.1.99 tqdm==4.66.1 tokenizers==0.22.1 click==8.1.7 transformers==4.57.3 bert-pytorch==0.0.1a4 pyyaml==6.0.3 stable-baselines3==0.10.0
pip install gtirb==2.0.0 gtirb-rewriting==0.3.0 networkx==3.0 pygraphviz==1.11 capstone==5.0.1 keystone-engine==0.9.2 gym==0.17.0 tensorboard==2.20.0 seaborn==0.13.2
```

If you need Jupyter for notebooks:

```bash
conda install jupyter -y
```

## Similarity Model Environments

The repository supports several BCSD backends. Each backend usually has its own Conda environment:

- `asm2vec`
- `CLAP`
- `SAFE`
- `jTrans`
- `BinCola`

### asm2vec

```bash
conda deactivate
conda create -n asm2vec python=3.8.15 -y
conda activate asm2vec
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 \
    -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html -i https://mirrors.bfsu.edu.cn/pypi/web/simple
pip install r2pipe==1.8.8 click==8.1.7 setuptools==65.5.0
```

### CLAP

```bash
conda deactivate
conda create -n CLAP python=3.8.19 -y
conda activate CLAP
pip install torch==1.10.0 --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu111
pip install transformers==4.42.3 click==8.1.7
```

If the above build does not work on your server, try a simpler variant:

```bash
conda deactivate
conda create -n CLAP python=3.8 -y
conda activate CLAP
pip install "numpy<2.0.0"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers huggingface_hub click==8.1.7
```

### SAFE

```bash
conda deactivate
conda create -n safe python=3.7 -y
conda activate safe
pip install torch==1.4.0+cu100 \
    -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html -i https://mirrors.bfsu.edu.cn/pypi/web/simple
pip install capstone==4.0.1 numpy==1.18.1 r2pipe==1.9.2 click==8.1.7
```

### jTrans

```bash
conda deactivate
conda create -n jTrans python=3.8.19 pandas tqdm -y
conda activate jTrans
pip install torch==1.10.0+cu111 \
    -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html -i https://mirrors.bfsu.edu.cn/pypi/web/simple
pip install numpy==1.24.3 tokenizers==0.20.0 simpletransformers==0.70.1 networkx==3.1 pyelftools==0.31 click==8.1.7
```

### BinCola

```bash
conda deactivate
conda create -n BinCola python=3.8.20 -y
conda activate BinCola
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 \
    -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html -i https://mirrors.bfsu.edu.cn/pypi/web/simple
pip install tensorboard numpy pandas coloredlogs matplotlib PyYAML seaborn scikit-learn tqdm info-nce-pytorch click
```

## IDA and ida_docker Support

SabreL includes repository helper scripts for IDA-based analysis containers:

- `ida_docker_run.sh`
- `ida_docker_stop.sh`

These scripts are used to launch and remove Docker containers named with the prefix `ida_server_`.
The container image is expected to be `ida-base:v2-with-license` and the repository is mounted into `/SABRE`.

### Example usage

```bash
cd /data/mxy/SabreL
bash ida_docker_run.sh /data/mxy/SabreL 8 "[BinCola, CLAP, jTrans]"
```

This command creates up to 8 containers for each target model type.

To stop and remove all IDA containers created by the helper script:

```bash
bash ida_docker_stop.sh
```

### IDA Pro notes

If you install IDA Pro manually, Ubuntu 24.04 may require Python 3.8 for the IDA plugin support.
You can use the `deadsnakes` PPA and `idapyswitch` to select Python 3.8 for IDA:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.8-full python3.8-dev libpython3.8
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
sudo update-alternatives --config python3
./idapyswitch
```

Then run IDA once to confirm it starts:

```bash
./idat
```

## Notes

- Keep the `GBO-main` environment active when running the main SabreL pipeline.
- Use the per-model environments only when testing or generating BCSD model scores.
- If any environment fails due to package compatibility, try adjusting the Python minor version or the CUDA wheel source.
