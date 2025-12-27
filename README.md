# GRAVEL: Malicious Domain Detection on Out-of-Distribution Gray Data through Graph Contrastive Learning with Structure Aggregation

## Project Overview

GRAVEL is the official implementation of the paper "Malicious Domain Detection on Out-of-Distribution Gray Data through Graph Contrastive Learning with Structure Aggregation". This project proposes a malicious domain detection method based on graph contrastive learning, specifically designed for detecting out-of-distribution gray data.

## Core Features

- **Graph Contrastive Learning**: Leverages graph structural information for representation learning
- **Structure Aggregation**: Enhances graph representations through structure aggregation mechanisms
- **Out-of-Distribution Detection**: Specialized malicious domain detection for OOD gray data
- **End-to-End Training**: Complete training and inference pipeline
- **Multi-Benchmark Support**: Supports pdns, minta, iochg, and iochg_small datasets

## Project Structure

```
GRAVEL/
├── Pretrain_GRAVEL.py          # Pre-training script
├── Fintune_GRAVEL.py           # Fine-tuning script
├── evaluate_finetune_gravel.py # Model evaluation script
├── finetune_baselines.py       # Baseline model fine-tuning script
├── model/
│   └── GRAVEL.py               # Core model implementation (RelGraphEmbedLayer, EntityClassify)
├── baselines/                  # Baseline model implementations
├── dataset/                    # Dataset directory
│   ├── pdns-dgl/              # PDNS dataset
│   ├── minta-dgl/             # MINTA dataset
│   ├── iochg-dgl/             # IOCHG dataset
│   └── iochg_small-dgl/       # IOCHG small dataset
├── checkpoint/                 # Model checkpoints
│   ├── {benchmark}_pretrained_gravel_model.pt
│   ├── {benchmark}_pretrained_gravel_embed.pt
│   ├── {benchmark}_finetune_model.pt
│   └── {benchmark}_finetune_embed.pt
├── environment.yml             # Conda environment file
└── requirements.txt            # Python dependencies
```

## Installation

### Option 1: Using Conda (Recommended)

```bash
# Create environment from exported file
conda env create -f environment.yml

# Activate environment
conda activate gnn_ghj
```

### Option 2: Manual Installation

```bash
# Create new conda environment
conda create -n gnn_ghj python=3.8
conda activate gnn_ghj

# Install PyTorch (CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install DGL
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html

# Install other dependencies
pip install numpy pandas scikit-learn tqdm
```

## Dataset Preparation

The project supports four benchmarks:
- **pdns**: Passive DNS dataset
- **minta**: MINTA dataset
- **iochg**: IOCHG dataset
- **iochg_small**: IOCHG small dataset

Each dataset should be organized in the following structure:
```
dataset/{benchmark}-dgl/
├── nodes/              # Node files (domain.csv, ip.csv, etc.)
├── edges/              # Edge files (edge_1.csv, edge_2.csv, etc.)
├── feats/              # Feature files (*_feats_rand.pt)
└── ood_*.csv          # Out-of-distribution test data
```

## Usage

### 1. Pre-training

Pre-train the GRAVEL model on a specific benchmark:

```bash
python Pretrain_GRAVEL.py \
    --benchmark pdns \
    --gpu 0 \
    --n-epochs 100 \
    --lr 1e-3 \
    --batch-size 4096 \
    --n-hidden 64 \
    --n-layers 3
```

**Key Parameters:**
- `--benchmark`: Dataset to use (pdns, minta, iochg, iochg_small)
- `--gpu`: GPU device ID
- `--n-epochs`: Number of training epochs
- `--lr`: Learning rate
- `--batch-size`: Batch size for training
- `--n-hidden`: Hidden dimension size
- `--n-layers`: Number of GNN layers
- `--fanout`: Neighbor sampling fanout (default: "20,20,20")
- `--pseudo-update-freq`: Frequency of pseudo-label updates (default: 5)
- `--confidence-threshold`: Confidence threshold for pseudo-labeling (default: 0)

**Output:**
- Model files saved to `checkpoint/{benchmark}_pretrained_gravel_model.pt`
- Embedding files saved to `checkpoint/{benchmark}_pretrained_gravel_embed.pt`

### 2. Fine-tuning

Fine-tune the pre-trained model on a specific benchmark:

```bash
python Fintune_GRAVEL.py \
    --benchmark pdns \
    --gpu 0 \
    --n-epochs 100 \
    --lr 1e-3 \
    --batch-size 4096 \
    --shot-ratio 1.0
```

**Key Parameters:**
- `--benchmark`: Dataset to use (pdns, minta, iochg, iochg_small)
- `--gpu`: GPU device ID
- `--n-epochs`: Number of training epochs
- `--lr`: Learning rate
- `--batch-size`: Batch size for training
- `--shot-ratio`: Ratio of OOD malicious samples to use (0.0-1.0, default: 1.0)
- `--fanout`: Neighbor sampling fanout (default: "20,20,20")

**Output:**
- Model files saved to `checkpoint/{benchmark}_finetune_model.pt`
- Embedding files saved to `checkpoint/{benchmark}_finetune_embed.pt`

### 3. Evaluation

Evaluate pre-trained or fine-tuned models:

```bash
# Evaluate pre-trained model
python evaluate_finetune_gravel.py \
    --benchmark pdns \
    --model-type pretrain \
    --eval-batch-size 1024

# Evaluate fine-tuned model
python evaluate_finetune_gravel.py \
    --benchmark pdns \
    --model-type finetune \
    --eval-batch-size 1024
```

**Key Parameters:**
- `--benchmark`: Dataset to evaluate (pdns, minta, iochg, iochg_small)
- `--model-type`: Model type to evaluate (pretrain or finetune)
- `--eval-batch-size`: Batch size for evaluation (default: 1024)
- `--fanout`: Neighbor sampling fanout (default: "20,20,20")
- `--num-workers`: Number of dataloader workers (default: 4)

**Output:**
- Validation and test set metrics (F1, Precision, Recall, AUC, Accuracy)

## Model Architecture

The GRAVEL model consists of two main components:

1. **RelGraphEmbedLayer**: Embeds heterogeneous graph nodes into a unified feature space
2. **EntityClassify**: Classifies nodes using multi-layer RelGraphConv with attention mechanisms

Key features:
- Multi-head attention mechanism
- Pseudo-labeling for semi-supervised learning
- Label propagation for graph structure enhancement
- Center loss for clustering

## Configuration

### GPU Settings

GPU device can be configured in each script:
- `Pretrain_GRAVEL.py`: Line 5, `os.environ['CUDA_VISIBLE_DEVICES']='2'`
- `Fintune_GRAVEL.py`: Line 3, `os.environ['CUDA_VISIBLE_DEVICES']='7'`
- `evaluate_finetune_gravel.py`: Line 7, `os.environ['CUDA_VISIBLE_DEVICES']='2'`

Or use the `--gpu` argument when running the scripts.

## Requirements

- Python 3.8+
- PyTorch 1.13+ (with CUDA 11.8 support)
- DGL 1.0+
- NumPy, Pandas
- scikit-learn
- tqdm

See `environment.yml` or `requirements.txt` for complete dependency list.

## Citation

If you use GRAVEL, please cite our paper:

```bibtex
@article{gravel2024,
  title={Malicious Domain Detection on Out-of-Distribution Gray Data through Graph Contrastive Learning with Structure Aggregation},
  author={[Author Information]},
  journal={[Journal Information]},
  year={2024}
}
```

## License

[Add license information here]

## Contributing

We welcome issues and pull requests to improve this project.

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `--batch-size` or `--eval-batch-size`
2. **Module not found**: Ensure conda environment is activated and dependencies are installed
3. **Dataset not found**: Check dataset paths in `dataset/` directory
4. **Model file not found**: Ensure pre-training is completed before fine-tuning

### File Path Issues

All file paths in the codebase use relative paths based on `SCRIPT_DIR`. If you encounter path issues:
- Ensure you run scripts from the project root directory
- Check that dataset files are in the correct locations
- Verify checkpoint directory exists
