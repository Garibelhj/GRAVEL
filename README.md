# GRAVEL: Malicious Domain Detection on Out-of-Distribution Gray Data through Graph Contrastive Learning with Structure Aggregation

## Project Overview

GRAVEL is the official implementation of the paper "Malicious Domain Detection on Out-of-Distribution Gray Data through Graph Contrastive Learning with Structure Aggregation". This project proposes a malicious domain detection method based on graph contrastive learning, specifically designed for detecting out-of-distribution gray data.

## Core Features

- **Graph Contrastive Learning**: Leverages graph structural information for representation learning
- **Structure Aggregation**: Enhances graph representations through structure aggregation mechanisms
- **Out-of-Distribution Detection**: Specialized malicious domain detection for OOD gray data
- **End-to-End Training**: Complete training and inference pipeline

## Project Structure

```
GRAVEL/
├── GRAVEL_GraphEncoder.py      # Graph encoder implementation
├── GRAVEL_endtoend.py          # End-to-end training script
├── gravel_backbone.py          # Backbone network implementation
├── contrastive_learning.py     # Contrastive learning implementation
├── fintune_GRAVEL.py           # GRAVEL fine-tuning script
├── finetune_baselines.py       # Baseline model fine-tuning script
├── evaluate_finetune_gravel.py # Evaluation script
├── split_dataset.py            # Dataset splitting script
├── baselines/                  # Baseline model implementations
├── dataset/                    # Dataset directory
├── checkpoint/                 # Model checkpoints
├── feats/                      # Feature files
└── plot/                       # Visualization charts
```

## Main Features

- **Graph Encoder**: Implements GNN-based graph representation learning
- **Contrastive Learning**: Supports multiple contrastive learning strategies
- **End-to-End Training**: Complete training pipeline
- **Model Fine-tuning**: Supports pre-trained model fine-tuning
- **Baseline Comparison**: Comparison with multiple baseline models
- **OOD Detection**: Malicious domain detection for out-of-distribution data

## Requirements

- Python 3.7+
- PyTorch
- PyTorch Geometric
- DGL (Deep Graph Library)
- Other dependencies see requirements.txt

## Usage

1. Install dependencies
2. Prepare dataset
3. Run training script
4. Evaluate model performance

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