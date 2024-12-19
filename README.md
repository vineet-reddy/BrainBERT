# BrainBERT

[![Paper](https://img.shields.io/badge/paper-arxiv-b31b1b)](https://arxiv.org/abs/2302.14367)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.12.1+-ee4c2c.svg)](https://pytorch.org/get-started/locally/)

BrainBERT is an modeling approach for learning self-supervised representations of intracranial electrode data. See [paper](https://arxiv.org/abs/2302.14367) for details.

## Quick Start

### Prerequisites
- PyTorch >= 1.12.1
- [PyTorch Gradual Warmup Scheduler](https://github.com/ildoonet/pytorch-gradual-warmup-lr)

### Installation
```
pip install -r requirements.txt
```

### Using Pre-trained Models
1. Download pre-trained weights from [here](https://drive.google.com/file/d/14ZBOafR7RJ4A6TsurOXjFVMXiVH6Kd_Q/view?usp=sharing)
2. See `notebooks/demo.ipynb` for example usage

## Seizure Detection Pipeline
Our specialized seizure detection pipeline leverages BrainBERT's neural representations for clinical applications.

### Pipeline Steps
1. **Data Processing**: 
   - Extract and filter sEEG channels
   - Create 5-second epochs
   - Apply artifact removal and 60Hz notch filter
   - Resample to 256 Hz

2. **Feature Extraction**:
   - Convert epochs to spectrograms
   - Generate BrainBERT embeddings
   - Create seizure/non-seizure labels

3. **Model Training**:
   - Train logistic regression classifier
   - Generate performance metrics
   - Visualize results (ROC curves, confusion matrices)

#### Key Components
```
seizure/
├── notebooks/
│   ├── preprocess_edf_pipeline.ipynb     # Data preprocessing
│   ├── create_labels.ipynb               # Label generation
│   ├── train_brainbert_logreg.ipynb      # Model training
│   └── brainbert_embed_logreg_analysis.ipynb  # Analysis
└── README.md
```

## Upstream
### BrainBERT pre-training data
The data directory should be structured as:
```
/pretrain_data
  |_manifests
    |_manifests.tsv  <-- each line contains the path to the example and the length
  |_<subject>
    |_<trial>
      |_<example>.npy
```

### BrainBERT pre-training
```
python3 run_train.py +exp=spec2vec ++exp.runner.device=cuda ++exp.runner.multi_gpu=True \
  ++exp.runner.num_workers=64 +data=masked_spec +model=masked_tf_model_large \
  +data.data=/path/to/data ++data.val_split=0.01 +task=fixed_mask_pretrain.yaml \
  +criterion=pretrain_masked_criterion +preprocessor=stft ++data.test_split=0.01 \
  ++task.freq_mask_p=0.05 ++task.time_mask_p=0.05 ++exp.runner.total_steps=500000
```
Example parameters:
```
/path/to/data = /storage/user123/self_supervised_seeg/pretrain_data/manifests
```
