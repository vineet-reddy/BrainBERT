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

## Core Features

### 1. Neural Data Processing
- Supports multiple neural data formats (EDF, HDF5)
- Built-in preprocessing pipeline for intracranial recordings
- Automatic electrode validation and artifact removal
- Laplacian re-referencing support

### 2. Model Architecture
- Transformer-based architecture optimized for neural signals
- Multiple training objectives:
  - Masked pretraining (primary pipeline)
  - Binary classification
  - Feature extraction
  - Fine-tuning
  - Wav2Vec-style processing

### 3. Seizure Detection Pipeline
Our specialized seizure detection pipeline leverages BrainBERT's neural representations for clinical applications.

#### Pipeline Steps
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

## Development

### Data Processing Pipeline
```bash
# Set up directory structure
python3 -m data.create_data_dirs +data=pretraining +hydra.job.chdir=False
```

2. Data Conversion:
```bash
# Convert EDF files to HDF5
python3 -m data.edf2h5 [options]

# Write preprocessed data
python3 -m data.write_preprocessed_inputs +data=tf_unmasked +data_prep=cwt_to_disk
```

3. Pretraining Data Preparation:
```bash
# Create wav format data for pretraining
python3 -m data.write_pretrain_data_wavs +data=pretraining_template.yaml +data_prep=write_pretrain_split

# Modify manifests if needed
python3 -m data.modify_manifest +data=pretrain_wavs_from_disk
```

4. Create Aligned Data Caches:
```bash
python3 -m data.make_aligned_data_caches [options]
```

The pipeline supports both linguistic and non-linguistic neural data processing, with specialized handling for speech vs. non-speech analysis. All utilities use Hydra for configuration management.

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
