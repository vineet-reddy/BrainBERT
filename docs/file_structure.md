# BrainBERT Repository Structure

BrainBERT is a self-supervised learning framework for intracranial electrode data. This document outlines the repository structure and provides guidance for extending its capabilities.

## Core Components

### Configuration (`/conf`)
Centralized configuration using Hydra for models, data, and experiments.
```bash
python run_train.py +exp=spec2vec +model=masked_tf_model_large
```

### Data Processing (`/data`)
Pipeline for converting intracranial recordings from EDF to HDF5 format.
- **Core Processing:** `edf2h5.py`, `h5_data.py`, `trial_data.py`
- **Author-Specific:** `corrupted_elec.json`, `test_split_trials.json`

### Datasets (`/datasets`)
Model-ready dataset implementations.
- `base_tf_dataset.py`: Base dataset architecture
- `masked_tf_dataset.py`: Self-supervised pretraining
- `finetuning_datasets.py`: Task-specific datasets

### Models (`/models`)
Core BrainBERT implementations.
- `masked_tf_model.py`: Main transformer model
- `base_model.py`: Base model interface
- `transformer_encoder_input.py`: Input encoder

### Preprocessors (`/preprocessors`)
Signal transformation utilities.
- Time-frequency: `stft.py`, `morelet_preprocessor.py`
- Raw signal: `wav_preprocessor.py`

### Training Components

#### Criterions (`/criterions`)
Loss functions for different training objectives.
- `pretrain_masked_criterion.py`: Masked pretraining
- `finetune_criterion.py`: Fine-tuning
- `feature_extract_criterion.py`: Feature extraction

#### Tasks (`/tasks`)
Training task implementations.
- `spec_pretrain.py`: Spectrogram pretraining
- `finetune_task.py`: Fine-tuning
- `feature_extract_task.py`: Feature extraction

#### Schedulers (`/schedulers`)
Learning rate optimization strategies.
- `ramp_up.py`: Warmup and step-down scheduling
- `reduce_on_plateau.py`: Plateau-based reduction

### Analysis & Testing

#### Testing (`/testing`)
Model evaluation and analysis tools.
- **Model Analysis:** `effective_dimensionality.py`, `collect_dataset_stats.py`
- **Few-shot Learning:** `run_fewshot_training_tests.py`, `select_fewshot_learning_electrode.py`

#### Utilities (`/util`)
Core functionality used across the codebase.
- `mask_utils.py`: Self-supervised masking strategies
- `tensorboard_utils.py`: Training visualization

### Clinical Applications (`/seizure`)
Seizure detection tools and analysis.
- **Pipeline:** `preprocess_edf_pipeline.py`, `create_labels.py`
- **Analysis:** `train_brainbert_logreg.py`, `predict_seizures_from_edf.ipynb`

### Examples (`/notebooks`)
Interactive demonstrations and tutorials.
- `demo.ipynb`: Basic usage examples
- Example data for testing

## Extending BrainBERT

### 1. Data Integration
1. Replace author-specific files with your dataset configuration
2. Adapt data loaders to your format
3. Configure preprocessing pipeline

### 2. Model Customization
1. Extend `masked_tf_model.py` for architecture changes
2. Create custom criterion if needed
3. Configure model parameters in `/conf/model/`

### 3. Training Setup
1. Create new task by extending `base_task.py`
2. Implement custom learning rate schedule if needed
3. Configure training parameters

### 4. Analysis & Validation
1. Add test scripts for your use case
2. Implement custom masking strategies if needed
3. Create analysis scripts for validation

## Quick Start

1. **Setup**
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Preparation**
   - Configure paths in `/conf/data/`
   - Run `create_data_dirs.py`

3. **Training**
   ```bash
   python run_train.py +exp=spec2vec +model=masked_tf_model_large
   ```

4. **Validation**
   ```bash
   python run_tests.py
   ```

