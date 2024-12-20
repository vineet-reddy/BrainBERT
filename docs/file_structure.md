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
- **Author-Specific-Data:** `corrupted_elec.json`, `test_split_trials.json`

### Datasets (`/datasets`)
Model-ready dataset implementations.
- `base_tf_dataset.py`: Base dataset architecture
- `masked_tf_dataset.py`: Self-supervised pretraining
- `finetuning_datasets.py`: Task-specific datasets

### Models (`/models`)
Core model implementations and architectures.

**Primary BrainBERT Implementation:**
- `masked_tf_model.py`: Main BrainBERT transformer model
  - Core self-supervised learning implementation
  - Handles sequence processing and attention mechanisms
  - Supports intermediate representation extraction
  - **This is the primary model to use**

**Base Classes:**
- `base_model.py`: Abstract base class for all models
  - Defines common interface and weight management
  - Implements save/load functionality

- `transformer_encoder_input.py`: Input encoding layer
  - Transforms raw/preprocessed signals into embeddings
  - Handles positional encoding and dimensionality

**Alternative Implementations:**
- `seeg_wav2vec.py`: Wave2Vec-style model (Experimental)
  - Alternative approach using contrastive learning
  - Specifically for SEEG signals
  - Currently experimental, prefer `masked_tf_model.py`

**Feature Extraction:**
- `feature_extract_model.py`: Base feature extractor
- `feature_extract_deep_model.py`: Deep network extractor
  - Used for extracting learned representations
- `feature_extract_hidden.py`: Hidden layer extraction

**Components:**
- `spec_prediction_head.py`: Prediction head for spectrogram tasks
- `finetune_model.py`: Fine-tuning model adaptations

**Deprecated Models:**
*These models were used in early experiments and are no longer maintained:*
- `linear_wav_baseline.py`: Simple linear baseline
- `linear_spec_baseline.py`: Spectrogram baseline
- `deep_linear_wav_baseline.py`: Deep linear baseline
- `hidden_linear_wav_model.py`: Hidden layer model

**Registration:**
- `__init__.py`: Model registry and factory methods

### Preprocessors (`/preprocessors`)
Signal transformation utilities that convert raw intracranial EEG data into structured representations for neural models. Each preprocessor transforms continuous voltage signals into sequences of time-frequency "frames" that serve as input tokens.

**Core Preprocessors:**
- `stft.py`: Short-Time Fourier Transform preprocessor
  - Computes spectrograms using sliding windows
  - Configurable window size and overlap
  - Provides consistent frequency resolution
  - Outputs time x frequency matrices

- `morelet_preprocessor.py`: Morlet Wavelet Transform
  - Uses Morlet wavelets for adaptive time-frequency decomposition
  - Better handles transient oscillations
  - Improved low-frequency resolution
  - Outputs smooth frequency-domain representations

- `superlet_preprocessor.py`: Superlet Transform (Based on Moca et al., 2021)
  - Advanced wavelet-based approach combining multiple cycle counts
  - Achieves high frequency resolution while maintaining temporal precision
  - Ideal for capturing subtle neurophysiological events
  - Produces high-quality embeddings for self-supervised tasks

- `wav_preprocessor.py`: Raw Waveform Processing
  - Direct processing of time-domain signals
  - Suitable for models with learned front-ends
  - Used for baseline comparisons and specific architectures

**Advanced Preprocessors:**
- `spec_pooled.py`: Pooled Spectrogram Preprocessor
  - Combines spectrogram preprocessing with temporal pooling
  - Takes the mean of a window around the middle timepoint
  - Useful for fixed-length feature extraction

- `spec_pretrained.py`: Pretrained Spectrogram Preprocessor
  - Combines spectrogram preprocessing with a pretrained transformer
  - Loads weights from a pretrained BrainBERT checkpoint
  - Extracts learned features from spectrograms

**Key Features:**
- Converts continuous signals into fixed-length embeddings
- Preserves both temporal and spectral information
- Facilitates self-supervised training through maskable frames
- Configurable parameters for resolution trade-offs

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
