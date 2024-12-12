# BrainBERT Repository Structure

This document provides a concise overview of the BrainBERT repository layout, clarifying which components are author-specific and which are essential for extending the BrainBERT architecture. The structure and conventions are inspired by documentation standards commonly found at large tech companies.

## Overview

**Goal:** BrainBERT learns self-supervised representations from intracranial electrode data (ECoG). This repository is organized for modularity and clarity, using Hydra for configuration and well-defined directories for each component of the pipeline.

**Key Concepts:**
- **Configuration**: Managed via Hydra in `/conf`
- **Data Processing & Loading**: Defined in `/data` and `/datasets`
- **Models**: Found in `/models`
- **Preprocessors**: Signal transformations in `/preprocessors`
- **Pretraining & Tasks**: Scripts and utilities in `/pretrain` and `/tasks`
- **Utilities & Tests**: Shared code in `/util`, tests in `/testing`
- **Author-Specific vs. General Files**: Certain files contain author's domain assumptions (e.g., electrode naming conventions) and should be adapted before reuse

## Core Directories

### `/conf`
**Purpose:** Central configuration using Hydra. Organizes model, data, preprocessor, and experiment settings.

**Key Subdirectories:**
- `conf/model/`: Model architecture definitions
- `conf/data/`: Dataset and data-loading parameters
- `conf/preprocessor/`: Signal preprocessing configs
- `conf/exp/`: Training and resource allocation settings

**Usage Example:**
```bash
python run_train.py +exp=spec2vec +model=masked_tf_model_large
python run_train.py +exp=spec2vec ++model.hidden_dim=512
```

**Key Takeaway:** If expanding BrainBERT, adjust model, data, and experiment configs here.

### `/data`
**Purpose:** Core data processing pipeline for ECoG data: transforming raw recordings (e.g., EDF) into manageable formats (e.g., HDF5), organizing trials, and preparing datasets for training.

**Key Files:**
- `edf2h5.py`, `h5_data.py`, `h5_data_reader.py`: Convert and load data from EDF to HDF5
- `trial_data.py`, `trial_data_reader.py`: Manage trial-based data loading
- `create_data_dirs.py`: Sets up directory structures
- `write_data_to_disk.py`: Persists processed data

**Author-Specific Files:**
- `corrupted_elec.json`
- `test_split_trials.json`
- `speech_nonspeech_subject_data.py`

These contain subject- or electrode-specific assumptions and splits. Modify or remove these when adapting BrainBERT to new datasets.

**If Expanding BrainBERT:**
You only need to ensure your new data fits the existing data loading formats (`subject_data.py`, `trial_data_reader.py`) and adjust the configs. The rest can remain unchanged.

### `/datasets`
**Purpose:** Defines dataset classes that integrate processed data into model-ready formats, including masked pretraining datasets and fine-tuning sets.

**Key Files:**
- `base_tf_dataset.py`: Base dataset class
- `masked_tf_dataset.py`: Masked dataset for self-supervised pretraining
- `finetuning_datasets.py`: Datasets for downstream tasks

**Author-Specific Notes:**
Certain electrode or subject assumptions in `single_subject_all_electrode.py` and `finetuning_datasets.py` may need updates.

**If Expanding BrainBERT:**
Focus on `base_tf_dataset.py` and `masked_tf_dataset.py` for new datasets. Adjust `finetuning_datasets.py` as needed for new tasks.

### `/models`
**Purpose:** Houses the BrainBERT models and related architectures.

**Key Files:**
- `masked_tf_model.py`: Core BrainBERT masked transformer model
- `base_model.py`: Base class for implementing new models
- `transformer_encoder_input.py`: Transformer encoder for ECoG signals

**Author-Specific Notes:**
`seeg_wav2vec.py` and certain baseline models assume specific data formats. These can be adapted if your domain differs.

**If Expanding BrainBERT:**
Start with `masked_tf_model.py` for the main architecture and tweak model configs in `/conf/model/`.

### `/preprocessors`
**Purpose:** Performs signal transformations (e.g., STFT, wavelet transforms).

**Key Files:**
- `stft.py`, `morelet_preprocessor.py`, `superlet_preprocessor.py`: Time-frequency transformations
- `wav_preprocessor.py`: Raw waveform preprocessing

**Author-Specific Notes:**
Frequency bands or normalization strategies might be tailored to a specific dataset.

**If Expanding BrainBERT:**
You may only need to add or adjust a preprocessor if using new signal modalities.

### `/pretrain`
**Purpose:** Includes scripts and methods for pretraining BrainBERT using self-supervised strategies like masking.

**Key Files:**
- `spec2vec/spec2vec.py`: Core pretraining script leveraging spectrogram-to-vector approaches

**Author-Specific Notes:**
Masking strategies and augmentation assumptions may be domain-specific.

**If Expanding BrainBERT:**
Customize pretraining tasks or objectives here. No need to reinvent the entire pipeline—just plug into existing configs and datasets.

### Other Directories
- `/schedulers`: Learning rate schedulers
- `/tasks`: Task definitions for training/evaluation
- `/testing`: Unit/integration tests
- `/util`: Shared utilities

### Root Files
- `requirements.txt`: Project dependencies
- `run_train.py`: Main training entry point
- `run_tests.py`: Executes test suite
- `runner.py`: Orchestrates the training loop

## What to Modify if Expanding BrainBERT

### Data Adaptation
1. Update or replace author-specific data files (`corrupted_elec.json`, `test_split_trials.json`, `speech_nonspeech_subject_data.py`)
2. Ensure your data loaders (`h5_data_reader.py`, `subject_data.py`) conform to your dataset format

### Model Architecture
1. Focus on `masked_tf_model.py` for core BrainBERT architecture changes
2. Adjust configurations in `/conf/model/`

### Preprocessing & Datasets
1. If adding new signal types or transforms, modify `/preprocessors`
2. For new training tasks or downstream datasets, create or adjust classes in `/datasets`

## Getting Started

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Set Up Data
Configure and run `create_data_dirs.py` and other data scripts as per your needs.

### Configure Hydra
Adjust `/conf` files for your training setup, data paths, and model parameters.

### Run Training
```bash
python run_train.py +exp=spec2vec +model=masked_tf_model_large
```

### Testing & Validation
```bash
python run_tests.py
```

## Best Practices
1. Keep all configuration in `/conf`
2. Test changes using `/testing`
3. Document changes and configurations clearly
4. For performance optimizations, utilize caching and consider in-memory datasets
5. Validate data quality and monitor model performance regularly

This streamlined structure should help you quickly navigate the repository, identify author-specific components, and understand which parts to modify when extending BrainBERT.