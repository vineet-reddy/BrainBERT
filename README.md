# BrainBERT

BrainBERT is an modeling approach for learning self-supervised representations of intracranial electrode data. See [paper](https://arxiv.org/abs/2302.14367) for details.

We provide the training pipeline below.

The trained weights have been released (see below) and pre-training data is available upon request.

## Installation
Requirements:
- pytorch >= 1.12.1
- [pytorch gradual warmup scheduler](https://github.com/ildoonet/pytorch-gradual-warmup-lr)

```
pip install -r requirements.txt
```

## Criterions

The `criterions` package provides loss functions for different training stages:

- `baseline_criterion`: Binary classification (BCE loss)
- `feature_extract_criterion`: Feature extraction for audio classification
- `finetune_criterion`: Model fine-tuning
- `pretrain_masked_criterion`: Masked pretraining (used in main pipeline)
- `seeg_wav2vec_criterion`: Wav2Vec-style SEEG processing

Usage in training configs:
```yaml
# config.yaml
criterion:
  type: pretrain_masked_criterion  # or other criterion name
  # criterion-specific settings here
```

### Input
It is expected that the input is intracranial electrode data that has been Laplacian re-referenced.

## Using BrainBERT embeddings
- pretrained weights are available [here](https://drive.google.com/file/d/14ZBOafR7RJ4A6TsurOXjFVMXiVH6Kd_Q/view?usp=sharing)
- see `notebooks/demo.ipynb` for an example input and example embedding

## Data Processing Utilities

The `data` folder contains a comprehensive pipeline for processing intracranial electrode (ECoG) data:

### Core Data Classes
- `subject_data.py`: Base class for handling subject-specific data
- `trial_data.py` and `trial_data_reader.py`: Core classes for reading and processing trial data
  - Handles data loading, filtering, and preprocessing
  - Supports different referencing methods including Laplacian
- `electrode_subject_data.py`: Manages electrode-specific data and metadata
- `timestamped_subject_data.py`: Handles time-aligned neural recordings
- `speech_nonspeech_subject_data.py`: Specialized classes for:
  - `NonLinguisticSubjectData`: Processing non-linguistic neural data
  - `SentenceOnsetSubjectData`: Handling sentence onset-related data

### Data Format and Storage
- `edf2h5.py`: Converts EDF (European Data Format) neurophysiological data to HDF5
- `h5_data.py` and `h5_data_reader.py`: Tools for HDF5 data management
  - Supports frequency filtering
  - Handles data chunking and caching
- `write_data_to_disk.py`: General-purpose data writing utility
- `write_preprocessed_inputs.py`: Prepares preprocessed data for model input
- `write_pretrain_data_wavs.py`: Converts neural data to wav format for pretraining

### Data Processing Tools
- `electrode_selection.py`: 
  - Identifies and validates Laplacian electrode configurations
  - Filters out corrupted or problematic electrodes
- `throw_out_zeros.py`: Removes zero-value or invalid data segments
- `make_aligned_data_caches.py`: Creates time-aligned data caches for efficient processing
- `modify_manifest.py`: Updates data manifests for different preprocessing configurations

### Data Organization
- `create_data_dirs.py`: Sets up the required directory structure
- Configuration files:
  - `corrupted_elec.json`: Lists problematic electrodes to exclude
  - `test_split_trials.json`: Defines train/test split configurations

### Usage

1. Initial Setup:
```bash
# Create directory structure
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
