# Seizure Detection Pipeline with BrainBERT

This repository contains a modified version of the BrainBERT model, fine-tuned for seizure detection. The original BrainBERT model can be found [here](https://github.com/czlwang/BrainBERT).

The `seizure` folder includes scripts for processing Stereo-Electroencephalography (sEEG) data, generating embeddings, and training a logistic regression model to detect seizures.

## Pipeline Overview

1. **Process sEEG data**: Extract relevant channels, filter data, and create 5-second epochs.
2. **Generate labels**: Label epochs as seizure or non-seizure based on event files.
3. **Generate embeddings**: Use BrainBERT to generate embeddings from sEEG data.
4. **Train and evaluate model**: Use logistic regression to classify embeddings and assess performance.

## Scripts

### 1. Demo Notebook
**File:** `brainbert_embed_logreg_analysis.ipynb`

- This notebook is an updated and re-annotated version of the original demo notebook located at `BrainBERT/notebooks/demo.ipynb`. It performs the following tasks:
  - Loads the BrainBERT model.
  - Converts sEEG data into spectrograms using the Short-Time Fourier Transform (STFT).
  - Generates embeddings from the spectrograms and saves both the embeddings and their corresponding labels.

### 2. SEEG Data Processing
**File:** `preprocess_edf_pipeline.ipynb`

- Filters sEEG channels, removes artifacts, and resamples to 256 Hz.
- Applies notch filter to remove 60 Hz noise.
- Segments data into 5-second epochs.
- Saves processed data as `.npy` files.

### 3. Label Creation
**File:** `create_labels.ipynb`

- Reads event files for seizure onset/offset.
- Labels 5-second epochs (1 for seizure, 0 for non-seizure).

### 4. Logistic Regression Model Training
**File:** `train_brainbert_logreg.ipynb`

- Loads BrainBERT embeddings and labels.
- Splits data into training/testing sets.
- Trains and saves the logistic regression model.

### 5. Model Evaluation and Visualization
**File:** `brainbert_embed_logreg_analysis.ipynb`

- Evaluates model performance on the test set.
- Generates confusion matrix and ROC curve.

## Results

Running the evaluation script provides:

- Model accuracy on training and test sets.
- Confusion matrix for classification performance.
- ROC curve for assessing model's seizure detection ability.
