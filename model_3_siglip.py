# ==============================================================================
# 0. Library Imports and Initial Setup
# ==============================================================================
import warnings
warnings.filterwarnings('ignore') # Suppress warnings

from pathlib import Path
from tqdm.auto import tqdm
import json
from copy import deepcopy
import polars as pl
import numpy as np
import os

import torch
from PIL import Image
from transformers import AutoProcessor, AutoImageProcessor, AutoModel

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.dummy import DummyRegressor

import catboost

# --- Initial Setup ---
# Set device to 'cuda' if GPU is available, otherwise 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Path where the data is stored
data_path = Path('/kaggle/input/csiro-biomass')

# List of target variables (types of biomass) to predict
labels = [
    "Dry_Clover_g",
    "Dry_Dead_g",
    "Dry_Green_g",
    "Dry_Total_g",
    "GDM_g"
]

# ==============================================================================
# 1. Feature Preparation
# ==============================================================================

# --- 1.1. Load SigLIP Model ---
print("Loading SigLIP model...")
# Load the pretrained SigLIP model
# SigLIP is a Vision-Language model capable of extracting high-quality feature vectors from images.
model_name = "/kaggle/input/google-siglip-so400m-patch14-384/transformers/default/1/"
model = AutoModel.from_pretrained(
    model_name,
)
model = model.to(device)  # Move the model to the GPU
model.eval()              # Set the model to evaluation mode (disables gradient calculation)

# Load the corresponding image processor (for preprocessing)
processor = AutoImageProcessor.from_pretrained(model_name)
print("Model loading complete.")

# --- 1.2. Load and Preprocess Training Data ---
print("Processing training data...")
train = pl.read_csv(data_path / 'train.csv')
df = (
    train
    # Create columns for each label name based on 'target_name' and store the 'target' value (pivot operation)
    .with_columns([
        pl.when(pl.col('target_name') == label).then(pl.col('target')).alias(label)
        for label in labels
    ])
    # Group by image_path
    .group_by('image_path')
    # Calculate the mean for each label and create a group key for GroupKFold
    .agg([
        pl.col(label).mean()
        for label in labels
    ] + [
        pl.concat_str(["Sampling_Date", "State"], separator=" ").alias("group").first()
    ])
    .sort('image_path') # Sort by image_path
)

# --- 1.3. Load and Preprocess Test Data ---
print("Processing test data...")
test = pl.read_csv(data_path / 'test.csv')
df_test = (
    test
    .group_by('image_path')
    .len() # Get the number of targets for each image (5 in this case)
    .sort('image_path')
)

# --- 1.4. Extract Image Features with SigLIP ---
def compute_features(images: list, save_path: str):
    """Function to take a list of images, compute features with the SigLIP model, and save them to an ndjson file."""
    batch_size = 20
    with torch.no_grad(), open(save_path, 'w') as f:
        for i in tqdm(range(0, len(images), batch_size), desc=f"Extracting {save_path}"):
            batch_paths = images[i:i + batch_size]
            batch = [Image.open(data_path / p) for p in batch_paths]
            inputs = processor(images=batch, return_tensors="pt").to(model.device)
            features = model.get_image_features(**inputs)
            for line in features:
                data = {f'x_{j}': line[j].item() for j in range(len(line))}
                f.write(json.dumps(data) + '\n')

compute_features(df['image_path'], 'features.ndjson')
compute_features(df_test['image_path'], 'features_test.ndjson')
print("Feature extraction complete.")

# --- 1.5. Combine Features with Original Data ---
responses = pl.read_ndjson('features.ndjson')
responses_test = pl.read_ndjson('features_test.ndjson')
df_aug = pl.concat([df, responses], how='horizontal')
df_test_aug = pl.concat([df_test, responses_test], how='horizontal')

# ==============================================================================
# 2. Validation Setup
# ==============================================================================

# --- 2.1. Define Evaluation Metric and Weights ---
weights = {
    'Dry_Green_g': 0.1,
    'Dry_Dead_g': 0.1,
    'Dry_Clover_g': 0.1,
    'GDM_g': 0.2,
    'Dry_Total_g': 0.5,
}

def competition_metric(y_true, y_pred) -> float:
    """Function to calculate the competition's official evaluation metric (weighted R2 score)."""
    weights_array = np.array([weights[l] for l in labels])
    
    # Align with this calculation method
    y_weighted_mean = np.average(y_true, weights=weights_array, axis=1).mean()
    
    # For ss_res and ss_tot, also take the weighted average on axis=1, then the mean of the result
    ss_res = np.average((y_true - y_pred)**2, weights=weights_array, axis=1).mean()
    ss_tot = np.average((y_true - y_weighted_mean)**2, weights=weights_array, axis=1).mean()
    
    return 1 - ss_res / ss_tot

# --- 2.2. Define Cross-Validation Logic ---
def cross_validate(model, data, data_test, x_columns, random_state=42) -> tuple:
    """Function to perform GroupKFold cross-validation with a given model."""
    X = data.select(x_columns).to_numpy()
    X_test = data_test.select(x_columns).to_numpy()
    y_true = data.select(labels).to_numpy()
    
    y_pred_oof = np.zeros_like(y_true)
    y_pred_test = np.zeros([len(X_test), len(labels)])

    n_splits = 5
    kf = GroupKFold(n_splits=n_splits)
    groups = data.select('group')

    for i, (train_index, val_index) in enumerate(kf.split(X, groups=groups)):
        for l in range(len(labels)):
            m = deepcopy(model)
            m.fit(X[train_index], y_true[train_index, l])
            y_pred_oof[val_index, l] = m.predict(X[val_index]).clip(0)
            y_pred_test[:, l] += m.predict(X_test).clip(0) / n_splits
        
        score = competition_metric(y_true[val_index], y_pred_oof[val_index])
        print(f'Fold {i}: Score = {score:.6f}')

    full_cv_score = competition_metric(y_true, y_pred_oof)
    print(f'Full CV Score: {full_cv_score:.6f}')

    return y_pred_oof, y_pred_test

# ==============================================================================
# 3. Model Selection & Ensemble
# ==============================================================================

feature_columns = sorted(responses.columns)

# --- 3.1. Compare Performance of Multiple Models ---
print("\n--- [Final Model] GradientBoostingRegressor ---")
oof_pred_gb, pred_test_gb = cross_validate(
    GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    random_state=42
), df_aug, df_test_aug, feature_columns)
# --- [Comparison] LightGBM Regressor (GPU) ---
print("\n--- [Comparison] LightGBM Regressor (GPU) ---")
from lightgbm import LGBMRegressor
cross_validate(
    LGBMRegressor(
        device="gpu",              # ✅ GPU
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=64,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1,
        reg_lambda=2,
        random_state=42,
        verbosity=-1
    ),
    df_aug,
    df_test_aug,
    feature_columns
)

# --- [Comparison] XGBoost Regressor (GPU) ---
print("\n--- [Comparison] XGBoost Regressor (GPU) ---")
from xgboost import XGBRegressor

cross_validate(
    XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        reg_alpha=1,
        reg_lambda=2,
        random_state=42,
        tree_method="gpu_hist",     # ✅ GPU
        predictor="gpu_predictor",
        verbosity=0
    ),
    df_aug,
    df_test_aug,
    feature_columns
)

# --- [Final Model] CatBoostRegressor (GPU) ---
print("\n--- [Final Model] CatBoostRegressor (GPU) ---")
oof_pred_cb, pred_test_cb = cross_validate(
    catboost.CatBoostRegressor(
        task_type="GPU",           # ✅ GPU
        devices="0",
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        verbose=False,
        random_state=42,
        early_stopping_rounds=50
    ),
    df_aug,
    df_test_aug,
    feature_columns
)
# ★★★★★ Ensemble and Save OOF Predictions ★★★★★
print("\nEnsembling and saving OOF predictions...")
oof_pred_ensemble = (oof_pred_gb + oof_pred_cb) / 2
oof_df = df_aug.select(['image_path']).to_pandas()
oof_df[labels] = oof_pred_ensemble
oof_df.to_csv('oof_model3.csv', index=False)

# --- 3.2. Ensemble Model Predictions ---
print("\nEnsembling the predictions of the two models...")
pred_test = (pred_test_gb + pred_test_cb) / 2

# ==============================================================================
# 4. Create Submission File
# ==============================================================================

# --- Convert prediction results to a DataFrame ---
pred_with_id = pl.concat([
    df_test.select("image_path"),
    pl.DataFrame(pred_test, schema=labels),
], how='horizontal')

# --- Format for submission ---
pred_save = (
    test
    .join(pred_with_id, on='image_path')
    .with_columns(
        pl.coalesce(*[  
            pl.when(pl.col('target_name') == col).then(pl.col(col))
            for col in labels
        ]).alias('target')
    )
    .select('sample_id', 'target')
)

# --- Save as CSV file ---
pred_save.write_csv('submission_SigLIP.csv')
print("\nCreated submission_SigLIP.csv.")
print("Partial submission file:")
print(pred_save.head())
