# ====================================================================================
# Model 1: DINOv2-Giant + Lasso Regression Inference Script
# ====================================================================================

import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoImageProcessor, AutoModel
from sklearn.linear_model import Lasso
from PIL import Image
import warnings

# --- Global Settings ---
warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- DINOv2 + Lasso Common Function ---
def run_dinov2_lasso_inference(model_id, desc):
    ROOT = "/kaggle/input/csiro-biomass/"
    
    # --- 1. Load Models ---
    processor = AutoImageProcessor.from_pretrained(f'/kaggle/input/dinov2/pytorch/{model_id}/1')
    model = AutoModel.from_pretrained(f'/kaggle/input/dinov2/pytorch/{model_id}/1/')
    model = model.to(DEVICE)
    print("  Models loaded.")

    # --- 2. Extract Features from Training Data & Train Lasso Model ---
    train_df = pd.read_csv(os.path.join(ROOT, "train.csv"))
    embeds, targets = [], [[] for _ in range(5)]
    target_mapping = {"Dry_Clover_g": 0, "Dry_Dead_g": 1, "Dry_Green_g": 2, "Dry_Total_g": 3, "GDM_g": 4}  
    train_df['target_name'] = train_df['sample_id'].apply(lambda x: x.split('__')[1])

    unique_train_images = train_df.drop_duplicates(subset=['image_path']).reset_index()
    for i, entry in tqdm(unique_train_images.iterrows(), total=len(unique_train_images), desc=f"  {desc} Extracting train features"):
        file_path = os.path.join(ROOT, entry['image_path'])
        with Image.open(file_path) as img:
            inputs = processor(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            embeds.append(outputs.pooler_output.cpu())

    # Correctly classify the target values
    for _, row in train_df.iterrows():
        target_idx = target_mapping[row['target_name']] 
        targets[target_idx].append(torch.tensor([[row['target']]]))

    embeds_np = np.array(torch.cat(embeds))
    regressors = [[None for _ in range(5)] for _ in range(5)]
    # Initialize an array to store OOF predictions
    oof_preds_np = np.zeros((len(embeds_np), 5)) # 5 is the number of targets
    
    print("  Training Lasso regression models...")
    for i in range(5): # For each target (Dry_Clover_g, Dry_Dead_g, ...)
        targets_np = np.array(torch.cat(targets[i]))
        
        # Split using KFold (more robust than random split)
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
        for fold, (train_idxs, val_idxs) in enumerate(kf.split(embeds_np)):
            X_train, y_train = embeds_np[train_idxs], targets_np[train_idxs]
            X_val, y_val = embeds_np[val_idxs], targets_np[val_idxs]
            
            reg = Lasso()
            reg.fit(X_train, y_train)
            
            # Calculate and save OOF predictions
            oof_preds_np[val_idxs, i] = reg.predict(X_val).flatten()
    
            regressors[i][fold] = reg # Also save the model for test prediction
    
    # Link and save image_path from unique_train_images with oof_preds_np
    target_columns = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    oof_df = pd.DataFrame(oof_preds_np, columns=target_columns)
    oof_df['image_path'] = unique_train_images['image_path']
    oof_df.to_csv('oof_model1.csv', index=False)

    # --- 3. Inference on Test Data ---
    print("  Running predictions on test data...")
    test_df = pd.read_csv(os.path.join(ROOT, "test.csv"))
    test_embeds = {}
    for img_path in tqdm(test_df['image_path'].unique(), desc=f"  {desc} Extracting test features"):
        full_path = os.path.join(ROOT, img_path)
        with Image.open(full_path) as img:
            inputs = processor(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            test_embeds[img_name] = outputs.pooler_output.cpu()

    predictions, sample_ids = [], []

    for _, entry in test_df.iterrows():
        sample_id = entry['sample_id']
        img_name, target_name = sample_id.split('__')
        X_test = np.array(test_embeds[img_name])
        target_idx = target_mapping[target_name]
        fold_preds = [reg.predict(X_test) for reg in regressors[target_idx]]
        prediction = np.mean(fold_preds)
        predictions.append(max(0.0, prediction))
        sample_ids.append(sample_id)

    submission = pd.DataFrame({'sample_id': sample_ids, 'target': predictions})
    return submission.sort_values('sample_id').reset_index(drop=True)

# --- Main Execution Block ---
if __name__ == "__main__":
    print(f"--- [Start] Model 1: DINOv2-Giant + Lasso ---")
    print(f"Inference device: {DEVICE}")

    submission1 = run_dinov2_lasso_inference("giant", "Model 1 (DINOv2-Giant)")
    
    output_path = "submission_dino_giant.csv"
    submission1.to_csv(output_path, index=False)
    print(f"--- [Done] Model 1: Predictions saved to {output_path} ---")

    # Free up memory
    del submission1
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()