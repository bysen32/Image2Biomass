# Imports

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys

os.environ["USE_TF"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import os
import pandas as pd
import random
import numpy as np
import torch
import gc

import math
from PIL import Image
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedGroupKFold
import matplotlib.pyplot as plt
import albumentations as A
import torch.optim as optim

import cv2
from albumentations.pytorch import ToTensorV2

import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import AutoModel


IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

# Config
if IS_KAGGLE:
    base_path = '/kaggle/input/csiro-biomass/'
else:
    base_path = './datasets/csiro-biomass/'

class CONFIG():
    train_path = os.path.join(base_path, 'train')
    test_path = os.path.join(base_path, 'test')
    test_csv_path = os.path.join(base_path, 'test.csv')
    train_csv_path = os.path.join(base_path, 'train.csv')
    data_dir = os.path.join(base_path,  "")
    nFolds = 5
    seed = 42
    pretrained = False
    pretrained_weights_path = os.path.join(base_path, "")
    best_model_dir = ""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 20
    batch_size = 8
    lr = 1e-3
    eta_min = 1e-5
    weight_decay = 1e-6
    
    # image
    img_size_h = 448
    img_size_w = 448
    in_chans = 3
    
    # target columns
    target_cols = [
        "Dry_Clover_g",
        "Dry_Dead_g",
        "Dry_Green_g",
        "Dry_Total_g",
        "GDM_g"
    ]
    n_targets = 3
    # we can predict multiple configs
    targets_configs = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"] # ["Dry_Green_g", "Dry_Total_g", "GDM_g"]
    weights = np.array([0.1, 0.1, 0.1, 0.5, 0.2])
    mapping = {"Dry_Clover_g": 0, "Dry_Dead_g": 1, "Dry_Green_g": 2, "Dry_Total_g": 3, "GDM_g": 4}
    train_backbone = False


# Functions
class BiomassTwoStreamDataset(Dataset):
    """
    Dataset that splits each image into two square crops (left & right)
    Returns:
        - x_left: (3, 224, 224)
        - x_right: (3, 224, 224)
        - y: (3,) targets if training/validation, else None
    """
    def __init__(self, df, transforms=None, is_test=False, input_res=224, targets=["Dry_Green_g", "Dry_Total_g", "GDM_g"]):
        self.df = df.reset_index(drop=True)
        self.image_paths = df["image_path"].values
        self.transforms = transforms
        self.is_test = is_test
        self.input_res = input_res

        if not is_test:
            self.targets_3 = df[targets].values.astype("float32")

    def __len__(self):
        return len(self.df)

    def to_tensor(self, img):
        """
        Convert image to tensor and normalize to ImageNet
        """
        img = torch.from_numpy(img).permute(2, 0, 1).float() #/ 255.0
        # mean = torch.tensor(IMAGENET_MEAN).view(3,1,1)
        # std  = torch.tensor(IMAGENET_STD).view(3,1,1)
        # img = (img - mean) / std
        return img

    def split_into_two_squares(self, img):
        """
        Split rectangular image into two square crops (left & right)
        """
        H, W, _ = img.shape
        if W < H:
            raise ValueError(f"Expected W >= H, got image shape {img.shape}")
        
        left  = img[:, :H, :]
        right = img[:, W-H:, :]
        return left, right

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        full_path = os.path.join(config.data_dir, image_path)
        img = cv2.imread(full_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {full_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Split into two squares
        x_left, x_right = self.split_into_two_squares(img)

        # Resize to input resolution
        # x_left  = cv2.resize(x_left, (self.input_res, self.input_res), interpolation=cv2.INTER_LINEAR)
        # x_right = cv2.resize(x_right, (self.input_res, self.input_res), interpolation=cv2.INTER_LINEAR)

        # Apply transforms if provided
        if self.transforms:
            x_left  = self.transforms(image=x_left)["image"]
            x_right = self.transforms(image=x_right)["image"]
        else:
            x_left  = self.to_tensor(x_left)
            x_right = self.to_tensor(x_right)

        if self.is_test:
            return x_left, x_right
        else:
            y = torch.tensor(self.targets_3[idx], dtype=torch.float32)
            return x_left, x_right, y

class DINOv2TwoStreamRegressor(nn.Module):
    def __init__(self, n_targets=3, model_path="/kaggle/input/dinov2/pytorch/large/1",
                 freeze_backbone=True, hidden_dim=512, dropout=0.1):
        super().__init__()

        model_path = 'facebook/dinov2-large'
        self.backbone = AutoModel.from_pretrained(model_path)
        embed_dim = self.backbone.config.hidden_size

        pooled_dim = embed_dim * 3  # CLS + avg + max

        self.regressor = nn.Sequential(
            nn.Linear(pooled_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_targets)
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _pool_tokens(self, outputs):
        last_hidden = outputs.last_hidden_state
        cls = last_hidden[:, 0]
        patches = last_hidden[:, 1:]

        avg_pool = patches.mean(dim=1)
        max_pool = patches.max(dim=1)[0]

        return torch.cat([cls, avg_pool, max_pool], dim=1)

    def forward(self, x_left, x_right):
        out_l = self.backbone(pixel_values=x_left)
        out_r = self.backbone(pixel_values=x_right)

        emb_l = self._pool_tokens(out_l)
        emb_r = self._pool_tokens(out_r)

        combined = torch.cat([emb_l, emb_r], dim=1)
        return self.regressor(combined)

# -------------------------------------------------------------
#  Training & Validation Loops
# -------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for img_left, img_right, labels in loader:
        img_left, img_right, labels = img_left.to(device), img_right.to(device), labels.to(device)

        optimizer.zero_grad()
        preds = model(img_left, img_right)
        loss = criterion(preds, labels)

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * img_left.size(0)

    return running_loss / len(loader.dataset)

def val_fn(model, dataloader, criterion, device, targets):
    """
    Validation / evaluation for 1 epoch.
    Assumes the dataset returns only 3 independent targets per sample.
    """
    model.eval()
    total_loss = 0
    all_targets_5_np = []
    all_preds_5_np = []

    with torch.no_grad():
        for img_left, img_right, targets_3 in dataloader:
            img_left, img_right, targets_3 = img_left.to(device), img_right.to(device), targets_3.to(device)  # shape (B, 3)

            # Model forward
            outputs_3 = model(img_left, img_right)  # (B, 3)

            # Compute loss directly on independent targets
            loss = criterion(outputs_3, targets_3)
            total_loss += loss.item() * img_left.size(0)  # scale by batch size

            # Expand predictions to 5 for metrics
            preds_5 = expand_predictions_torch(outputs_3, targets)
            all_preds_5_np.append(preds_5.cpu().numpy())

            # Expand targets to 5 as well
            targets_5 = expand_predictions_torch(targets_3, targets)
            all_targets_5_np.append(targets_5.cpu().numpy())

    # Average loss over dataset
    val_loss = total_loss / len(dataloader.dataset)

    # Concatenate all predictions / targets
    y_true_5 = np.concatenate(all_targets_5_np, axis=0)
    y_pred_5 = np.concatenate(all_preds_5_np, axis=0)

    # Compute weighted R² metric on full 5 targets
    weighted_r2, _ = weighted_r2_score(y_true_5, y_pred_5)

    return val_loss, weighted_r2


def weighted_r2_score(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Metric
    y_true, y_pred: shape (N, 5)
    """
    weights = config.weights
    r2_scores = []
    
    for i in range(y_true.shape[1]):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2_scores.append(r2)
        
    r2_scores = np.array(r2_scores)
    weighted_r2 = np.sum(r2_scores * weights) / np.sum(weights)
    return weighted_r2, r2_scores

# ======== 3. Preprocessing ========
def get_processed_data():
    """
    'long' -> 'wide'
    """
    train_df = pd.read_csv(config.train_csv_path)
    
    # unique id
    train_df["image_id"] = train_df["image_path"].apply(lambda x: x.split('/')[-1].split('.')[0])
    
    # pivot
    train_pivot = train_df.pivot(
        index="image_id", 
        columns="target_name", 
        values="target"
    ).reset_index()
    
    meta_df = train_df.drop_duplicates(subset="image_id").drop(
        columns=["sample_id", "target_name", "target"]
    )
    
    train_processed_df = meta_df.merge(train_pivot, on="image_id", how="left")
    
    # CV fold
    #kf = KFold(n_splits=config.nFolds, shuffle=True, random_state=config.seed)
    #train_processed_df["fold"] = -1
    #for fold, (train_idx, val_idx) in enumerate(kf.split(train_processed_df)):
    #    train_processed_df.loc[val_idx, "fold"] = fold

    skf = StratifiedKFold(n_splits=config.nFolds, shuffle=True, random_state=config.seed)
    train_processed_df["fold"] = -1
    # Add the stratification column (State)
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_processed_df, train_processed_df["State"])):
        train_processed_df.loc[val_idx, "fold"] = fold

    # StratifiedKFold
    #sgkf = StratifiedGroupKFold(n_splits=config.nFolds, shuffle=True, random_state=config.seed)
    #train_processed_df["fold"] = -1
    #groups = train_processed_df["Sampling_Date"]
    #y = train_processed_df["State"]
    #for fold, (train_idx, val_idx) in enumerate(sgkf.split(train_processed_df, y, groups=groups)):
    #    train_processed_df.loc[val_idx, "fold"] = fold
        
    return train_processed_df

def get_transforms(is_train):
    if is_train:
        return A.Compose([
            A.RandomResizedCrop((config.img_size_h, config.img_size_w), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
            A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(config.img_size_h, config.img_size_w),
            A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ToTensorV2(),
        ])
    
def get_two_stream_transforms(is_train):
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
            A.Resize(224, 224),  # ensure DINOv2 input size
            A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ToTensorV2(),
        ])

def expand_predictions_torch(preds_3, targets):
    """
    [torch] Expand the 3 NN predictions (N, 3) to 5 predictions (N, 5).
    """
    if targets == ["Dry_Green_g", "Dry_Total_g", "GDM_g"]:
        # clip
        P_Green = torch.clamp(preds_3[:, 0], min=0)
        P_Total = torch.clamp(preds_3[:, 1], min=0)
        P_GDM = torch.clamp(preds_3[:, 2], min=0)
        
        # Compute derived targets based on constraints.
        P_Clover = torch.clamp(P_GDM - P_Green, min=0)
        P_Dead = torch.clamp(P_Total - P_GDM, min=0)
    
    elif targets == ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"]:
        P_Clover = torch.clamp(preds_3[:, 0], min=0)
        P_Dead = torch.clamp(preds_3[:, 1], min=0)
        P_Green = torch.clamp(preds_3[:, 2], min=0)

        # Compute derived targets based on constraints.
        P_GDM = torch.clamp(P_Green + P_Clover, min=0)
        P_Total = torch.clamp(P_GDM + P_Dead, min=0)
    elif targets == ["Dry_Clover_g", "Dry_Dead_g", "GDM_g"]:
        P_Clover = torch.clamp(preds_3[:, 0], min=0)
        P_Dead =torch.clamp(preds_3[:, 1], min=0)
        P_GDM = torch.clamp(preds_3[:, 2], min=0)

        # Compute derived targets based on constraints.
        P_Green = torch.clamp(P_GDM - P_Clover, min=0)
        P_Total = torch.clamp(P_GDM + P_Dead, min=0)

    
    preds_5 = torch.stack(
        [
            P_Clover, # Index 0
            P_Dead,   # Index 1
            P_Green,  # Index 2
            P_Total,  # Index 3
            P_GDM     # Index 4
        ],
        dim=1
    )
    return preds_5


def expand_predictions_np(preds_3, targets):
    """
    [Numpy] Expand the three NN predictions (N, 3) to five predictions (N, 5).
    """

    if targets == ["Dry_Green_g", "Dry_Total_g", "GDM_g"]:
        # clip
        P_Green = np.clip(preds_3[:, 0], a_min=0, a_max=None)
        P_Total = np.clip(preds_3[:, 1], a_min=0, a_max=None)
        P_GDM = np.clip(preds_3[:, 2], a_min=0, a_max=None)
        
        # Compute derived targets based on constraints.
        P_Clover = np.clip(P_GDM - P_Green, a_min=0, a_max=None)
        P_Dead = np.clip(P_Total - P_GDM, a_min=0, a_max=None)
    
    elif targets == ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"]:
        P_Clover = np.clip(preds_3[:, 0], a_min=0, a_max=None)
        P_Dead = np.clip(preds_3[:, 1], a_min=0, a_max=None)
        P_Green = np.clip(preds_3[:, 2], a_min=0, a_max=None)

        # Compute derived targets based on constraints.
        P_GDM = np.clip(P_Green + P_Clover, a_min=0, a_max=None)
        P_Total = np.clip(P_GDM + P_Dead, a_min=0, a_max=None)

    elif targets == ["Dry_Clover_g", "Dry_Dead_g", "GDM_g"]:
        P_Clover = np.clip(preds_3[:, 0], a_min=0, a_max=None)
        P_Dead = np.clip(preds_3[:, 1], a_min=0, a_max=None)
        P_GDM = np.clip(preds_3[:, 2], a_min=0, a_max=None)

        # Compute derived targets based on constraints.
        P_Green = np.clip(P_GDM - P_Clover, a_min=0, a_max=None)
        P_Total = np.clip(P_GDM + P_Dead, a_min=0, a_max=None)
    
    preds_5 = np.stack(
        [P_Clover, P_Dead, P_Green, P_Total, P_GDM],
        axis=1
    )
    return preds_5
        
# ======== 8. Inference ========
def run_inference(targets):
    print("\n======== Starting Inference ========")
    
    # data
    test_df = pd.read_csv(config.test_csv_path)
    test_unique_df = test_df.drop_duplicates(subset="image_path").reset_index(drop=True)
    
    test_dataset = BiomassTwoStreamDataset(
        test_unique_df, 
        transforms=get_two_stream_transforms(is_train=False), 
        is_test=True,
        targets=targets
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size * 2, shuffle=False,
        num_workers=os.cpu_count(), pin_memory=True
    )
    
    # Prediction using 5-fold models
    all_fold_preds = []
    for fold in range(config.nFolds):
        print(f"Predicting with Fold {fold+1}...")
        model_path = os.path.join(config.best_model_dir, f"model_fold_{fold}.pth")

        model = DINOv2TwoStreamRegressor(n_targets=3, freeze_backbone=True).to(config.device)

        try:
            model.load_state_dict(torch.load(model_path))
        except FileNotFoundError:
            print(f"Warning: Model file {model_path} not found. Skipping fold {fold+1}.")
            continue
            
        model.eval()
        
        fold_preds = []
        with torch.no_grad():
            for img_left, img_right in test_loader:
                img_left, img_right = img_left.to(config.device), img_right.to(config.device)
                outputs = model(img_left, img_right)
                fold_preds.append(outputs.cpu().numpy())
        
        all_fold_preds.append(np.concatenate(fold_preds, axis=0))
    
    if not all_fold_preds:
        print("Error: No models were loaded. Cannot perform inference.")
        return

    # average
    # (nFolds, n_test_images, n_targets) -> (n_test_images, n_targets)
    avg_preds_3 = np.mean(all_fold_preds, axis=0) # (n_test_images, 3)
    avg_preds_5 = expand_predictions_np(avg_preds_3, targets) # (n_test_images, 5)
    
    # submission
    
    preds_df = pd.DataFrame(avg_preds_5, columns=config.target_cols)
    test_unique_df = pd.concat([test_unique_df, preds_df], axis=1)
    
    test_pred_long_df = test_unique_df.melt(
        id_vars=["image_path"], 
        value_vars=config.target_cols,
        var_name="target_name",
        value_name="target"
    )

    submission_df = test_df[["sample_id", "image_path", "target_name"]].merge(
        test_pred_long_df,
        on=["image_path", "target_name"],
        how="left"
    )
    
    final_submission = submission_df[["sample_id", "target"]].copy()

    # sanitize
    final_submission["target"] = final_submission["target"].fillna(0)
    final_submission["target"] = final_submission["target"].replace([np.inf, -np.inf], 0)
    final_submission["target"] = final_submission["target"].clip(lower=0)
    
    # submission
    final_submission.to_csv("submission.csv", index=False)
    print("Inference complete. submission.csv saved.")
    print(final_submission.head())

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None

    def step(self, score):
        if self.best_score is None or score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            return False

    def should_stop(self):
        return self.counter >= self.patience
    
    
# Training
config = CONFIG()
train_df = get_processed_data()

oof_predictions = np.zeros((len(train_df), len(config.target_cols)))

for fold in range(config.nFolds):
    print(f"\n======== FOLD {fold+1}/{config.nFolds} ========")
    # targets = ["Dry_Green_g", "Dry_Total_g", "GDM_g"]
    targets = config.targets_configs # ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"]

    # -----------------------------
    # Split fold 
    # -----------------------------
    train_fold_df = train_df[train_df.fold != fold].reset_index(drop=True)
    val_fold_df_with_idx = train_df[train_df.fold == fold]
    val_indices = val_fold_df_with_idx.index
    val_fold_df = val_fold_df_with_idx.reset_index(drop=True)

    # -----------------------------
    # Datasets & loaders
    # -----------------------------
    train_dataset = BiomassTwoStreamDataset(
        train_fold_df,
        transforms=get_transforms(is_train=True),
        targets=targets
    )
    val_dataset = BiomassTwoStreamDataset(
        val_fold_df,
        transforms=get_transforms(is_train=False),
        targets=targets
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = DINOv2TwoStreamRegressor(
        n_targets=3,
        freeze_backbone=True,
    ).to(config.device)

    criterion = nn.SmoothL1Loss(beta=0.5)
    best_val = -np.inf
    model_path = f"model_fold_{fold}.pth"

    # ======================================================
    # STAGE 1 — Train head only
    # ======================================================
    print("Training head only...")

    early_stop = EarlyStopping(patience=7)

    optimizer = torch.optim.AdamW(
        model.regressor.parameters(),
        lr=config.lr,
        weight_decay=1e-3,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.eta_min,
    )

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            config.device,
        )

        val_loss, weighted_r2 = val_fn(
            model,
            val_loader,
            criterion,
            config.device,
            targets
        )

        print(
            f"[HEAD] Epoch {epoch+1}/{config.epochs} "
            f"- Train: {train_loss:.4f} | Val: {val_loss:.4f} |  Val R2: {weighted_r2:.4f}"
        )

        scheduler.step()

        # ---- Save best model ----
        if weighted_r2 > best_val:
            best_val = weighted_r2
            torch.save(model.state_dict(), model_path)

        # ---- Early stopping logic ----
        early_stop.step(val_loss)
        if early_stop.should_stop():
            print("Early stopping (head)")
            break

    if config.train_backbone:
        # ======================================================
        # STAGE 2 — LayerNorm-only fine-tuning
        # ======================================================
        print("Fine-tuning LayerNorm only...")
    
        # Freeze entire backbone
        for param in model.backbone.parameters():
            param.requires_grad = False
    
        # Unfreeze only LayerNorms
        for name, param in model.backbone.named_parameters():
            if "norm" in name.lower():
                param.requires_grad = True
    
        early_stop = EarlyStopping(patience=4)
    
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": model.regressor.parameters(),
                    "lr": config.lr,
                    "weight_decay": 1e-3,
                },
                {
                    "params": filter(
                        lambda p: p.requires_grad,
                        model.backbone.parameters(),
                    ),
                    "lr": config.lr * 0.01,
                    "weight_decay": 1e-5,
                },
            ]
        )
    
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.3,
            patience=2,
        )
    
        for epoch in range(config.epochs // 2):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                config.device,
            )
    
            val_loss, weighted_r2 = val_fn(
                model,
                val_loader,
                criterion,
                config.device,
                targets
            )
    
            print(
                f"[FT] Epoch {epoch+1}/{config.epochs//2} "
                f"- Train: {train_loss:.4f} | Val: {val_loss:.4f} | Val R2: {weighted_r2:.4f} "
            )
    
            scheduler.step(val_loss)
    
            # ---- Save best model ----
            if weighted_r2 > best_val:
                best_val = weighted_r2
                torch.save(model.state_dict(), model_path)
            # ---- Early stopping logic ----
            early_stop.step(val_loss)
            if early_stop.should_stop():
                print("Early stopping (head)")
                break

    # -----------------------------
    # OOF predictions
    # -----------------------------
    print(f"Loading best model from {model_path}")

    model.load_state_dict(torch.load(model_path))
    model.eval()

    fold_preds_3 = []

    with torch.no_grad():
        for img_left, img_right, _ in val_loader:
            img_left, img_right = img_left.to(config.device), img_right.to(config.device)
            preds = model(img_left, img_right)
            fold_preds_3.append(preds.cpu().numpy())

    fold_preds_3 = np.concatenate(fold_preds_3, axis=0)

    # Expand 3 → 5
    fold_preds_5 = expand_predictions_np(fold_preds_3, targets)
    oof_predictions[val_indices] = fold_preds_5

    # -----------------------------
    # Cleanup
    # -----------------------------
    del model, train_dataset, val_dataset, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()

# OOF evaluation

# -----------------------------
# Final OOF score
# -----------------------------
oof_score, oof_scores_by_target = weighted_r2_score(train_df[config.target_cols].values, oof_predictions)
print(f"\n======== CV Finished ========")
print(f"Overall OOF Weighted R2 Score: {oof_score:.4f}")
for i, col in enumerate(config.target_cols):
    print(f"  - {col}: {oof_scores_by_target[i]:.4f}")

print("----------------------------------------------------------------------------------------------")
print("--------------------------------------------END-----------------------------------------------")
print("----------------------------------------------------------------------------------------------")

run_inference(targets)
