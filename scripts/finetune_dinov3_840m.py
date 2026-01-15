# show timm version
############################################################
# 0.Import
############################################################
from contextlib import nullcontext
import pandas as pd
import cv2
import math
import torch.optim as optim
import torch.nn as nn
import gc
import torch
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedGroupKFold
from timm.utils import ModelEmaV2
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import timm
print(timm.__version__)

is_kaggle = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

############################################################
# 1.Config
############################################################


class CFG:
    CREATE_SUBMISSION = False
    USE_TQDM = False
    PRETRAINED_DIR = None
    PRETRAINED = True
    BASE_PATH = '/kaggle/input/csiro-biomass'
    SEED = 82947501
    FOLDS_TO_TRAIN = [0, 1, 2, 3, 4]
    TRAIN_CSV = os.path.join(BASE_PATH, 'train.csv')
    TRAIN_IMAGE_DIR = os.path.join(BASE_PATH, 'train')
    TEST_IMAGE_DIR = '/kaggle/input/csiro-biomass/test' if is_kaggle else './datasets/csiro-biomass/test'
    TEST_CSV = '/kaggle/input/csiro-biomass/test.csv'
    SUBMISSION_DIR = '/kaggle/working/'
    MODEL_DIR = '/kaggle/working/5-folds-dinov3-840m'
    MODEL_DIR_012 = '/kaggle/input/5-folds-dinov3-840m/other/fold-0-1-2/1'
    MODEL_DIR_34 = '/kaggle/input/5-folds-dinov3-840m/other/fold-0-1-2/2'
    N_FOLDS = 5

    # MODEL_NAME      = 'vit_large_patch16_dinov3.lvd1689m'
    # BACKBONE_PATH   = '/kaggle/input/vit-large-patch16-dinov3-lvd1689m-backbone-pth/vit_large_patch16_dinov3.lvd1689m_backbone.pth'
    MODEL_NAME = 'vit_huge_plus_patch16_dinov3.lvd1689m'
    BACKBONE_PATH = '/kaggle/input/vit-huge-plus-patch16-dinov3-lvd1689m/vit_huge_plus_patch16_dinov3.lvd1689m_backbone.pth'

    IMG_SIZE = 512

    VAL_TTA_TIMES = 1
    TTA_STEPS = 1

    BATCH_SIZE = 1
    GRAD_ACC = 4
    NUM_WORKERS = 4
    EPOCHS = 1
    FREEZE_EPOCHS = 0
    WARMUP_EPOCHS = 3
    LR_REST = 1e-3
    LR_BACKBONE = 5e-4
    WD = 1e-2
    EMA_DECAY = 0.9
    PATIENCE = 5
    TARGET_COLS = ['Dry_Total_g', 'GDM_g', 'Dry_Green_g']
    DERIVED_COLS = ['Dry_Clover_g', 'Dry_Dead_g']
    ALL_TARGET_COLS = ['Dry_Green_g', 'Dry_Dead_g',
                       'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
    R2_WEIGHTS = np.array([0.1, 0.1, 0.1, 0.2, 0.5])
    LOSS_WEIGHTS = np.array([0.1, 0.1, 0.1, 0.0, 0.0])
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print(f'Device : {CFG.DEVICE}')
print(f'Backbone: {CFG.MODEL_NAME} | Input: {CFG.IMG_SIZE}')
print(f'Freeze Epochs: {CFG.FREEZE_EPOCHS} | Warmup: {CFG.WARMUP_EPOCHS}')
print(f'EMA Decay: {CFG.EMA_DECAY} | Grad Acc: {CFG.GRAD_ACC}')

##################################
# 2.Metrics
############################################################

def weighted_r2_score(y_true: np.ndarray, y_pred: np.ndarray):
    weights = CFG.R2_WEIGHTS
    r2_scores = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2_scores.append(r2)
    r2_scores = np.array(r2_scores)
    weighted = np.sum(r2_scores * weights) / np.sum(weights)
    return weighted, r2_scores


def weighted_r2_score_global(y_true: np.ndarray, y_pred: np.ndarray):
    weights = CFG.R2_WEIGHTS
    flat_true = y_true.reshape(-1)
    flat_pred = y_pred.reshape(-1)
    w = np.tile(weights, y_true.shape[0])
    mean_w = np.sum(w * flat_true) / np.sum(w)
    ss_res = np.sum(w * (flat_true - flat_pred) ** 2)
    ss_tot = np.sum(w * (flat_true - mean_w) ** 2)
    global_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    avg_r2, per_r2 = weighted_r2_score(y_true, y_pred)
    return global_r2, avg_r2, per_r2


def analyze_errors(val_df, y_true, y_pred, targets, top_n=5):
    print(f'\n--- Top {top_n} High Loss Samples per Target ---')
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    for i, target in enumerate(targets):
        errors = np.abs(y_true[:, i] - y_pred[:, i])
        top_indices = np.argsort(errors)[::-1][:top_n]

        print(f'\nTarget: {target}')
        print(
            f'{"Index":<6} | {"Image Path":<40} | {"True":<10} | {"Pred":<10} | {"AbsErr":<10}')
        print('-' * 90)

        for idx in top_indices:
            path = val_df.iloc[idx]['image_path']
            path_disp = os.path.basename(path)
            t_val = y_true[idx, i]
            p_val = y_pred[idx, i]
            err = errors[idx]
            print(
                f'{idx:<6} | {path_disp:<40} | {t_val:<10.4f} | {p_val:<10.4f} | {err:<10.4f}')


def compare_train_val(tr_df, val_df, targets, show_plots=True):
    """Quick comparison of target distributions and metadata between train and val splits."""
    print("\n--- Train / Val Comparison ---")

    for t in targets:
        tr = tr_df.get(t, pd.Series(dtype=float)).dropna()
        val = val_df.get(t, pd.Series(dtype=float)).dropna()
        print(f"\nTarget: {t}")
        print(
            f"  Train: n={len(tr)} mean={tr.mean():.3f} std={tr.std():.3f} min={tr.min():.3f} max={tr.max():.3f}")
        print(
            f"  Val  : n={len(val)} mean={val.mean():.3f} std={val.std():.3f} min={val.min():.3f} max={val.max():.3f}")
        if show_plots:
            try:
                plt.figure(figsize=(6, 3))
                sns.kdeplot(tr, label='train', fill=True)
                sns.kdeplot(val, label='val', fill=True)
                plt.legend()
                plt.title(f'Distribution: {t}')
                plt.show()
            except Exception as e:
                print('  Could not plot distributions for', t, '-', e)

    # Compare Sampling_Date and State if present
    if 'Sampling_Date' in tr_df.columns:
        try:
            tr_dates = pd.to_datetime(tr_df['Sampling_Date'], errors='coerce')
            val_dates = pd.to_datetime(
                val_df['Sampling_Date'], errors='coerce')
            print("\nSampling_Date range:")
            print(
                f"  Train: {tr_dates.min()} -> {tr_dates.max()} (missing {tr_dates.isna().sum()})")
            print(
                f"  Val  : {val_dates.min()} -> {val_dates.max()} (missing {val_dates.isna().sum()})")
        except Exception as e:
            print('  Could not parse Sampling_Date:', e)
    if 'State' in tr_df.columns:
        print("\nState distribution (train vs val):")
        tr_state = tr_df['State'].value_counts(normalize=True)
        val_state = val_df['State'].value_counts(normalize=True)
        state_df = pd.concat([tr_state, val_state], axis=1,
                             keys=['train', 'val']).fillna(0)

        print(state_df)



# Dataset & Augmentation

def get_train_transforms():
    return A.Compose([
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=(-10, 10), p=0.3, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101),
        A.ColorJitter(brightness=0.1, contrast=0.1,
                      saturation=0.1, hue=0.05, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], p=1.0)


def get_val_transforms():
    return A.Compose([
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], p=1.0)


def get_tta_transforms(mode=0):
    # mode 0: original
    # mode 1: hflip
    # mode 2: vflip
    # mode 3: rotate90
    transforms_list = [
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
    ]

    if mode == 1:
        transforms_list.append(A.HorizontalFlip(p=1.0))
    elif mode == 2:
        transforms_list.append(A.VerticalFlip(p=1.0))
    elif mode == 3:
        # RandomRotate90 with p=1.0 rotates 90, 180, 270 randomly?
        transforms_list.append(A.RandomRotate90(p=1.0))
        # Albumentations RandomRotate90 rotates by 90, 180, 270.
        # Reference uses transforms.RandomRotation([90, 90]) which is exactly 90 degrees.
        # To match exactly 90 degrees in Albumentations, we might need Rotate(limit=(90,90), p=1.0)
        # But RandomRotate90 is standard TTA. Let's use Rotate(limit=(90,90)) to be precise if that's what reference does.
        # Reference: transforms.RandomRotation([90, 90]) -> rotates by exactly 90 degrees.
        transforms_list.append(A.Rotate(limit=(
            90, 90), p=1.0, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101))

    transforms_list.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    return A.Compose(transforms_list, p=1.0)


def clean_image(img):
    # 1. Safe Crop (Remove artifacts at the bottom)
    h, w = img.shape[:2]
    # Cut bottom 10% where artifacts often appear
    img = img[0:int(h*0.90), :]

    # 2. Inpaint Date Stamp (Remove orange text)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Define orange color range (adjust as needed)
    lower = np.array([5, 150, 150])
    upper = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Dilate mask to cover text edges and reduce noise
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)

    # Inpaint if mask is not empty
    if np.sum(mask) > 0:
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    return img


class BiomassDataset(Dataset):
    def __init__(self, df, transform, img_dir):
        self.df = df
        self.transform = transform
        self.img_dir = img_dir
        self.paths = df['image_path'].values
        self.labels = df[CFG.ALL_TARGET_COLS].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, os.path.basename(self.paths[idx]))
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = clean_image(img)
        h, w, _ = img.shape
        mid = w // 2
        left = img[:, :mid]
        right = img[:, mid:]
        left = self.transform(image=left)['image']
        right = self.transform(image=right)['image']
        label = torch.from_numpy(self.labels[idx])
        return left, right, label


############################################################
# 3.Model
############################################################

# class BiomassModel(nn.Module):
#     def __init__(self, model_name, pretrained=True):
#         super().__init__()
#         self.model_name = model_name
#         self.backbone = timm.create_model(self.model_name, pretrained=False, num_classes=0, global_pool='avg')
#         nf = self.backbone.num_features
#         comb = nf * 2
#         self.head_green_raw  = nn.Sequential(nn.Linear(comb, comb//2), nn.GELU(), nn.Dropout(0.3), nn.Linear(comb//2, 1))
#         self.head_clover_raw = nn.Sequential(nn.Linear(comb, comb//2), nn.GELU(), nn.Dropout(0.3), nn.Linear(comb//2, 1))
#         self.head_dead_raw   = nn.Sequential(nn.Linear(comb, comb//2), nn.GELU(), nn.Dropout(0.3), nn.Linear(comb//2, 1))
#         if pretrained:
#             self.load_pretrained()

#     def load_pretrained(self):
#         try:
#             sd = timm.create_model(self.model_name, pretrained=True, num_classes=0, global_pool='avg').state_dict()
#             self.backbone.load_state_dict(sd, strict=False)
#             print('Pretrained weights loaded.')
#         except Exception as e:
#             print(f'Warning: pretrained load failed: {e}')

#     def forward(self, left, right):
#         fl = self.backbone(left)
#         fr = self.backbone(right)
#         x  = torch.cat([fl, fr], dim=1)
#         green  = self.head_green_raw(x)
#         # clover = torch.nn.functional.softplus(self.head_clover_raw(x))
#         # dead   = torch.nn.functional.softplus(self.head_dead_raw(x))
#         clover = self.head_clover_raw(x)
#         dead   = self.head_dead_raw(x)
#         gdm    = green + clover
#         total  = gdm + dead
#         return total, gdm, green, clover, dead

class LocalMambaBlock(nn.Module):
    """
    Lightweight Mamba-style block (Gated CNN) from the reference notebook.
    Efficiently mixes tokens with linear complexity.
    """

    def __init__(self, dim, kernel_size=5, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # Depthwise conv mixes spatial information locally
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Tokens, Dim)
        shortcut = x
        x = self.norm(x)
        # Gating mechanism
        g = torch.sigmoid(self.gate(x))
        x = x * g
        # Spatial mixing via 1D Conv (requires transpose)
        x = x.transpose(1, 2)  # -> (B, D, N)
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # -> (B, N, D)
        # Projection
        x = self.proj(x)
        x = self.drop(x)
        return shortcut + x


class BiomassModel(nn.Module):
    def __init__(self, model_name, pretrained=True, backbone_path=None):
        super().__init__()
        self.model_name = model_name
        self.backbone_path = backbone_path

        # 1. Load Backbone with global_pool='' to keep patch tokens
        #    (B, 197, 1024) instead of (B, 1024)
        self.backbone = timm.create_model(
            self.model_name, pretrained=False, num_classes=0, global_pool='')

        # 2. Enable Gradient Checkpointing (Crucial for ViT-Large memory!)
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(True)
            print("✓ Gradient Checkpointing enabled (saves ~50% VRAM)")

        nf = self.backbone.num_features

        # 3. Mamba Fusion Neck
        #    Mixes the concatenated tokens [Left, Right]
        self.fusion = nn.Sequential(
            LocalMambaBlock(nf, kernel_size=5, dropout=0.1),
            LocalMambaBlock(nf, kernel_size=5, dropout=0.1)
        )

        # 4. Pooling & Heads
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Heads (using the same logic as before, but on fused features)
        self.head_green_raw = nn.Sequential(
            nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(nf//2, 1), nn.Softplus()
        )
        self.head_clover_raw = nn.Sequential(
            nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(nf//2, 1), nn.Softplus()
        )
        self.head_dead_raw = nn.Sequential(
            nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(nf//2, 1), nn.Softplus()
        )

        if pretrained:
            self.load_pretrained()

    def load_pretrained(self):
        try:
            # Load weights normally
            if self.backbone_path and os.path.exists(self.backbone_path):
                print(
                    f"Loading backbone weights from local file: {self.backbone_path}")
                sd = torch.load(self.backbone_path, map_location='cpu')
                # Handle common checkpoint wrappers (e.g. if saved with 'model' key)
                if 'model' in sd:
                    sd = sd['model']
                elif 'state_dict' in sd:
                    sd = sd['state_dict']
            else:
                # Original behavior: Download from internet
                print("Downloading backbone weights...")
                sd = timm.create_model(
                    self.model_name, pretrained=True, num_classes=0, global_pool='').state_dict()

            # Interpolate pos_embed if needed (for 256x256 vs 224x224)
            if 'pos_embed' in sd and hasattr(self.backbone, 'pos_embed'):
                pe_ck = sd['pos_embed']
                pe_m = self.backbone.pos_embed
                if pe_ck.shape != pe_m.shape:
                    print(
                        f"Interpolating pos_embed: {pe_ck.shape} -> {pe_m.shape}")
                    # (Simple interpolation logic here or rely on timm's load if strict=False handles it well enough)
                    # For robust interpolation, use the snippet provided in previous turn

            self.backbone.load_state_dict(sd, strict=False)
            print('Pretrained weights loaded.')
        except Exception as e:
            print(f'Warning: pretrained load failed: {e}')

    def forward(self, left, right):
        # 1. Extract Tokens (B, N, D)
        #    Note: ViT usually returns [CLS, Patch1, Patch2...]
        #    We remove CLS token for spatial mixing, or keep it. Let's keep it.
        x_l = self.backbone(left)
        x_r = self.backbone(right)

        # 2. Concatenate Left and Right tokens along sequence dimension
        #    (B, N, D) + (B, N, D) -> (B, 2N, D)
        x_cat = torch.cat([x_l, x_r], dim=1)

        # 3. Apply Mamba Fusion
        #    This allows tokens from Left image to interact with tokens from Right image
        x_fused = self.fusion(x_cat)

        # 4. Global Pooling
        #    (B, 2N, D) -> (B, D, 2N) -> (B, D, 1) -> (B, D)
        x_pool = self.pool(x_fused.transpose(1, 2)).flatten(1)

        # 5. Prediction Heads
        green = self.head_green_raw(x_pool)
        clover = self.head_clover_raw(x_pool)
        dead = self.head_dead_raw(x_pool)

        # Summation logic
        gdm = green + clover
        total = gdm + dead

        return total, gdm, green, clover, dead

############################################################
# Loss
############################################################


def biomass_loss(outputs, labels, w=None):
    total, gdm, green, clover, dead = outputs
    mse = nn.MSELoss()
    # Huber loss for robust regression (beta=5.0 as recommended)
    huber = nn.SmoothL1Loss(beta=5.0)

    l_green = huber(green.squeeze(),  labels[:, 0])
    l_dead = huber(dead.squeeze(), labels[:, 1])  # Use Huber loss for Dead
    l_clover = huber(clover.squeeze(), labels[:, 2])
    l_gdm = huber(gdm.squeeze(),    labels[:, 3])
    l_total = huber(total.squeeze(),  labels[:, 4])

    # Stack per-target losses in the SAME order as CFG.ALL_TARGET_COLS
    losses = torch.stack([l_green, l_dead, l_clover, l_gdm, l_total])
    # losses = torch.stack([l_green, l_dead, l_clover])

    # Use provided weights, or default to CFG.R2_WEIGHTS
    if w is None:
        return losses.mean()
    w = torch.as_tensor(w, device=losses.device, dtype=losses.dtype)
    w = w / w.sum()
    return (losses * w).sum()


# Train Functions with EMA & Gradient Accumulation

scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())


@torch.no_grad()
def valid_epoch(eval_model, loader, device):
    eval_model.eval()
    running = 0.0
    preds_total, preds_gdm, preds_green, preds_clover, preds_dead, all_labels = [
    ], [], [], [], [], []
    amp_ctx = (lambda: torch.amp.autocast(device_type='cuda')
               ) if torch.cuda.is_available() else (lambda: nullcontext())

    for l, r, lab in loader:
        l, r, lab = l.to(device, non_blocking=True), r.to(
            device, non_blocking=True), lab.to(device, non_blocking=True)
        with amp_ctx():
            p_total, p_gdm, p_green, p_clover, p_dead = eval_model(l, r)
            loss = biomass_loss(
                (p_total, p_gdm, p_green, p_clover, p_dead), lab, w=CFG.LOSS_WEIGHTS)
        running += loss.item() * l.size(0)
        preds_total.extend(p_total.cpu().numpy().ravel())
        preds_gdm.extend(p_gdm.cpu().numpy().ravel())
        preds_green.extend(p_green.cpu().numpy().ravel())
        preds_clover.extend(p_clover.cpu().numpy().ravel())
        preds_dead.extend(p_dead.cpu().numpy().ravel())
        all_labels.extend(lab.cpu().numpy())

    pred_total = np.array(preds_total)
    pred_gdm = np.array(preds_gdm)
    pred_green = np.array(preds_green)
    pred_clover = np.array(preds_clover)
    pred_dead = np.array(preds_dead)
    true_labels = np.stack(all_labels)

    pred_all = np.stack(
        [pred_green, pred_dead, pred_clover, pred_gdm, pred_total], axis=1)
    global_r2, avg_r2, per_r2 = weighted_r2_score_global(true_labels, pred_all)
    return running / len(loader.dataset), global_r2, avg_r2, per_r2, pred_all, true_labels


@torch.no_grad()
def valid_epoch_tta(eval_model, loaders, device):
    eval_model.eval()
    amp_ctx = (lambda: torch.amp.autocast(device_type='cuda')
               ) if torch.cuda.is_available() else (lambda: nullcontext())

    # We need to aggregate predictions from all loaders
    # Assuming all loaders have same order and size (which they should if shuffle=False)

    all_preds_accum = None
    all_labels = None
    total_loss = 0.0

    for loader_idx, loader in enumerate(loaders):
        preds_total, preds_gdm, preds_green, preds_clover, preds_dead = [], [], [], [], []
        current_labels = []
        running_loss = 0.0

        for l, r, lab in loader:
            l, r, lab = l.to(device, non_blocking=True), r.to(
                device, non_blocking=True), lab.to(device, non_blocking=True)
            with amp_ctx():
                p_total, p_gdm, p_green, p_clover, p_dead = eval_model(l, r)
                loss = biomass_loss(
                    (p_total, p_gdm, p_green, p_clover, p_dead), lab, w=CFG.LOSS_WEIGHTS)

            running_loss += loss.item() * l.size(0)

            preds_total.extend(p_total.cpu().numpy().ravel())
            preds_gdm.extend(p_gdm.cpu().numpy().ravel())
            preds_green.extend(p_green.cpu().numpy().ravel())
            preds_clover.extend(p_clover.cpu().numpy().ravel())
            preds_dead.extend(p_dead.cpu().numpy().ravel())

            if loader_idx == 0:
                current_labels.extend(lab.cpu().numpy())

        total_loss += (running_loss / len(loader.dataset))

        # Stack predictions for this loader: (N, 5)
        # Order: Green, Dead, Clover, GDM, Total (matching CFG.ALL_TARGET_COLS order roughly, but let's be precise)
        # CFG.ALL_TARGET_COLS = ['Dry_Green_g','Dry_Dead_g','Dry_Clover_g','GDM_g','Dry_Total_g']
        # preds lists are just raw outputs.
        # Let's stack them in the order expected by weighted_r2_score_global which expects:
        # y_true, y_pred where columns match.
        # The model returns: total, gdm, green, clover, dead
        # We need to stack them to match true_labels which comes from CFG.ALL_TARGET_COLS
        # CFG.ALL_TARGET_COLS is ['Dry_Green_g','Dry_Dead_g','Dry_Clover_g','GDM_g','Dry_Total_g']

        pred_stack = np.stack([
            np.array(preds_green),
            np.array(preds_dead),
            np.array(preds_clover),
            np.array(preds_gdm),
            np.array(preds_total)
        ], axis=1)

        if all_preds_accum is None:
            all_preds_accum = pred_stack
            all_labels = np.stack(current_labels)
        else:
            all_preds_accum += pred_stack

    # Average predictions
    avg_preds = all_preds_accum / len(loaders)
    avg_loss = total_loss / len(loaders)

    global_r2, avg_r2, per_r2 = weighted_r2_score_global(all_labels, avg_preds)
    return avg_loss, global_r2, avg_r2, per_r2, avg_preds, all_labels


def set_backbone_requires_grad(model: BiomassModel, requires_grad: bool):
    for p in model.backbone.parameters():
        p.requires_grad = requires_grad


def build_optimizer(model: BiomassModel):
    head_params = (list(model.head_green_raw.parameters()) +
                   list(model.head_clover_raw.parameters()) +
                   list(model.head_dead_raw.parameters()))
    backbone_params = list(model.backbone.parameters())
    return optim.AdamW([
        {'params': backbone_params, 'lr': CFG.LR_BACKBONE, 'weight_decay': CFG.WD},
        {'params': head_params,     'lr': CFG.LR_HEAD,     'weight_decay': CFG.WD},
    ])


def build_optimizer(model: BiomassModel):
    # 1. Get backbone parameter IDs for exclusion
    backbone_ids = {id(p) for p in model.backbone.parameters()}

    # 2. Separate params into backbone vs. everything else (heads, fusion, etc.)
    backbone_params = []
    rest_params = []

    for p in model.parameters():
        if p.requires_grad:
            if id(p) in backbone_ids:
                backbone_params.append(p)
            else:
                rest_params.append(p)

    return optim.AdamW([
        {'params': backbone_params, 'lr': CFG.LR_BACKBONE, 'weight_decay': CFG.WD},
        {'params': rest_params,     'lr': CFG.LR_REST,     'weight_decay': CFG.WD},
    ])


def build_scheduler(optimizer):
    def lr_lambda(epoch):
        e = max(0, epoch - 1)
        if e < CFG.WARMUP_EPOCHS:
            return float(e + 1) / float(max(1, CFG.WARMUP_EPOCHS))
        progress = (e - CFG.WARMUP_EPOCHS) / \
            float(max(1, CFG.EPOCHS - CFG.WARMUP_EPOCHS))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, opt, scheduler, device, ema: ModelEmaV2 | None = None):
    model.train()
    running = 0.0
    opt.zero_grad()
    amp_ctx = (lambda: torch.amp.autocast(device_type='cuda')
               ) if torch.cuda.is_available() else (lambda: nullcontext())
    itera = tqdm(loader, desc='train', leave=False) if CFG.USE_TQDM else loader
    for i, (l, r, lab) in enumerate(itera):
        l, r, lab = l.to(device, non_blocking=True), r.to(
            device, non_blocking=True), lab.to(device, non_blocking=True)
        with amp_ctx():
            total, gdm, green, clover, dead = model(l, r)
            loss = biomass_loss((total, gdm, green, clover, dead),
                                lab, w=CFG.LOSS_WEIGHTS) / CFG.GRAD_ACC
        scaler.scale(loss).backward()
        running += loss.item() * l.size(0) * CFG.GRAD_ACC

        if (i + 1) % CFG.GRAD_ACC == 0 or (i + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(model.module if hasattr(model, 'module') else model)
            opt.zero_grad()
    scheduler.step()
    return running / len(loader.dataset)


# 5-Fold Training Loop with EMA

print('Loading data...')
df_long = pd.read_csv(CFG.TRAIN_CSV)
df_wide = df_long.pivot(index='image_path', columns='target_name', values='target').reset_index()
assert df_wide['image_path'].is_unique, 'Leakage risk: duplicate image_path rows'

# Merge metadata (Sampling_Date, State) for stratification
if 'Sampling_Date' in df_long.columns and 'State' in df_long.columns:
    print('Merging metadata for stratification...')
    meta_df = df_long[['image_path', 'Sampling_Date', 'State']].drop_duplicates()
    df_wide = df_wide.merge(meta_df, on='image_path', how='left')

# Keep necessary columns
df_wide = df_wide[['image_path', 'Sampling_Date', 'State'] + CFG.ALL_TARGET_COLS]
print(f'{len(df_wide)} training images')

# Use StratifiedGroupKFold
sgkf = StratifiedGroupKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
oof_true, oof_pred, fold_summary = [], [], []

# Split based on groups (Sampling_Date) and stratification target (State)
groups = df_wide['Sampling_Date']
y_stratify = df_wide['State']

# models_list = [] # Removed to save memory

for fold, (tr_idx, val_idx) in enumerate(sgkf.split(df_wide, y_stratify, groups=groups)):
    if fold not in CFG.FOLDS_TO_TRAIN:
        print(f'Skipping fold {fold} as per configuration.')
        continue
    print('\n' + '='*70)
    print(f'FOLD {fold+1}/{CFG.N_FOLDS} | {len(tr_idx)} train / {len(val_idx)} val')
    print('='*70)
    torch.cuda.empty_cache(); gc.collect()
    
    tr_df  = df_wide.iloc[tr_idx].reset_index(drop=True)
    val_df = df_wide.iloc[val_idx].reset_index(drop=True)

    # Quick train/val comparison for this fold
    try:
        compare_train_val(tr_df, val_df, CFG.ALL_TARGET_COLS, show_plots=True)
    except Exception as e:
        print('Warning: compare_train_val failed:', e)

    tr_set = BiomassDataset(tr_df,  get_train_transforms(), CFG.TRAIN_IMAGE_DIR)
    
    # Create TTA loaders
    val_loaders = []
    for mode in range(CFG.VAL_TTA_TIMES): # 0: orig, 1: hflip, 2: vflip, 3: rot90
        val_set_tta = BiomassDataset(val_df, get_tta_transforms(mode), CFG.TRAIN_IMAGE_DIR)
        val_loader_tta = DataLoader(val_set_tta, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)
        val_loaders.append(val_loader_tta)

    tr_loader  = DataLoader(tr_set, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)
    
    print('Building model...')
    backbone_path = getattr(CFG, 'BACKBONE_PATH', None)
    base_model = BiomassModel(CFG.MODEL_NAME, pretrained=CFG.PRETRAINED, backbone_path=backbone_path).to(CFG.DEVICE)
    
    # Load pretrained fold weights if available (for resuming or fine-tuning)
    if getattr(CFG, 'PRETRAINED_DIR', None) and os.path.isdir(CFG.PRETRAINED_DIR):
        pretrained_path = os.path.join(CFG.PRETRAINED_DIR, f'best_model_fold{fold}.pth')
        if os.path.exists(pretrained_path):
            try:
                state = torch.load(pretrained_path, map_location='cpu')
                # support raw state_dict or dict-with-keys
                if isinstance(state, dict) and ('model_state_dict' in state or 'state_dict' in state):
                    key = 'model_state_dict' if 'model_state_dict' in state else 'state_dict'
                    sd = state[key]
                else:
                    sd = state
                base_model.load_state_dict(sd, strict=False)
                base_model.to(CFG.DEVICE)
                print(f'  ✓ Loaded pretrained weights for fold {fold} from {pretrained_path}')
            except Exception as e:
                print(f'  ✗ Failed to load pretrained fold {fold}: {e}')
        else:
            print(f'  (No pretrained file for fold {fold} at {pretrained_path})')
    else:
        print('  (No PRETRAINED_DIR configured or directory missing)')
        
    model = nn.DataParallel(base_model)
    set_backbone_requires_grad(base_model, False)
    optimizer = build_optimizer(base_model)
    scheduler = build_scheduler(optimizer)
    ema = ModelEmaV2(base_model, decay=CFG.EMA_DECAY)
    
    best_global_r2 = -np.inf
    patience = 0
    best_fold_preds = None; best_fold_true = None
    best_avg_r2 = -np.inf
    
    # Define save path
    if not os.path.exists(CFG.MODEL_DIR):
        os.makedirs(CFG.MODEL_DIR)
    save_path = os.path.join(CFG.MODEL_DIR, f'best_model_fold{fold}.pth')
    
    for epoch in range(1, CFG.EPOCHS + 1):
        if epoch == CFG.FREEZE_EPOCHS + 1:
            patience = 0
            set_backbone_requires_grad(base_model, True)
            print(f'Epoch {epoch}: backbone unfrozen')
        
        tr_loss = train_epoch(model, tr_loader, optimizer, scheduler, CFG.DEVICE, ema)
        eval_model = ema.module if ema is not None else (model.module if hasattr(model, 'module') else model)
        
        # Use TTA validation
        val_loss, global_r2, avg_r2, per_r2, preds_fold, true_fold = valid_epoch_tta(eval_model, val_loaders, CFG.DEVICE)
        
        per_r2_str = ' | '.join([f'{CFG.ALL_TARGET_COLS[i][:5]}: {r2:.3f}' for i, r2 in enumerate(per_r2)])
        lrs = [pg['lr'] for pg in optimizer.param_groups]
        print(f'Fold {fold} | Epoch {epoch:02d} | TLoss {tr_loss:.5f} | VLoss {val_loss:.5f} |avgR2 {avg_r2:.4f}| GlobalR² {global_r2:.4f} {"[BEST]" if global_r2 > best_global_r2 else ""}')
        print(f'  → {per_r2_str}')
        
        if global_r2 > best_global_r2:
            best_global_r2 = global_r2
            best_avg_r2 = avg_r2
            
            # Save the EMA weights (best state) to disk immediately
            # Clone to CPU to avoid memory issues
            best_state = {k: v.cpu().clone() for k, v in eval_model.state_dict().items()}
            torch.save(best_state, save_path)
            print(f'  → SAVED EMA weights to {save_path} (GlobalR²: {best_global_r2:.4f})')
            del best_state # Free memory
            
            patience = 0
            best_fold_preds = preds_fold; best_fold_true = true_fold
        else:
            patience += 1
            if patience >= CFG.PATIENCE:
                    print(f'  → EARLY STOP (no improvement in {CFG.PATIENCE} epochs)')
                    break
                
        del preds_fold, true_fold
        torch.cuda.empty_cache()
        gc.collect()
    
    if best_fold_preds is not None:
        oof_true.append(best_fold_true); oof_pred.append(best_fold_preds)
        fold_summary.append({'fold': fold, 'global_r2': best_global_r2,'avg_r2':avg_r2})
    
    # Cleanup for this fold
    del model, base_model, tr_loader, val_loaders, optimizer, scheduler, ema
    if 'eval_model' in locals(): del eval_model
    torch.cuda.empty_cache(); gc.collect()

if oof_true:
    oof_true_arr = np.concatenate(oof_true, axis=0)
    oof_pred_arr = np.concatenate(oof_pred, axis=0)
    oof_global_r2, oof_avg_r2, oof_per_r2 = weighted_r2_score_global(oof_true_arr, oof_pred_arr)

    print('\nTraining complete! Models saved in:', CFG.MODEL_DIR)
    print('Fold summary:')
    for fs in fold_summary:
        print(f"  Fold {fs['fold']}: Global R² = {fs['global_r2']:.4f}, Avg R² = {fs.get('avg_r2', float('nan')):.4f}")
    print(f'OOF Global Weighted R²: {oof_global_r2:.4f} | OOF Avg Target R²: {oof_avg_r2:.4f}')
    print('OOF Per-target:', dict(zip(CFG.ALL_TARGET_COLS, [f"{r:.4f}" for r in oof_per_r2])))
else:
    print('No OOF predictions collected.')


# Submit

# ===============================================================
# 4. DEFINE TTA TRANSFORMS
# ===============================================================
def get_tta_transforms(num_transforms):
    """
    Returns a list of TTA transform pipelines.
    Each pipeline represents a different augmentation view.
    """
    normalize = A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    to_tensor = ToTensorV2()
    
    all_tta_transforms = [
        # View 1: Original (no flip)
        A.Compose([
            A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
            normalize,
            to_tensor
        ]),
        
        # View 2: Horizontal Flip
        A.Compose([
            A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
            A.HorizontalFlip(p=1.0),
            normalize,
            to_tensor
        ]),
        
        # View 3: Vertical Flip
        A.Compose([
            A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
            A.VerticalFlip(p=1.0),
            normalize,
            to_tensor
        ]),
        
        # View 4: Both Flips
        A.Compose([
            A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            normalize,
            to_tensor
        ]),
    ]
    tta_transforms = all_tta_transforms[:num_transforms]
    return tta_transforms

print(f"✓ TTA transforms defined ({CFG.TTA_STEPS} views)")

# ===============================================================
# 5. CREATE TEST DATASET
# ===============================================================
def clean_image(img):
    # Safe crop (remove bottom artifacts) + inpaint orange date stamp
    h, w = img.shape[:2]
    img = img[0:int(h * 0.90), :]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower = np.array([5, 150, 150])
    upper = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    if np.sum(mask) > 0:
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return img
class BiomassTestDataset(Dataset):
    """
    Test dataset for biomass images.
    Splits each 2000×1000 image into left and right 1000×1000 halves.
    """
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.paths = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.filenames = [os.path.basename(p) for p in self.paths]
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        
        # Read image
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not read {path}, using blank image")
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = clean_image(img)
        # Split into left and right halves
        h, w = img.shape[:2]
        mid = w // 2
        left = img[:, :mid].copy()
        right = img[:, mid:].copy()
        
        return left, right, self.filenames[idx]

print("✓ Test dataset class defined")

# ===============================================================
# 8. RUN INFERENCE WITH TTA (UPDATED to honor CFG.FOLDS_TO_TRAIN)
# ===============================================================

@torch.no_grad()
def predict_with_tta(model, left_np, right_np, tta_transforms):
    """
    Predict using TTA with a SINGLE model.
    
    Args:
        model: Single trained model
        left_np: Left half of image (numpy array)
        right_np: Right half of image (numpy array)
        tta_transforms: List of augmentation transforms
    
    Returns:
        numpy array: [total, gdm, green] predictions (averaged over TTA)
    """
    all_tta_preds = []
    
    # Loop over TTA views
    for tfm in tta_transforms:
        # Apply transform to both halves
        left_tensor = tfm(image=left_np)['image'].unsqueeze(0).to(CFG.DEVICE)
        right_tensor = tfm(image=right_np)['image'].unsqueeze(0).to(CFG.DEVICE)
        
        total, gdm, green, clover, dead = model(left_tensor, right_tensor)
        
        # Extract values
        p_total = total.cpu().item()
        p_gdm = gdm.cpu().item()
        p_green = green.cpu().item()
        
        all_tta_preds.append([p_total, p_gdm, p_green])
    
    # Average across TTA views
    final_pred = np.mean(all_tta_preds, axis=0)
    
    return final_pred


def run_inference():
    """
    Main inference function.
    Returns: (predictions_array, image_filenames)
    Notes:
      - Now respects `CFG.FOLDS_TO_TRAIN` (if set) and averages only over successfully loaded folds.
      - If no fold weights are found for the requested folds, an error is raised.
    """
    print("\n" + "="*70)
    print("STARTING INFERENCE")
    print("="*70)
    
    # Create dataset and loader
    dataset = BiomassTestDataset(CFG.TEST_IMAGE_DIR)
    # Note: batch_size=1 is required for the current predict_with_tta implementation
    loader = DataLoader(
        dataset,
        batch_size=1,  
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True
    )
    
    tta_transforms = get_tta_transforms(CFG.TTA_STEPS)
    
    # Initialize accumulator for predictions
    # Shape: (num_samples, 3) for Total, GDM, Green
    accumulated_preds = np.zeros((len(dataset), 3), dtype=np.float32)

    # Use configured folds, fallback to full range if not set or empty
    folds_to_use = getattr(CFG, 'FOLDS_TO_TRAIN', list(range(CFG.N_FOLDS)))
    if not folds_to_use:
        folds_to_use = list(range(CFG.N_FOLDS))

    print(f"Folds requested for inference: {folds_to_use}")

    # Use filenames from dataset (guaranteed consistent ordering with loader because shuffle=False)
    filenames = dataset.filenames.copy()

    successful_folds = 0
    
    # Loop over requested folds only
    for fold in folds_to_use:
        print(f"\nProcessing Fold {fold}...")
        # Load model for this fold
        model_dir = CFG.MODEL_DIR_012 if fold in [0,1,2] else CFG.MODEL_DIR_34
        backbone_path = getattr(CFG, 'BACKBONE_PATH', None)
        model = BiomassModel(CFG.MODEL_NAME, pretrained=False, backbone_path=backbone_path)
        
        # Load weights
        weight_path = os.path.join(model_dir, f'best_model_fold{fold}.pth')
        if not os.path.exists(weight_path):
            print(f"Warning: Model file {weight_path} not found! Skipping fold {fold}.")
            del model
            torch.cuda.empty_cache(); gc.collect()
            continue
            
        state = torch.load(weight_path, map_location='cpu')
        # Handle state dict keys if necessary (e.g. if saved with 'model_state_dict' key)
        if isinstance(state, dict) and ('model_state_dict' in state or 'state_dict' in state):
             key = 'model_state_dict' if 'model_state_dict' in state else 'state_dict'
             sd = state[key]
        else:
             sd = state
        
        model.load_state_dict(sd)
        model.to(CFG.DEVICE)
        model.eval()
        
        # Run inference for this fold
        for i, (left, right, filename) in enumerate(tqdm(loader, desc=f"Fold {fold}")):
            # left and right are batches of size 1, convert to numpy for TTA function
            left_np = left[0].numpy()
            right_np = right[0].numpy()
            
            # Predict
            pred = predict_with_tta(model, left_np, right_np, tta_transforms)
            accumulated_preds[i] += pred
            
        successful_folds += 1
        # Cleanup model to save memory
        del model
        torch.cuda.empty_cache(); gc.collect()
        
    if successful_folds == 0:
        raise FileNotFoundError(f"No model weights found for requested folds: {folds_to_use}")

    # Average predictions over the number of successfully loaded folds
    final_predictions = accumulated_preds / successful_folds
    
    print(f"\nInference complete. Successfully used {successful_folds} fold(s) out of {len(folds_to_use)} requested.")
    return final_predictions, filenames

# ===============================================================
# 9. POST-PROCESS PREDICTIONS
# ===============================================================
def postprocess_predictions(preds_direct):
    """
    Calculate derived targets from direct predictions.
    
    Input: (n_samples, 3) array with [total, gdm, green]
    Output: (n_samples, 5) array with [green, dead, clover, gdm, total]
    """
    print("\nPost-processing predictions...")
    
    # Extract direct predictions
    pred_total = preds_direct[:, 0]
    pred_gdm = preds_direct[:, 1]
    pred_green = preds_direct[:, 2]
    
    # Calculate derived targets with non-negativity constraint
    pred_clover = np.maximum(0, pred_gdm - pred_green)
    pred_dead = np.maximum(0, pred_total - pred_gdm)
    
    # Stack in the order of ALL_TARGET_COLS
    # ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
    preds_all = np.stack([
        pred_green,
        pred_dead,
        pred_clover,
        pred_gdm,
        pred_total
    ], axis=1)
    
    print(f"✓ Post-processing complete")
    print(f"  Output shape: {preds_all.shape}")
    print(f"\nPrediction statistics:")
    for i, col in enumerate(CFG.ALL_TARGET_COLS):
        print(f"  {col:15s}: mean={preds_all[:, i].mean():.2f}, "
              f"std={preds_all[:, i].std():.2f}, "
              f"min={preds_all[:, i].min():.2f}, "
              f"max={preds_all[:, i].max():.2f}")
    
    return preds_all

# ===============================================================
# 10. CREATE SUBMISSION FILE (FIXED)
# ===============================================================
def create_submission(predictions, filenames):
    """
    Create submission file in the required format.
    
    Args:
        predictions: (n_images, 5) array with all target predictions
        filenames: list of test image filenames
    """
    print("\n" + "="*70)
    print("CREATING SUBMISSION FILE")
    print("="*70)
    
    # Step 0: Load test.csv first to check the image_path format
    test_df = pd.read_csv(CFG.TEST_CSV)
    print(f"\nTest CSV loaded: {len(test_df)} rows")
    print(f"Sample image_path from test.csv: {test_df['image_path'].iloc[0]}")
    print(f"Sample filename from predictions: {filenames[0]}")
    
    # Step 1: Fix image_path format to match test.csv
    # If test.csv has "test/ID123.jpg" but we have "ID123.jpg", add the prefix
    test_path_example = test_df['image_path'].iloc[0]
    if '/' in test_path_example:
        # Extract the subdirectory prefix (e.g., "test/")
        prefix = test_path_example.rsplit('/', 1)[0] + '/'
        corrected_filenames = [prefix + fn for fn in filenames]
        print(f"Corrected path format: {corrected_filenames[0]}")
    else:
        corrected_filenames = filenames
    
    # Step 2: Create wide-format DataFrame with corrected paths
    preds_wide = pd.DataFrame(predictions, columns=CFG.ALL_TARGET_COLS)
    preds_wide.insert(0, 'image_path', corrected_filenames)
    
    print(f"\nWide format predictions:")
    print(preds_wide.head())
    
    # Step 3: Convert to long format (melt)
    preds_long = preds_wide.melt(
        id_vars=['image_path'],
        value_vars=CFG.ALL_TARGET_COLS,
        var_name='target_name',
        value_name='target'
    )
    
    print(f"\nLong format predictions (first 10 rows):")
    print(preds_long.head(10))
    
    # Step 4: Debug the merge
    print(f"\nDebug: Checking if paths match...")
    print(f"Unique paths in test_df: {test_df['image_path'].nunique()}")
    print(f"Unique paths in preds_long: {preds_long['image_path'].nunique()}")
    
    common_paths = set(test_df['image_path'].unique()) & set(preds_long['image_path'].unique())
    print(f"Common paths found: {len(common_paths)}")
    
    if len(common_paths) == 0:
        print("\n❌ ERROR: No matching paths found!")
        print(f"Test CSV paths sample: {list(test_df['image_path'].unique()[:3])}")
        print(f"Prediction paths sample: {list(preds_long['image_path'].unique()[:3])}")
        raise ValueError("Path mismatch between test.csv and predictions")
    
    # Step 5: Merge to get sample_ids
    submission = pd.merge(
        test_df[['sample_id', 'image_path', 'target_name']],
        preds_long,
        on=['image_path', 'target_name'],
        how='left'
    )
    
    # Step 6: Keep only required columns
    submission = submission[['sample_id', 'target']]
    
    # Step 7: Check for missing values
    missing_count = submission['target'].isna().sum()
    if missing_count > 0:
        print(f"\n⚠ Warning: {missing_count} missing predictions found!")
        print("Sample missing entries:")
        print(submission[submission['target'].isna()].head())
        submission.loc[submission['target'].isna(), 'target'] = 0.0
    
    # Step 8: Sort by sample_id
    submission = submission.sort_values('sample_id').reset_index(drop=True)
    
    # Step 9: Save to CSV
    output_path = os.path.join(CFG.SUBMISSION_DIR, 'submission_dinoV3.csv')
    submission.to_csv(output_path, index=False)
    
    print(f"\n✓ Submission file saved: {output_path}")
    print(f"  Total rows: {len(submission)}")
    print(f"\nPrediction statistics:")
    print(f"  Min: {submission['target'].min():.4f}")
    print(f"  Max: {submission['target'].max():.4f}")
    print(f"  Mean: {submission['target'].mean():.4f}")
    print(f"  Non-zero values: {(submission['target'] > 0).sum()}/{len(submission)}")
    
    print(f"\nFirst 10 rows:")
    print(submission.head(10))
    print(f"\nLast 10 rows:")
    print(submission.tail(10))
    
    # Step 10: Validation checks
    print(f"\n" + "="*70)
    print("VALIDATION CHECKS")
    print("="*70)
    print(f"✓ Expected rows: {len(test_df)}")
    print(f"✓ Actual rows: {len(submission)}")
    print(f"✓ Match: {len(submission) == len(test_df)}")
    print(f"✓ No missing values: {not submission['target'].isna().any()}")
    print(f"✓ All sample_ids unique: {submission['sample_id'].is_unique}")
    print(f"✓ Has non-zero predictions: {(submission['target'] > 0).any()}")
    
    return submission

# Create submission
# Post-process predictions
# Run inference
if CFG.CREATE_SUBMISSION:
    predictions_direct, test_filenames = run_inference()
    predictions_all = postprocess_predictions(predictions_direct)
    submission_df = create_submission(predictions_all, test_filenames)
    