import os, gc, math, cv2, numpy as np, pandas as pd
from tqdm import tqdm
import torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
from timm.utils import ModelEmaV2
from sklearn.model_selection import KFold, StratifiedGroupKFold
import matplotlib.pyplot as plt
import seaborn as sns

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")



class CFG:
    CREATE_SUBMISSION = True
    USE_TQDM        = False
    PRETRAINED_DIR  = None
    PRETRAINED      = True
    BASE_PATH       = '/kaggle/input/csiro-biomass'
    SEED            = 82947501
    FOLDS_TO_TRAIN   = [0,1,2,3,4]
    TRAIN_CSV       = os.path.join(BASE_PATH, 'train.csv')
    TRAIN_IMAGE_DIR = os.path.join(BASE_PATH, 'train')
    TEST_IMAGE_DIR = '/kaggle/input/csiro-biomass/test'
    TEST_CSV = '/kaggle/input/csiro-biomass/test.csv'
    SUBMISSION_DIR  = './'
    MODEL_DIR = '/kaggle/input/baseline-dinov3/pytorch/default/1/baseline-dinov3'
    N_FOLDS         = 5

    MODEL_NAME      = 'vit_huge_plus_patch16_dinov3.lvd1689m'
    BACKBONE_PATH   = '/kaggle/input/vit-huge-plus-patch16-dinov3-lvd1689m/vit_huge_plus_patch16_dinov3.lvd1689m_backbone.pth'
    DINO_GRAD_CHECKPOINTING = False

    IMG_SIZE        = 512

    VAL_TTA_TIMES   = 1
    TTA_STEPS       = 1
    
    
    BATCH_SIZE      = 1
    GRAD_ACC        = 4
    NUM_WORKERS     = 4
    EPOCHS          = 1
    FREEZE_EPOCHS   = 0
    WARMUP_EPOCHS   = 3
    LR_REST         = 1e-3
    LR_BACKBONE     = 5e-4
    WD              = 1e-2
    EMA_DECAY       = 0.9
    CLIP_GRAD_NORM  = 1.0
    PATIENCE        = 5
    TARGET_COLS     = ['Dry_Total_g', 'GDM_g', 'Dry_Green_g']
    DERIVED_COLS    = ['Dry_Clover_g', 'Dry_Dead_g']
    ALL_TARGET_COLS = ['Dry_Green_g','Dry_Dead_g','Dry_Clover_g','GDM_g','Dry_Total_g']
    R2_WEIGHTS      = np.array([0.1, 0.1, 0.1, 0.2, 0.5])
    LOSS_WEIGHTS    = np.array([0.1, 0.1, 0.1, 0.0, 0.0])
    DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(CFG.MODEL_DIR, exist_ok=True)
print(f'Device : {CFG.DEVICE}')
print(f'Backbone: {CFG.MODEL_NAME} | Input: {CFG.IMG_SIZE}')
print(f'Freeze Epochs: {CFG.FREEZE_EPOCHS} | Warmup: {CFG.WARMUP_EPOCHS}')
print(f'EMA Decay: {CFG.EMA_DECAY} | Grad Acc: {CFG.GRAD_ACC}')


# Metrics
def weighted_r2_score(y_true: np.ndarray, y_pred: np.ndarray):
    weights = CFG.R2_WEIGHTS
    r2_scores = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]; yp = y_pred[:, i]
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
        print(f'{"Index":<6} | {"Image Path":<40} | {"True":<10} | {"Pred":<10} | {"AbsErr":<10}')
        print('-' * 90)
        
        for idx in top_indices:
            path = val_df.iloc[idx]['image_path']
            path_disp = os.path.basename(path)
            t_val = y_true[idx, i]
            p_val = y_pred[idx, i]
            err = errors[idx]
            print(f'{idx:<6} | {path_disp:<40} | {t_val:<10.4f} | {p_val:<10.4f} | {err:<10.4f}')


def analyze_errors(val_df, y_true, y_pred, targets, top_n=5):
    print(f'\n--- Top {top_n} High Loss Samples per Target ---')
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    for i, target in enumerate(targets):
        errors = np.abs(y_true[:, i] - y_pred[:, i])
        top_indices = np.argsort(errors)[::-1][:top_n]
        
        print(f'\nTarget: {target}')
        header = f'{"Index":<6} | {"Image Path":<40} | {"State":<6} | {"True":<10} | {"Pred":<10} | {"AbsErr":<10}'
        print(header)
        print('-' * len(header))
        
        for idx in top_indices:
            path = val_df.iloc[idx]['image_path']
            path_disp = os.path.basename(path)
            state = val_df.iloc[idx]['State'] if 'State' in val_df.columns else 'NA'
            t_val = y_true[idx, i]
            p_val = y_pred[idx, i]
            err = errors[idx]
            print(f'{idx:<6} | {path_disp:<40} | {str(state):<6} | {t_val:<10.4f} | {p_val:<10.4f} | {err:<10.4f}')
def compare_train_val(tr_df, val_df, targets, show_plots=True):
    """Quick comparison of target distributions and metadata between train and val splits."""
    print("\n--- Train / Val Comparison ---")

    for t in targets:
        tr = tr_df.get(t, pd.Series(dtype=float)).dropna()
        val = val_df.get(t, pd.Series(dtype=float)).dropna()
        print(f"\nTarget: {t}")
        print(f"  Train: n={len(tr)} mean={tr.mean():.3f} std={tr.std():.3f} min={tr.min():.3f} max={tr.max():.3f}")
        print(f"  Val  : n={len(val)} mean={val.mean():.3f} std={val.std():.3f} min={val.min():.3f} max={val.max():.3f}")
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
            val_dates = pd.to_datetime(val_df['Sampling_Date'], errors='coerce')
            print("\nSampling_Date range:")
            print(f"  Train: {tr_dates.min()} -> {tr_dates.max()} (missing {tr_dates.isna().sum()})")
            print(f"  Val  : {val_dates.min()} -> {val_dates.max()} (missing {val_dates.isna().sum()})")
        except Exception as e:
            print('  Could not parse Sampling_Date:', e)
    if 'State' in tr_df.columns:
        print("\nState distribution (train vs val):")
        tr_state = tr_df['State'].value_counts(normalize=True)
        val_state = val_df['State'].value_counts(normalize=True)
        state_df = pd.concat([tr_state, val_state], axis=1, keys=['train', 'val']).fillna(0)

        print(state_df)

    
def biomass_loss(outputs, labels, w=None):
    total, gdm, green, clover, dead = outputs
    mse = nn.MSELoss()
    huber = nn.SmoothL1Loss(beta=5.0) # Huber loss for robust regression (beta=5.0 as recommended)
    
    l_green  = huber(green.squeeze(),  labels[:,0])
    l_dead   = huber(dead.squeeze(), labels[:,1]) # Use Huber loss for Dead
    l_clover = huber(clover.squeeze(), labels[:,2])
    l_gdm    = huber(gdm.squeeze(),    labels[:,3])
    l_total  = huber(total.squeeze(),  labels[:,4])

    # Stack per-target losses in the SAME order as CFG.ALL_TARGET_COLS
    losses = torch.stack([l_green, l_dead, l_clover, l_gdm, l_total])
    # losses = torch.stack([l_green, l_dead, l_clover])

    # Use provided weights, or default to CFG.R2_WEIGHTS
    if w is None:
        return losses.mean()
    w = torch.as_tensor(w, device=losses.device, dtype=losses.dtype)
    w = w / w.sum()
    return (losses * w).sum()


# Transforms
def get_train_transforms():
    return A.Compose([
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=(-10, 10), p=0.3, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
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
        transforms_list.append(A.RandomRotate90(p=1.0)) # RandomRotate90 with p=1.0 rotates 90, 180, 270 randomly? 
        # Albumentations RandomRotate90 rotates by 90, 180, 270. 
        # Reference uses transforms.RandomRotation([90, 90]) which is exactly 90 degrees.
        # To match exactly 90 degrees in Albumentations, we might need Rotate(limit=(90,90), p=1.0)
        # But RandomRotate90 is standard TTA. Let's use Rotate(limit=(90,90)) to be precise if that's what reference does.
        # Reference: transforms.RandomRotation([90, 90]) -> rotates by exactly 90 degrees.
        transforms_list.append(A.Rotate(limit=(90, 90), p=1.0, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101))

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
    mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=2)

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
    
    
    
# Layers
class LocalMambaBlock(nn.Module):
    """
    Lightweight Mamba-style block (Gated CNN) from the reference notebook.
    Efficiently mixes tokens with linear complexity.
    """
    def __init__(self, dim, kernel_size=5, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # Depthwise conv mixes spatial information locally
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
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
    
    
# Model
class BiomassModel(nn.Module):
    def __init__(self, model_name, pretrained=True, backbone_path=None):
        super().__init__()
        self.model_name = model_name
        self.backbone_path = backbone_path
        
        # 1. Load Backbone with global_pool='' to keep patch tokens
        #    (B, 197, 1024) instead of (B, 1024)
        self.backbone = timm.create_model(self.model_name, pretrained=False, num_classes=0, global_pool='')
        
        # 2. Enable Gradient Checkpointing (Crucial for ViT-Large memory!)
        if hasattr(self.backbone, 'set_grad_checkpointing') and CFG.DINO_GRAD_CHECKPOINTING:
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
        self.head_green_raw  = nn.Sequential(
            nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.2), 
            nn.Linear(nf//2, 1), nn.Softplus()
        )
        self.head_clover_raw = nn.Sequential(
            nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.2), 
            nn.Linear(nf//2, 1), nn.Softplus()
        )
        self.head_dead_raw   = nn.Sequential(
            nn.Linear(nf, nf//2), nn.GELU(), nn.Dropout(0.2), 
            nn.Linear(nf//2, 1), nn.Softplus()
        )
        
        if pretrained:
            self.load_pretrained()
    
    def load_pretrained(self):
        try:
            # Load weights normally
            if self.backbone_path and os.path.exists(self.backbone_path):
                print(f"Loading backbone weights from local file: {self.backbone_path}")
                sd = torch.load(self.backbone_path, map_location='cpu')
                # Handle common checkpoint wrappers (e.g. if saved with 'model' key)
                if 'model' in sd: sd = sd['model']
                elif 'state_dict' in sd: sd = sd['state_dict']
            else:
                # Original behavior: Download from internet
                print("Downloading backbone weights...")
                sd = timm.create_model(self.model_name, pretrained=True, num_classes=0, global_pool='').state_dict()
            
            # Interpolate pos_embed if needed (for 256x256 vs 224x224)
            if 'pos_embed' in sd and hasattr(self.backbone, 'pos_embed'):
                pe_ck = sd['pos_embed']
                pe_m  = self.backbone.pos_embed
                if pe_ck.shape != pe_m.shape:
                    print(f"Interpolating pos_embed: {pe_ck.shape} -> {pe_m.shape}")
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
        green  = self.head_green_raw(x_pool)
        clover = self.head_clover_raw(x_pool)
        dead   = self.head_dead_raw(x_pool)
        
        # Summation logic
        gdm    = green + clover
        total  = gdm + dead
        
        return total, gdm, green, clover, dead
    
    
    # Utility Functions
def set_backbone_requires_grad(model: BiomassModel, requires_grad: bool):
    for p in model.backbone.parameters():
        p.requires_grad = requires_grad


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
        progress = (e - CFG.WARMUP_EPOCHS) / float(max(1, CFG.EPOCHS - CFG.WARMUP_EPOCHS))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)
# Training and Validation Loops 

from contextlib import nullcontext

USE_BF16 = True
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16

scaler = torch.amp.GradScaler(
    'cuda',
    enabled=(torch.cuda.is_available() and AMP_DTYPE == torch.float16)
)

def autocast_ctx():
    if not torch.cuda.is_available():
        return nullcontext()
    return torch.amp.autocast(device_type='cuda', dtype=AMP_DTYPE)

def _as_col(x: torch.Tensor, bs: int) -> torch.Tensor:
    """Force predictions to (B,1) regardless of model head returning (B,) or (B,1)."""
    if x.ndim == 1:
        return x.view(bs, 1)
    if x.ndim == 2:
        return x
    return x.view(bs, 1)

def _ensure_2d_lab(lab: torch.Tensor, bs: int) -> torch.Tensor:
    """Force labels to (B,T)."""
    if lab.ndim == 1:
        return lab.view(bs, -1)
    return lab

def _pred_pack(p_total, p_gdm, p_green, p_clover, p_dead, bs: int) -> torch.Tensor:
    """
    Pack predictions in the exact order expected by your metric:
    CFG.ALL_TARGET_COLS = ['Dry_Green_g','Dry_Dead_g','Dry_Clover_g','GDM_g','Dry_Total_g']
    """
    pg = _as_col(p_green,  bs)
    pd = _as_col(p_dead,   bs)
    pc = _as_col(p_clover, bs)
    pgdm = _as_col(p_gdm,  bs)
    pt = _as_col(p_total,  bs)
    return torch.cat([pg, pd, pc, pgdm, pt], dim=1)  # (B,5)

@torch.inference_mode()
def valid_epoch(eval_model, loader, device):
    eval_model.eval()

    n = len(loader.dataset)
    n_targets = len(CFG.ALL_TARGET_COLS)

    preds_cpu  = torch.empty((n, n_targets), dtype=torch.float32)
    labels_cpu = torch.empty((n, n_targets), dtype=torch.float32)

    total_loss = 0.0
    offset = 0

    for l, r, lab in loader:
        bs = l.size(0)
        l   = l.to(device, non_blocking=True)
        r   = r.to(device, non_blocking=True)
        lab = lab.to(device, non_blocking=True)

        with autocast_ctx():
            p_total, p_gdm, p_green, p_clover, p_dead = eval_model(l, r)
            loss = biomass_loss((p_total, p_gdm, p_green, p_clover, p_dead),
                                lab, w=CFG.LOSS_WEIGHTS)

        total_loss += loss.detach().float().item() * bs

        batch_pred = _pred_pack(p_total, p_gdm, p_green, p_clover, p_dead, bs).float().cpu()
        batch_lab  = _ensure_2d_lab(lab, bs).float().cpu()

        # Safety: ensure dims match (avoids silent 4/5 bugs)
        if batch_pred.shape[1] != n_targets or batch_lab.shape[1] != n_targets:
            raise RuntimeError(
                f"Target dim mismatch: pred={batch_pred.shape}, lab={batch_lab.shape}, "
                f"CFG.ALL_TARGET_COLS={n_targets}"
            )

        preds_cpu[offset:offset+bs]  = batch_pred
        labels_cpu[offset:offset+bs] = batch_lab
        offset += bs

    pred_all    = preds_cpu.numpy()
    true_labels = labels_cpu.numpy()
    global_r2, avg_r2, per_r2 = weighted_r2_score_global(true_labels, pred_all)

    return total_loss / n, global_r2, avg_r2, per_r2, pred_all, true_labels


@torch.inference_mode()
def valid_epoch_tta(eval_model, loaders, device):
    """
    Fixed: always accumulate (N,5). Works even if a head outputs (B,1).
    Assumes each loader iterates in identical order (shuffle=False).
    """
    eval_model.eval()
    assert len(loaders) > 0

    n = len(loaders[0].dataset)
    n_targets = len(CFG.ALL_TARGET_COLS)

    labels_cpu = torch.empty((n, n_targets), dtype=torch.float32)
    preds_sum  = torch.zeros((n, n_targets), dtype=torch.float32)

    total_loss = 0.0

    for tta_i, loader in enumerate(loaders):
        offset = 0
        tta_loss = 0.0

        for l, r, lab in loader:
            bs = l.size(0)
            l   = l.to(device, non_blocking=True)
            r   = r.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)

            with autocast_ctx():
                p_total, p_gdm, p_green, p_clover, p_dead = eval_model(l, r)
                loss = biomass_loss((p_total, p_gdm, p_green, p_clover, p_dead),
                                    lab, w=CFG.LOSS_WEIGHTS)

            tta_loss += loss.detach().float().item() * bs

            batch_pred = _pred_pack(p_total, p_gdm, p_green, p_clover, p_dead, bs).float().cpu()
            preds_sum[offset:offset+bs] += batch_pred

            if tta_i == 0:
                batch_lab = _ensure_2d_lab(lab, bs).float().cpu()
                if batch_lab.shape[1] != n_targets:
                    raise RuntimeError(f"Label dim mismatch: {batch_lab.shape} vs {n_targets}")
                labels_cpu[offset:offset+bs] = batch_lab

            offset += bs

        total_loss += tta_loss / n

    avg_preds   = (preds_sum / len(loaders)).numpy()
    true_labels = labels_cpu.numpy()
    global_r2, avg_r2, per_r2 = weighted_r2_score_global(true_labels, avg_preds)

    return total_loss / len(loaders), global_r2, avg_r2, per_r2, avg_preds, true_labels


def train_epoch(model, loader, opt, scheduler, device, ema: ModelEmaV2 | None = None):
    model.train()
    running = 0.0

    opt.zero_grad(set_to_none=True)
    itera = tqdm(loader, desc='train', leave=False) if CFG.USE_TQDM else loader

    for i, (l, r, lab) in enumerate(itera):
        bs = l.size(0)
        l   = l.to(device, non_blocking=True)
        r   = r.to(device, non_blocking=True)
        lab = lab.to(device, non_blocking=True)

        with autocast_ctx():
            p_total, p_gdm, p_green, p_clover, p_dead = model(l, r)
            loss = biomass_loss((p_total, p_gdm, p_green, p_clover, p_dead),
                                lab, w=CFG.LOSS_WEIGHTS)
            loss = loss / CFG.GRAD_ACC


        if not torch.isfinite(loss):
            print("Non-finite loss detected; skipping batch")
            opt.zero_grad(set_to_none=True)
            continue

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        running += loss.detach().float().item() * bs * CFG.GRAD_ACC

        do_step = ((i + 1) % CFG.GRAD_ACC == 0) or ((i + 1) == len(loader))
        if do_step:
            if scaler.is_enabled():
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            if ema is not None:
                ema.update(model.module if hasattr(model, "module") else model)

            opt.zero_grad(set_to_none=True)

    scheduler.step()
    return running / len(loader.dataset)



# Inference   

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
        model_dir = CFG.MODEL_DIR
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
    output_path = os.path.join(CFG.SUBMISSION_DIR, 'submission70.csv')
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