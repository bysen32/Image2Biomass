# ===============================================================
# Training for CSIRO Biomass (Dual-stream, 3-head)
# k-fold / AMP / Cosine / EMA / Constraint penalty
# ===============================================================
import os, gc, math, random, time
import numpy as np
import pandas as pd
from tqdm import tqdm

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from albumentations import (
    Compose, Resize, HorizontalFlip, VerticalFlip, RandomRotate90,
    ShiftScaleRotate, RandomBrightnessContrast, HueSaturationValue,
    RandomResizedCrop, CoarseDropout, Normalize
)
from albumentations.pytorch import ToTensorV2
import timm

# -------------------------
# 1) Config
# -------------------------
class CFG:
    # paths
    SELECT_BEST_BY = 'r2' 
    BASE_PATH = '/kaggle/input/csiro-biomass'
    TRAIN_CSV = os.path.join(BASE_PATH, 'train.csv')
    TRAIN_IMAGE_DIR = os.path.join(BASE_PATH, 'train')
    OUT_DIR = '.'           # 重み・OOFの保存先
    os.makedirs(OUT_DIR, exist_ok=True)

    # model
    MODEL_NAME = 'convnext_tiny'
    IMG_SIZE = 512
    IN_CHANS = 3
    DUAL_STREAM = True  # True: 左右二流, False: 単一ストリーム（全画像）

    # folds
    N_FOLDS = 5
    SEED = 42

    # train setup
    EPOCHS = 80
    BATCH_SIZE = 16
    NUM_WORKERS = 2
    LR = 3e-4
    WD = 0.05
    WARMUP_EPOCHS = 1
    GRAD_ACCUM = 1
    MAX_NORM = 1.0
    USE_AMP = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # loss / targets
    TARGETS = ['Dry_Total_g', 'GDM_g', 'Dry_Green_g']
    TRAIN_FIVE_OUTPUT_LOSS = False   # 5出力（Green, Dead, Clover, GDM, Total）で学習するか
    USE_LOG1P = False             # log1p変換で安定化

    # ema
    USE_EMA = False
    EMA_DECAY = 0.999

    # deterministic
    DETERMINISTIC = True

    # inference
    INFERENCE_MODE =  True  # False=学習, True=推論
    INFERENCE_MODEL_DIR = "/kaggle/input/convnext-tiny"  # NoneならOUT_DIRを使用、指定時はそのディレクトリからモデルを読み込み
    INFERENCE_BATCH_SIZE = 32
    USE_TTA = False  # Test-time augmentation
    TEST_CSV = os.path.join(BASE_PATH, 'test.csv')
    TEST_IMAGE_DIR = os.path.join(BASE_PATH, 'test')
    SUBMISSION_OUTPUT = os.path.join(OUT_DIR, 'submission.csv')
    INFERENCE_FOLDS = None  # Noneなら全fold自動検出、リスト指定時はそのfoldのみ使用

print(f"Device: {CFG.DEVICE}")

def set_seed(seed=42, deterministic=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(CFG.SEED, CFG.DETERMINISTIC)

# -------------------------
# 2) Augmentations
# -------------------------
def get_train_tf(img_size, aug_strength=1.0):
    """
    Get training augmentations with adjustable strength.
    
    Args:
        img_size: Image size
        aug_strength: Augmentation strength multiplier (1.0 = default, 0.0 = no augmentation, >1.0 = stronger)
    """
    # Scale augmentation parameters by strength
    shift_limit = 0.02 * aug_strength
    scale_limit = 0.1 * aug_strength
    rotate_limit = int(10 * aug_strength)
    hue_shift = int(10 * aug_strength)
    sat_shift = int(10 * aug_strength)
    val_shift = int(10 * aug_strength)
    brightness_limit = 0.15 * aug_strength
    contrast_limit = 0.15 * aug_strength
    dropout_p = min(0.3 * aug_strength, 1.0)
    
    return Compose([
        RandomResizedCrop(size=(img_size, img_size), scale=(0.85, 1.0), ratio=(0.95, 1.05), p=1.0),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.2),
        RandomRotate90(p=0.2),
        ShiftScaleRotate(shift_limit=shift_limit, scale_limit=scale_limit, rotate_limit=rotate_limit, 
                        border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        HueSaturationValue(hue_shift_limit=hue_shift, sat_shift_limit=sat_shift, val_shift_limit=val_shift, p=0.3),
        RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit, p=0.3),
        CoarseDropout(max_holes=4, max_height=int(img_size*0.08), max_width=int(img_size*0.08),
                      min_holes=1, fill_value=0, p=dropout_p),
        Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2()
    ], additional_targets={'image_right': 'image'} if CFG.DUAL_STREAM else {})

def get_valid_tf(img_size):
    return Compose([
        Resize(img_size, img_size),
        Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2()
    ], additional_targets={'image_right': 'image'} if CFG.DUAL_STREAM else {})

# -------------------------
# 3) Dataset (dual-stream: left/right)
# -------------------------
class TrainDataset(Dataset):
    def __init__(self, df, image_dir, tf, use_log1p=True):
        self.df = df.reset_index(drop=True)
        self.paths = self.df['image_path'].values
        self.y = self.df[CFG.TARGETS].values.astype(np.float32)
        self.image_dir = image_dir
        self.tf = tf
        self.use_log1p = use_log1p

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        raw_path = self.paths[idx]
        # Use provided path if it exists; otherwise, fall back to joining with image_dir and basename
        candidate = raw_path if os.path.exists(raw_path) else os.path.join(self.image_dir, os.path.basename(raw_path))
        img = cv2.imread(candidate)
        if img is None:
            img = np.zeros((1000,2000,3), np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if CFG.DUAL_STREAM:
            h, w, _ = img.shape
            mid = w//2
            left = img[:, :mid]
            right = img[:, mid:]
            t = self.tf(image=left, image_right=right)
            left = t['image']
            right = t['image_right']
        else:
            t = self.tf(image=img)
            left = t['image']
            right = left  # インターフェース維持：モデルが単一ストリーム時は右を無視

        target = self.y[idx].copy()
        if self.use_log1p:
            target = np.log1p(target)  # 安定化
        target = torch.from_numpy(target)  # [3]

        return left, right, target

# -------------------------
# 4) Model (dual-stream, separate heads per target)
# -------------------------
class BiomassModel(nn.Module):
    def __init__(self, model_name='convnext_tiny', pretrained=True, target_names=None, dual_stream=True, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg', in_chans=CFG.IN_CHANS)
        self.target_names = target_names if target_names is not None else ['Dry_Total_g', 'GDM_g', 'Dry_Green_g']
        self.num_outputs = len(self.target_names)
        self.dual_stream = dual_stream
        self.dropout = dropout
        nf = self.backbone.num_features
        self.n_combined_features = nf * 2 if self.dual_stream else nf
        
        # 各ターゲットごとに独立したheadを作成
        for target_name in self.target_names:
            head = nn.Sequential(
                nn.Linear(self.n_combined_features, self.n_combined_features // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.n_combined_features // 2, 1)
            )
            setattr(self, f'head_{target_name.lower().replace("_", "")}', head)

    def forward(self, l, r=None):
        fl = self.backbone(l)
        if self.dual_stream:
            fr = self.backbone(r)
            x = torch.cat([fl, fr], dim=1)
        else:
            x = fl
        
        # 各headから出力を取得して結合
        outputs = []
        for target_name in self.target_names:
            head = getattr(self, f'head_{target_name.lower().replace("_", "")}')
            out = head(x).squeeze(1)  # [B,1] -> [B]
            outputs.append(out)
        
        return torch.stack(outputs, dim=1)  # [B, num_outputs]

# -------------------------
# 5) Loss / Metrics
# -------------------------
class WeightedMSELoss(nn.Module):
    """
    Weighted MSE loss for 3 targets (Total, GDM, Green).
    """
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, predictions, targets):
        self.weights = self.weights.to(predictions.device)
        # predictions/targets: [B, 3]
        mse_per_target = (predictions - targets) ** 2
        weighted_mse = mse_per_target * self.weights.unsqueeze(0)
        return weighted_mse.mean()

class ConstraintLoss(nn.Module):
    """
    主損失: L1(予測, 正解)
    補助: 物理制約違反ペナルティ
      - monotonic: Total >= GDM >= Green  （違反分だけReLUで加算）
      - non-neg:   各出力 >= 0            （違反分だけReLUで加算）
    ログ空間で学習している場合、逆変換 expm1 してから制約評価。
    """
    def __init__(self, l1_w=1.0, cons_w=0.1, nonneg_w=0.05, use_log1p=True):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.l1_w = l1_w
        self.cons_w = cons_w
        self.nonneg_w = nonneg_w
        self.use_log1p = use_log1p

    def forward(self, pred, target):
        # pred/target: tuple of three tensors (B,)
        pT, pGDM, pGR = pred
        tT, tGDM, tGR = target[:,0], target[:,1], target[:,2]

        # 主損失：ログ空間ならそのままL1
        loss_main = self.l1(pT, tT) + self.l1(pGDM, tGDM) + self.l1(pGR, tGR)
        loss_main = self.l1_w * loss_main / 3.0

        # 制約は実数空間で評価
        if self.use_log1p:
            PT = torch.expm1(pT)
            PG = torch.expm1(pGDM)
            PR = torch.expm1(pGR)
        else:
            PT, PG, PR = pT, pGDM, pGR

        zero = torch.zeros_like(PT)
        # monotonic violation
        v1 = torch.relu(PG - PT)   # want PT >= PG
        v2 = torch.relu(PR - PG)   # want PG >= PR
        loss_cons = (v1 + v2).mean() * self.cons_w

        # non-negative violation
        n1 = torch.relu(-PT); n2 = torch.relu(-PG); n3 = torch.relu(-PR)
        loss_nonneg = (n1 + n2 + n3).mean() * self.nonneg_w

        return loss_main + loss_cons + loss_nonneg

def rmse_torch(y_pred, y_true):
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))

def metric_rmse(pred_tuple, target, use_log1p=True):
    pT, pGDM, pGR = pred_tuple
    tT, tGDM, tGR = target[:,0], target[:,1], target[:,2]
    if use_log1p:
        pT, pGDM, pGR = [torch.expm1(x) for x in (pT, pGDM, pGR)]
        tT, tGDM, tGR = [torch.expm1(x) for x in (tT, tGDM, tGR)]
    rmse_T = rmse_torch(pT, tT)
    rmse_G = rmse_torch(pGDM, tGDM)
    rmse_R = rmse_torch(pGR, tGR)
    return (rmse_T + rmse_G + rmse_R) / 3.0, (rmse_T, rmse_G, rmse_R)


# =========================
# CSIRO weighted R2 metric
# =========================
CSIRO_WEIGHTS = {
    'Dry_Green_g': 0.10,
    'Dry_Dead_g':  0.10,
    'Dry_Clover_g':0.10,
    'GDM_g':       0.20,
    'Dry_Total_g': 0.50,
}

# 5出力の順序（学習用）
FIVE_TARGET_ORDER = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

def _r2_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]; y_pred = y_pred[m]
    if y_true.size == 0: return np.nan
    rss = np.sum((y_true - y_pred) ** 2)
    tss = np.sum((y_true - y_true.mean()) ** 2)
    if tss <= 0:
        return 1.0 if np.allclose(y_true, y_pred) else 0.0
    return 1.0 - rss / tss

def _five_from_three(total, gdm, green):
    clover = gdm - green
    dead   = total - gdm
    return {
        'Dry_Green_g':  green,
        'Dry_Clover_g': clover,
        'Dry_Dead_g':   dead,
        'GDM_g':        gdm,
        'Dry_Total_g':  total,
    }

def csiro_weighted_r2_from_three_tensors(
    p_total, p_gdm, p_green,
    t_total, t_gdm, t_green,
    use_log1p=True
):
    """
    3ターゲットのTensor([N])から、5ターゲットに拡張して重み付きR²を返す。
    戻り: overall(float), per_target(dict)
    """
    to_np = lambda x: x.detach().cpu().numpy()
    # log1p学習なら実数空間に戻してから評価
    if use_log1p:
        p_total, p_gdm, p_green = [torch.expm1(x) for x in (p_total, p_gdm, p_green)]
        t_total, t_gdm, t_green = [torch.expm1(x) for x in (t_total, t_gdm, t_green)]

    y_true = _five_from_three(to_np(t_total), to_np(t_gdm), to_np(t_green))
    y_pred = _five_from_three(to_np(p_total), to_np(p_gdm), to_np(p_green))

    per = {k: _r2_1d(y_true[k], y_pred[k]) for k in y_true.keys()}
    wsum = sum(CSIRO_WEIGHTS.values())
    overall = float(np.nansum([CSIRO_WEIGHTS[k]/wsum * per[k] for k in per.keys()]))
    return overall, per


def csiro_weighted_r2_from_five_tensors(
    pred5: torch.Tensor,
    true5: torch.Tensor,
    columns=FIVE_TARGET_ORDER,
    use_log1p=True
):
    """
    5ターゲット（列順 columns）から重み付きR²を返す。
    戻り: overall(float), per_target(dict)
    """
    if use_log1p:
        pred5 = torch.expm1(pred5)
        true5 = torch.expm1(true5)

    to_np = lambda x: x.detach().cpu().numpy()
    p = to_np(pred5)
    t = to_np(true5)

    per = {}
    for j, name in enumerate(columns):
        per[name] = _r2_1d(t[:, j], p[:, j])
    wsum = sum(CSIRO_WEIGHTS.values())
    overall = float(np.nansum([CSIRO_WEIGHTS[k]/wsum * per[k] for k in columns]))
    return overall, per


def build_five_from_three_tensors(pred_tuple, target, use_log1p=True):
    """
    3出力のテンソルから、学習用の5出力テンソルを構築する。
    - 入力: pred_tuple = (pT, pGDM, pGR) 各[B]
            target: [B,3] (列順は Total, GDM, Green)
    - 変換は実数空間で行う（log1p学習時は expm1 で戻す）
    - 出力: (pred5[B,5], target5[B,5]) with order = FIVE_TARGET_ORDER
    """
    pT, pGDM, pGR = pred_tuple
    tT, tGDM, tGR = target[:,0], target[:,1], target[:,2]

    if use_log1p:
        PT = torch.expm1(pT); PG = torch.expm1(pGDM); PR = torch.expm1(pGR)
        TT = torch.expm1(tT); TG = torch.expm1(tGDM); TR = torch.expm1(tGR)
    else:
        PT, PG, PR = pT, pGDM, pGR
        TT, TG, TR = tT, tGDM, tGR

    # 5出力（Green, Dead, Clover, GDM, Total）
    pred_dead = PT - PG
    pred_clover = PG - PR
    tgt_dead = TT - TG
    tgt_clover = TG - TR

    pred_map = {
        'Dry_Green_g': PR,
        'Dry_Dead_g': pred_dead,
        'Dry_Clover_g': pred_clover,
        'GDM_g': PG,
        'Dry_Total_g': PT,
    }
    tgt_map = {
        'Dry_Green_g': TR,
        'Dry_Dead_g': tgt_dead,
        'Dry_Clover_g': tgt_clover,
        'GDM_g': TG,
        'Dry_Total_g': TT,
    }

    pred5 = torch.stack([pred_map[k] for k in FIVE_TARGET_ORDER], dim=1)
    tgt5 = torch.stack([tgt_map[k] for k in FIVE_TARGET_ORDER], dim=1)
    return pred5, tgt5


# -------------------------
# 6) EMA
# -------------------------
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = BiomassModel(CFG.MODEL_NAME, pretrained=False, target_names=model.target_names, 
                                dual_stream=model.dual_stream, dropout=model.dropout).to(CFG.DEVICE)
        self.ema.load_state_dict(model.state_dict())
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            v.copy_(v * d + (1. - d) * msd[k])

# -------------------------
# 7) Utilities
# -------------------------
def kfold_split(df, n_folds=5, seed=42):
    from sklearn.model_selection import KFold
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df['fold'] = -1
    for i, (_, val_idx) in enumerate(kf.split(df)):
        df.loc[val_idx, 'fold'] = i
    return df

def save_checkpoint(model, path):
    sd = model.state_dict()
    torch.save(sd, path)

# -------------------------
# 8) Train One Fold
# -------------------------
def train_one_fold(df, fold, lr=None, batch_size=None, wd=None, warmup_epochs=None, dropout=None, aug_strength=None):
    """
    Train one fold with optional parameter overrides.
    
    Args:
        df: DataFrame with fold column
        fold: Fold number to train
        lr: Learning rate override (default: CFG.LR)
        batch_size: Batch size override (default: CFG.BATCH_SIZE)
        wd: Weight decay override (default: CFG.WD)
        warmup_epochs: Warmup epochs override (default: CFG.WARMUP_EPOCHS)
        dropout: Dropout rate override (default: 0.3)
        aug_strength: Augmentation strength override (default: 1.0)
    """
    # Use overrides if provided, otherwise use CFG defaults
    train_lr = lr if lr is not None else CFG.LR
    train_batch_size = batch_size if batch_size is not None else CFG.BATCH_SIZE
    train_wd = wd if wd is not None else CFG.WD
    train_warmup_epochs = warmup_epochs if warmup_epochs is not None else CFG.WARMUP_EPOCHS
    train_dropout = dropout if dropout is not None else 0.3
    train_aug_strength = aug_strength if aug_strength is not None else 1.0
    
    print(f"\n===== FOLD {fold} / {CFG.N_FOLDS} =====")
    print(f"  LR={train_lr:.2e}, BATCH_SIZE={train_batch_size}, WD={train_wd:.4f}")
    print(f"  WARMUP_EPOCHS={train_warmup_epochs}, DROPOUT={train_dropout:.2f}, AUG_STRENGTH={train_aug_strength:.2f}")
    trn_df = df[df['fold'] != fold].reset_index(drop=True)
    val_df = df[df['fold'] == fold].reset_index(drop=True)

    train_tf = get_train_tf(CFG.IMG_SIZE, aug_strength=train_aug_strength)
    valid_tf = get_valid_tf(CFG.IMG_SIZE)

    trn_ds = TrainDataset(trn_df, CFG.TRAIN_IMAGE_DIR, train_tf, use_log1p=CFG.USE_LOG1P)
    val_ds = TrainDataset(val_df, CFG.TRAIN_IMAGE_DIR, valid_tf, use_log1p=CFG.USE_LOG1P)

    # Deterministic worker seeding
    def seed_worker(worker_id):
        s = torch.initial_seed() % 2**32
        np.random.seed(s)
        random.seed(s)
    g = torch.Generator()
    g.manual_seed(CFG.SEED)

    trn_dl = DataLoader(
        trn_ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(CFG.NUM_WORKERS > 0),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=train_batch_size*2,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(CFG.NUM_WORKERS > 0),
    )

    model = BiomassModel(CFG.MODEL_NAME, pretrained=True, target_names=CFG.TARGETS, dual_stream=CFG.DUAL_STREAM, dropout=train_dropout).to(CFG.DEVICE)
    optimizer = AdamW(model.parameters(), lr=train_lr, weight_decay=train_wd)
    # True warmup + cosine, stepped per optimizer step
    steps_per_epoch = max(1, math.ceil(len(trn_dl) / CFG.GRAD_ACCUM))
    warmup_steps = max(1, train_warmup_epochs * steps_per_epoch)
    total_steps = max(1, CFG.EPOCHS * steps_per_epoch)
    cosine_steps = max(1, total_steps - warmup_steps)
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=train_lr*1e-2)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.USE_AMP)
    ema = ModelEMA(model, decay=CFG.EMA_DECAY) if CFG.USE_EMA else None

    # 損失の重み（CFG.TARGETSの順序に基づく）
    weights = [CSIRO_WEIGHTS[k] for k in CFG.TARGETS]
    criterion = WeightedMSELoss(weights=weights)

    select_is_r2 = CFG.SELECT_BEST_BY.lower() == 'r2'
    best_metric = -float('inf') if select_is_r2 else float('inf')
    best_preds = None

    global_step = 0
    for epoch in range(1, CFG.EPOCHS+1):
        model.train()
        train_loss = 0.0

        for i, (l, r, y) in enumerate(tqdm(trn_dl, desc=f"Train ep{epoch}")):
            l = l.to(CFG.DEVICE, non_blocking=True); r = r.to(CFG.DEVICE, non_blocking=True); y = y.to(CFG.DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=CFG.USE_AMP):
                pred = model(l, r)  # [B,K]
                loss = criterion(pred, y) / CFG.GRAD_ACCUM
            scaler.scale(loss).backward()

            if (i+1) % CFG.GRAD_ACCUM == 0:
                if CFG.MAX_NORM is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.MAX_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema: ema.update(model)
                scheduler.step()
                global_step += 1

            train_loss += loss.item()

        # ---- validation (EMA優先) ----
        model.eval()
        eval_model = ema.ema if ema else model
        val_loss = 0.0
        y_pred_all, y_true_all = [], []

        with torch.no_grad():
            for l, r, y in tqdm(val_dl, desc="Valid"):
                l = l.to(CFG.DEVICE, non_blocking=True); r = r.to(CFG.DEVICE, non_blocking=True); y = y.to(CFG.DEVICE, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=CFG.USE_AMP):
                    pred = eval_model(l, r)  # [B,K]
                    loss = criterion(pred, y)
                val_loss += loss.item()
                y_pred_all.append(pred.detach().cpu())  # [B,K]
                y_true_all.append(y.detach().cpu())

        y_pred_all = torch.cat(y_pred_all, dim=0)  # [N,K]
        y_true_all = torch.cat(y_true_all, dim=0)  # [N,K]

        # --- 既存: RMSE（参考） ---
        # RMSE（参考）: 常に Total/GDM/Green の3指標で計算
        idxT = CFG.TARGETS.index('Dry_Total_g')
        idxG = CFG.TARGETS.index('GDM_g')
        idxR = CFG.TARGETS.index('Dry_Green_g')
        rmse_mean, (rmse_T, rmse_G, rmse_R) = metric_rmse(
            (y_pred_all[:,idxT], y_pred_all[:,idxG], y_pred_all[:,idxR]),
            torch.stack([y_true_all[:,idxT], y_true_all[:,idxG], y_true_all[:,idxR]], dim=1),
            use_log1p=CFG.USE_LOG1P
        )

        # --- 新規: コンペ評価(重み付きR²) ---
        if len(CFG.TARGETS) == 5:
            wr2, per_r2 = csiro_weighted_r2_from_five_tensors(
                pred5=y_pred_all,
                true5=y_true_all,
                columns=CFG.TARGETS,
                use_log1p=CFG.USE_LOG1P
            )
        else:
            wr2, per_r2 = csiro_weighted_r2_from_three_tensors(
                p_total=y_pred_all[:,0], p_gdm=y_pred_all[:,1], p_green=y_pred_all[:,2],
                t_total=y_true_all[:,0], t_gdm=y_true_all[:,1], t_green=y_true_all[:,2],
                use_log1p=CFG.USE_LOG1P
            )

        print(f"[Fold {fold}] Epoch {epoch}: "
              f"train_loss={train_loss/len(trn_dl):.4f}  "
              f"val_loss={val_loss/len(val_dl):.4f}  "
              f"RMSE_mean={rmse_mean:.4f} (T:{rmse_T:.4f} G:{rmse_G:.4f} R:{rmse_R:.4f})  "
              f"WeightedR2={wr2:.5f}  "
              f"R2(Total:{per_r2['Dry_Total_g']:.3f} GDM:{per_r2['GDM_g']:.3f} "
              f"Green:{per_r2['Dry_Green_g']:.3f} Dead:{per_r2['Dry_Dead_g']:.3f} "
              f"Clover:{per_r2['Dry_Clover_g']:.3f})"
        )

        # ---- best の判定（CFG.SELECT_BEST_BY）----
        current_metric = wr2 if select_is_r2 else rmse_mean.item()
        improved = (current_metric > best_metric) if select_is_r2 else (current_metric < best_metric)
        if epoch == 1:
            improved = True

        if improved:
            best_path = os.path.join(CFG.OUT_DIR, f'best_model_fold{fold}.pth')
            save_checkpoint(eval_model, best_path)
            best_metric = current_metric
            best_preds = y_pred_all.numpy()  # OOF保存用（ベスト）
            print(f"  -> Best updated ({CFG.SELECT_BEST_BY.upper()}). Save {best_path}")

    
    # 逆変換してOOF保存
    # Save OOF from best epoch
    oof_preds = best_preds if best_preds is not None else y_pred_all.numpy()
    if CFG.USE_LOG1P:
        oof_preds = np.expm1(oof_preds)
    oof_df = val_df[['image_path'] + CFG.TARGETS].copy()
    for i,t in enumerate(CFG.TARGETS):
        oof_df[f'pred_{t}'] = oof_preds[:, i]
    oof_df.to_csv(os.path.join(CFG.OUT_DIR, f'oof_fold{fold}.csv'), index=False)
    return best_metric



# ============================================
# 学習用のピボット & 層化Fold作成ユーティリティ
# ============================================
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ALL_TARGET_COLS = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'GDM_g', 'Dry_Total_g']
INDEX_COLS = ['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm']

def _dup_check_for_pivot(df_long, index_cols=INDEX_COLS, name_col='target_name'):
    keys = index_cols + [name_col]
    dup_mask = df_long.duplicated(keys, keep=False)
    return df_long.loc[dup_mask, keys].value_counts().reset_index(name='count')

def long_to_wide_for_training(
    df_long: pd.DataFrame,
    targets=('Dry_Total_g','GDM_g','Dry_Green_g'),
    strict=True,
    aggfunc='first'
) -> pd.DataFrame:
    """
    長形式(train.csv) → 学習用の広形式（画像1行＋3ターゲット列）。
    余剰の2ターゲット(Dry_Dead/Clover)は保持してもOKだが、学習では未使用。
    """
    # 1) 重複チェック（必要なら平均等で潰す）
    if strict:
        dups = _dup_check_for_pivot(df_long)
        if len(dups):
            raise ValueError(
                f"Pivot keys have duplicates ({len(dups)} rows). "
                f"Set strict=False or aggfunc='mean'.\n{dups.head()}"
            )

    # 2) ピボット
    wide = df_long.pivot_table(
        index=INDEX_COLS,
        columns='target_name',
        values='target',
        aggfunc=aggfunc
    ).reset_index()

    # 3) 学習で使う3列が無ければエラー
    for t in targets:
        if t not in wide.columns:
            raise KeyError(f"Required target column missing after pivot: {t}")

    # 4) image_id 付与（任意）
    wide['image_id'] = wide['image_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

    # 5) 学習に最低限必要なカラムだけ抜き出し（メタは残してOK）
    keep_cols = list(INDEX_COLS) + list(targets) + ['image_id']
    keep_cols = [c for c in keep_cols if c in wide.columns]
    wide = wide[keep_cols].copy()
    return wide

def add_stratified_folds(
    df: pd.DataFrame,
    n_folds=5,
    label_col='Dry_Total_g',
    bins=5,
    seed=42
) -> pd.DataFrame:
    """
    Dry_Total_g をビニングして層化KFold。ターゲット分布を各foldで均等化しやすい。
    """
    df = df.copy()
    # 欠損・定数対応（qcutが失敗しないように）
    y = df[label_col].values
    # 一意値が少なければ bin 数を調整
    uniq = np.unique(y)
    bins = min(bins, max(2, len(uniq)))
    # 量子化。重複binに配慮して duplicates='drop'
    df['_strat'] = pd.qcut(y, q=bins, labels=False, duplicates='drop')

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df['fold'] = -1
    for f, (_, val_idx) in enumerate(skf.split(df, df['_strat'])):
        df.loc[val_idx, 'fold'] = f
    df = df.drop(columns=['_strat'])
    return df

def save_training_csv_for_existing_pipeline(df_wide: pd.DataFrame, out_path: str):
    """
    既存の学習スクリプトが読む `train.csv` 形式に合わせて保存。
    （= image_path と 3ターゲット列が含まれていればOK）
    """
    cols_needed = ['image_path', 'Dry_Total_g', 'GDM_g', 'Dry_Green_g']
    missing = [c for c in cols_needed if c not in df_wide.columns]
    if missing:
        raise KeyError(f"Columns missing for training: {missing}")
    df_wide.to_csv(out_path, index=False)
    print(f"Saved training CSV for pipeline: {out_path}  shape={df_wide.shape}")


# -------------------------
# 9) Main: k-fold
# -------------------------

def main(lr=None, batch_size=None, wd=None, warmup_epochs=None, dropout=None, aug_strength=None):
    """
    Main training/inference function with optional parameter overrides.
    
    Args:
        lr: Learning rate override (default: CFG.LR)
        batch_size: Batch size override (default: CFG.BATCH_SIZE)
        wd: Weight decay override (default: CFG.WD)
        warmup_epochs: Warmup epochs override (default: CFG.WARMUP_EPOCHS)
        dropout: Dropout rate override (default: 0.3)
        aug_strength: Augmentation strength override (default: 1.0)
    
    Returns:
        dict: Results containing 'cv_mean', 'cv_std', and 'fold_scores' (training mode)
              or DataFrame with predictions (inference mode)
    """
    # 推論モードの場合
    if CFG.INFERENCE_MODE:
        print("=" * 50)
        print("INFERENCE MODE")
        print("=" * 50)
        
        # モデル読み込みディレクトリを決定
        model_dir = CFG.INFERENCE_MODEL_DIR if CFG.INFERENCE_MODEL_DIR is not None else CFG.OUT_DIR
        print(f"Loading models from: {model_dir}")
        
        # 使用するfoldを決定
        if CFG.INFERENCE_FOLDS is None:
            # 全foldを自動検出
            model_paths = []
            for fold in range(CFG.N_FOLDS):
                model_path = os.path.join(model_dir, f'best_model_fold{fold}.pth')
                if os.path.exists(model_path):
                    model_paths.append(model_path)
                else:
                    print(f"Warning: Model not found: {model_path}")
            
            if len(model_paths) == 0:
                raise ValueError(f"No model files found in {model_dir}")
            
            print(f"Found {len(model_paths)} model(s): {[os.path.basename(p) for p in model_paths]}")
        else:
            # 指定されたfoldのみ使用
            model_paths = []
            for fold in CFG.INFERENCE_FOLDS:
                model_path = os.path.join(model_dir, f'best_model_fold{fold}.pth')
                if os.path.exists(model_path):
                    model_paths.append(model_path)
                else:
                    raise FileNotFoundError(f"Model not found: {model_path}")
            print(f"Using {len(model_paths)} model(s) from folds: {CFG.INFERENCE_FOLDS}")
        
        # 推論実行
        submission_df = predict_and_save_submission(
            test_csv_path=CFG.TEST_CSV,
            output_path=CFG.SUBMISSION_OUTPUT,
            model_paths=model_paths,
            image_dir=CFG.TEST_IMAGE_DIR,
            model_name=CFG.MODEL_NAME,
            target_names=CFG.TARGETS,
            dual_stream=CFG.DUAL_STREAM,
            dropout=dropout if dropout is not None else 0.3,
            batch_size=CFG.INFERENCE_BATCH_SIZE,
            device=CFG.DEVICE,
            num_workers=CFG.NUM_WORKERS,
            tta=CFG.USE_TTA,
            weights=None,
            use_log1p=CFG.USE_LOG1P
        )
        
        print("=" * 50)
        print(f"Inference completed. Submission saved to: {CFG.SUBMISSION_OUTPUT}")
        print("=" * 50)
        
        return submission_df
    
    # 学習モード（既存の処理）
    print("=" * 50)
    print("TRAINING MODE")
    print("=" * 50)
    df = pd.read_csv(CFG.TRAIN_CSV)

    # 2) 学習用に広形式へ（ターゲット列を作る）
    target_list = FIVE_TARGET_ORDER if CFG.TRAIN_FIVE_OUTPUT_LOSS else ['Dry_Total_g','GDM_g','Dry_Green_g']
    # CFG に反映（Dataset/OOF保存で使用する列順）
    CFG.TARGETS = target_list
    df = long_to_wide_for_training(
        df,
        targets=tuple(target_list),
        strict=True,          # 重複があればエラーにしたくないなら False に
        aggfunc='first'       # 重複を平均で潰すなら 'mean'
    )
    
    # 3) fold 付与（Dry_Total_g 層化。評価寄与が大きいので相性○）
    df = add_stratified_folds(
        df,
        n_folds=5,
        label_col='Dry_Total_g',
        bins=5,
        seed=42
    )
    # 期待カラム: image_path と CFG.TARGETS
    # 必要に応じてここで欠損処理や外れ値clipを実施
    assert set(CFG.TARGETS).issubset(df.columns), f"train.csvに{CFG.TARGETS}が必要です"


    # fold作成（GroupKFoldを使いたい場合は plot_id 等で置き換え）
    if 'fold' not in df.columns:
        df = kfold_split(df, n_folds=CFG.N_FOLDS, seed=CFG.SEED)

    df.to_csv(os.path.join(CFG.OUT_DIR, 'train_folds.csv'), index=False)
    print("Folds saved:", os.path.join(CFG.OUT_DIR, 'train_folds.csv'))

    bests = []
    for f in range(CFG.N_FOLDS):
        best_metric = train_one_fold(df, f, lr=lr, batch_size=batch_size, wd=wd, 
                                      warmup_epochs=warmup_epochs, dropout=dropout, aug_strength=aug_strength)
        bests.append(best_metric)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    metric_name = 'Weighted R2' if CFG.SELECT_BEST_BY.lower() == 'r2' else 'RMSE'
    cv_mean = np.mean(bests)
    cv_std = np.std(bests)
    print(f"\n=== CV {metric_name} (mean±std) ===")
    print(f"{cv_mean:.5f} ± {cv_std:.5f}")
    
    return {
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'fold_scores': bests
    }

# -------------------------
# 10) Inference Functions
# -------------------------

class TestDataset(Dataset):
    """Test dataset for inference (no labels)."""
    def __init__(self, df, image_dir, tf, dual_stream=True):
        self.df = df.reset_index(drop=True)
        self.paths = self.df['image_path'].values
        self.image_dir = image_dir
        self.tf = tf
        self.dual_stream = dual_stream

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        raw_path = self.paths[idx]
        # Use provided path if it exists; otherwise, fall back to joining with image_dir and basename
        candidate = raw_path if os.path.exists(raw_path) else os.path.join(self.image_dir, os.path.basename(raw_path))
        img = cv2.imread(candidate)
        if img is None:
            img = np.zeros((1000, 2000, 3), np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.dual_stream:
            h, w, _ = img.shape
            mid = w // 2
            left = img[:, :mid]
            right = img[:, mid:]
            t = self.tf(image=left, image_right=right)
            left = t['image']
            right = t['image_right']
        else:
            t = self.tf(image=img)
            left = t['image']
            right = left  # インターフェース維持：モデルが単一ストリーム時は右を無視

        return left, right, idx


def load_model(model_path, model_name=None, target_names=None, dual_stream=None, dropout=0.3, device=None):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint (.pth file)
        model_name: Model name (default: CFG.MODEL_NAME)
        target_names: Target names (default: CFG.TARGETS)
        dual_stream: Whether to use dual stream (default: CFG.DUAL_STREAM)
        dropout: Dropout rate (default: 0.3)
        device: Device to load model on (default: CFG.DEVICE)
    
    Returns:
        Loaded model in eval mode
    """
    if model_name is None:
        model_name = CFG.MODEL_NAME
    if target_names is None:
        target_names = CFG.TARGETS
    if dual_stream is None:
        dual_stream = CFG.DUAL_STREAM
    if device is None:
        device = CFG.DEVICE
    
    model = BiomassModel(
        model_name=model_name,
        pretrained=False,
        target_names=target_names,
        dual_stream=dual_stream,
        dropout=dropout
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    return model


def predict_single_image(model, image_path, transform, dual_stream=None, device=None, tta=False):
    """
    Predict on a single image with optional TTA.
    
    Args:
        model: Trained model
        image_path: Path to image file
        transform: Transform to apply
        dual_stream: Whether to use dual stream (default: CFG.DUAL_STREAM)
        device: Device to use (default: CFG.DEVICE)
        tta: Whether to use test-time augmentation (horizontal/vertical flips)
    
    Returns:
        numpy array of predictions [num_targets]
    """
    if dual_stream is None:
        dual_stream = CFG.DUAL_STREAM
    if device is None:
        device = CFG.DEVICE
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        img = np.zeros((1000, 2000, 3), np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    predictions = []
    
    if not tta:
        # Single prediction
        if dual_stream:
            h, w, _ = img.shape
            mid = w // 2
            left = img[:, :mid]
            right = img[:, mid:]
            t = transform(image=left, image_right=right)
            left = t['image']
            right = t['image_right']
        else:
            t = transform(image=img)
            left = t['image']
            right = left
        
        left = left.unsqueeze(0).to(device)
        right = right.unsqueeze(0).to(device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=CFG.USE_AMP):
                pred = model(left, right)  # [1, num_targets]
        
        predictions.append(pred.cpu().numpy()[0])
    else:
        # TTA: original, hflip, vflip, hvflip
        augs = [
            {'hflip': False, 'vflip': False},
            {'hflip': True, 'vflip': False},
            {'hflip': False, 'vflip': True},
            {'hflip': True, 'vflip': True},
        ]
        
        for aug in augs:
            img_aug = img.copy()
            if aug['hflip']:
                img_aug = cv2.flip(img_aug, 1)
            if aug['vflip']:
                img_aug = cv2.flip(img_aug, 0)
            
            if dual_stream:
                h, w, _ = img_aug.shape
                mid = w // 2
                left = img_aug[:, :mid]
                right = img_aug[:, mid:]
                t = transform(image=left, image_right=right)
                left = t['image']
                right = t['image_right']
            else:
                t = transform(image=img_aug)
                left = t['image']
                right = left
            
            left = left.unsqueeze(0).to(device)
            right = right.unsqueeze(0).to(device)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=CFG.USE_AMP):
                    pred = model(left, right)  # [1, num_targets]
            
            predictions.append(pred.cpu().numpy()[0])
        
        # Average predictions
        predictions = np.mean(predictions, axis=0)
    
    return predictions[0] if isinstance(predictions, list) else predictions


def predict_test_set(
    model,
    test_df,
    image_dir,
    transform,
    batch_size=32,
    dual_stream=None,
    device=None,
    num_workers=2,
    tta=False
):
    """
    Predict on entire test set.
    
    Args:
        model: Trained model
        test_df: DataFrame with 'image_path' column
        image_dir: Directory containing test images
        transform: Transform to apply
        batch_size: Batch size for inference
        dual_stream: Whether to use dual stream (default: CFG.DUAL_STREAM)
        device: Device to use (default: CFG.DEVICE)
        num_workers: Number of workers for DataLoader
        tta: Whether to use test-time augmentation
    
    Returns:
        numpy array of predictions [N, num_targets]
    """
    if dual_stream is None:
        dual_stream = CFG.DUAL_STREAM
    if device is None:
        device = CFG.DEVICE
    
    dataset = TestDataset(test_df, image_dir, transform, dual_stream=dual_stream)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    
    all_predictions = []
    model.eval()
    
    with torch.no_grad():
        for left, right, indices in tqdm(dataloader, desc="Predicting"):
            left = left.to(device, non_blocking=True)
            right = right.to(device, non_blocking=True)
            
            if tta:
                # TTA: original, hflip, vflip, hvflip
                preds_tta = []
                for hflip, vflip in [(False, False), (True, False), (False, True), (True, True)]:
                    left_aug = torch.flip(left, dims=[3] if hflip else []) if hflip else left
                    left_aug = torch.flip(left_aug, dims=[2] if vflip else []) if vflip else left_aug
                    right_aug = torch.flip(right, dims=[3] if hflip else []) if hflip else right
                    right_aug = torch.flip(right_aug, dims=[2] if vflip else []) if vflip else right_aug
                    
                    with torch.cuda.amp.autocast(enabled=CFG.USE_AMP):
                        pred = model(left_aug, right_aug)  # [B, num_targets]
                    preds_tta.append(pred)
                
                pred = torch.stack(preds_tta, dim=0).mean(dim=0)  # Average over TTA
            else:
                with torch.cuda.amp.autocast(enabled=CFG.USE_AMP):
                    pred = model(left, right)  # [B, num_targets]
            
            all_predictions.append(pred.cpu().numpy())
    
    return np.concatenate(all_predictions, axis=0)


def ensemble_predict(
    test_df,
    model_paths,
    image_dir,
    model_name=None,
    target_names=None,
    dual_stream=None,
    dropout=0.3,
    batch_size=32,
    device=None,
    num_workers=2,
    tta=False,
    weights=None
):
    """
    Ensemble predictions from multiple models (typically different folds).
    
    Args:
        test_df: DataFrame with 'image_path' column
        model_paths: List of paths to model checkpoints
        image_dir: Directory containing test images
        model_name: Model name (default: CFG.MODEL_NAME)
        target_names: Target names (default: CFG.TARGETS)
        dual_stream: Whether to use dual stream (default: CFG.DUAL_STREAM)
        dropout: Dropout rate (default: 0.3)
        batch_size: Batch size for inference
        device: Device to use (default: CFG.DEVICE)
        num_workers: Number of workers for DataLoader
        tta: Whether to use test-time augmentation
        weights: Weights for each model (default: equal weights)
    
    Returns:
        numpy array of ensemble predictions [N, num_targets]
    """
    if model_name is None:
        model_name = CFG.MODEL_NAME
    if target_names is None:
        target_names = CFG.TARGETS
    if dual_stream is None:
        dual_stream = CFG.DUAL_STREAM
    if device is None:
        device = CFG.DEVICE
    if weights is None:
        weights = [1.0] * len(model_paths)
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    transform = get_valid_tf(CFG.IMG_SIZE)
    
    all_predictions = []
    
    for i, model_path in enumerate(model_paths):
        print(f"Loading model {i+1}/{len(model_paths)}: {model_path}")
        model = load_model(
            model_path,
            model_name=model_name,
            target_names=target_names,
            dual_stream=dual_stream,
            dropout=dropout,
            device=device
        )
        
        pred = predict_test_set(
            model=model,
            test_df=test_df,
            image_dir=image_dir,
            transform=transform,
            batch_size=batch_size,
            dual_stream=dual_stream,
            device=device,
            num_workers=num_workers,
            tta=tta
        )
        
        all_predictions.append(pred * weights[i])
        
        # Clean up
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    ensemble_pred = np.sum(all_predictions, axis=0)
    return ensemble_pred


def predict_and_save_submission(
    test_csv_path,
    output_path,
    model_paths,
    image_dir=None,
    model_name=None,
    target_names=None,
    dual_stream=None,
    dropout=0.3,
    batch_size=None,
    device=None,
    num_workers=None,
    tta=None,
    weights=None,
    use_log1p=None
):
    """
    Predict on test set and save submission file in the required format.
    
    Args:
        test_csv_path: Path to test.csv (long format)
        output_path: Path to save submission CSV
        model_paths: List of paths to model checkpoints (or single path)
        image_dir: Directory containing test images (default: inferred from BASE_PATH)
        model_name: Model name (default: CFG.MODEL_NAME)
        target_names: Target names (default: CFG.TARGETS)
        dual_stream: Whether to use dual stream (default: CFG.DUAL_STREAM)
        dropout: Dropout rate (default: 0.3)
        batch_size: Batch size for inference
        device: Device to use (default: CFG.DEVICE)
        num_workers: Number of workers for DataLoader
        tta: Whether to use test-time augmentation
        weights: Weights for each model (default: equal weights)
        use_log1p: Whether model was trained with log1p (default: CFG.USE_LOG1P)
    
    Returns:
        DataFrame with predictions
    """
    if image_dir is None:
        image_dir = CFG.TEST_IMAGE_DIR
    if model_name is None:
        model_name = CFG.MODEL_NAME
    if target_names is None:
        target_names = CFG.TARGETS
    if dual_stream is None:
        dual_stream = CFG.DUAL_STREAM
    if device is None:
        device = CFG.DEVICE
    if use_log1p is None:
        use_log1p = CFG.USE_LOG1P
    if batch_size is None:
        batch_size = CFG.INFERENCE_BATCH_SIZE
    if num_workers is None:
        num_workers = CFG.NUM_WORKERS
    if tta is None:
        tta = CFG.USE_TTA
    
    # Load test CSV (long format)
    test_df_long = pd.read_csv(test_csv_path)
    
    # Get unique image paths
    unique_images = test_df_long['image_path'].unique()
    test_df = pd.DataFrame({'image_path': unique_images})
    
    # Convert model_paths to list if single path
    if isinstance(model_paths, str):
        model_paths = [model_paths]
    
    # Get predictions
    print(f"Predicting on {len(test_df)} images using {len(model_paths)} model(s)...")
    predictions = ensemble_predict(
        test_df=test_df,
        model_paths=model_paths,
        image_dir=image_dir,
        model_name=model_name,
        target_names=target_names,
        dual_stream=dual_stream,
        dropout=dropout,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        tta=tta,
        weights=weights
    )
    
    # Convert log1p predictions back to original scale
    if use_log1p:
        predictions = np.expm1(predictions)
    
    # Create prediction DataFrame
    pred_df = pd.DataFrame(
        predictions,
        columns=target_names,
        index=test_df.index
    )
    pred_df['image_path'] = test_df['image_path'].values
    
    # Convert to long format for submission
    submission_rows = []
    for _, row in test_df_long.iterrows():
        image_path = row['image_path']
        target_name = row['target_name']
        
        # Find corresponding prediction
        pred_row = pred_df[pred_df['image_path'] == image_path].iloc[0]
        
        # Get the 3 main targets
        total = pred_row['Dry_Total_g']
        gdm = pred_row['GDM_g']
        green = pred_row['Dry_Green_g']
        
        # Derive all 5 targets
        if target_name == 'Dry_Total_g':
            value = total
        elif target_name == 'GDM_g':
            value = gdm
        elif target_name == 'Dry_Green_g':
            value = green
        elif target_name == 'Dry_Dead_g':
            value = total - gdm
        elif target_name == 'Dry_Clover_g':
            value = gdm - green
        else:
            value = 0.0
        
        submission_rows.append({
            'sample_id': row['sample_id'],
            'target': max(0.0, value)  # Ensure non-negative
        })
    
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")
    
    return submission_df
main()