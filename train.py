import random
import numpy as np
import torch
import warnings
import timm
import torch.nn as nn
import os
import pytorch_lightning as pl


from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import pandas as pd
from sklearn.model_selection import KFold
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoModel, AutoProcessor

warnings.filterwarnings('ignore')


class CFG:
    IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
    
    SEED = 42

    if IS_KAGGLE:
        DATA_ROOT = "/kaggle/input/csiro-biomass"

        # Kaggle 离线模式下，模型通常存放在 input 目录下
        MODEL_WEIGHTS_PATH = "/kaggle/input/dinov2/pytorch/giant/1"
        DEVICES = 2

        OUTPUT_DIR = "/kaggle/working"

    else:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        os.environ['TIMM_MODEL_HUB'] = 'https://hf-mirror.com'

        DATA_ROOT = "./datasets/csiro-biomass"

        MODEL_WEIGHTS_PATH = "facebook/dinov2-giant"
        DEVICES = 1

        OUTPUT_DIR = "./outputs"

        # if not os.path.exists(MODEL_WEIGHTS_PATH):
        #     os.makedirs(MODEL_WEIGHTS_PATH, exist_ok=True)
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int=42):
    '''
    Seed everything for reproducibility
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def weighted_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray]:
    '''
    Calculate the weighted R2 score
    y_true, y_pred: shape (n_samples, 5)
    '''
    weights = np.array([0.1, 0.1, 0.1, 0.2, 0.5])
    r2_scores = []
    for i in range(5):
        y_true_i = y_true[:, i]
        y_pred_i = y_pred[:, i]
        ss_res = np.sum((y_true_i - y_pred_i) **2)
        ss_tot = np.sum((y_true_i - np.mean(y_true_i)) **2)
        r2_score = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2_scores.append(r2_score)
    r2_scores = np.array(r2_scores)
    weighted_r2_score = np.sum(r2_scores * weights) / np.sum(weights)
    return weighted_r2_score, r2_scores



# ========== Dataset ===========
class BiomassDataset(Dataset):
    def __init__(self, df, transforms=None, mode='train'):
        self.df = df
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        image = Image.open(f'{CFG.DATA_ROOT}/{row["image_path"]}').convert('RGB')
        if self.transforms:
            image = self.transforms(image)
        targets = torch.tensor([
            row["Dry_Green_g"],
            row["Dry_Dead_g"],
            row["Dry_Clover_g"],
            row["GDM_g"],
            row["Dry_Total_g"]
        ], dtype=torch.float32)
        return image, targets


class MultiRegressionModel(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModel.from_pretrained(CFG.MODEL_WEIGHTS_PATH)
        self.head = nn.Linear(self.model.config.hidden_size, 5) # 5 is the number of targets
        self.criterion = nn.SmoothL1Loss()
        self.val_outputs = []

    def forward(self, x):
        x = self.model(x).pooler_output # shape: (N, hidden_size)
        x = self.head(x) # shape: (N, 5)
        return x # shape: (N, 5)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.val_outputs.append((y_pred.detach().cpu(), y.detach().cpu()))
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss
    
    def on_validation_epoch_end(self):
        if len(self.val_outputs) == 0:
            self.log("val_weighted_r2", 0.0, prog_bar=True, on_epoch=True)
            for name in ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]:
                self.log(f"val_r2_{name}", 0.0, on_epoch=True)
            self.val_outputs.clear()
            return
        
        y_pred, y_true = zip(*self.val_outputs)
        y_pred = torch.cat(y_pred).numpy()
        y_true = torch.cat(y_true).numpy()
        weighted_r2, r2_scores = weighted_r2_score(y_true, y_pred)
        self.log("val_weighted_r2", weighted_r2, prog_bar=True, on_epoch=True)
        for i, name in enumerate(["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]):
            self.log(f"val_r2_{name}", r2_scores[i], on_epoch=True)
        self.val_outputs.clear()
        return
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# =========== DataModule ===========
class ImageRegressionDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, batch_size=8, num_workers=4, img_size=1000):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
    
    def setup(self, stage=None):
        self.train_transforms = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.val_transforms = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.train_dataset = BiomassDataset(self.train_df, transforms=self.train_transforms, mode='train')
        self.val_dataset = BiomassDataset(self.val_df, transforms=self.val_transforms, mode='val')
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)



train_df = pd.read_csv(f'{CFG.DATA_ROOT}/train.csv')
train_df = pd.pivot_table(train_df, index='image_path', columns=['target_name'], values='target').reset_index()

kf = KFold(n_splits=5, shuffle=True, random_state=CFG.SEED)

for fold, (train_idxs, val_idxs) in enumerate(kf.split(train_df)):
    seed_everything(CFG.SEED)
    datamodule = ImageRegressionDataModule(train_df.iloc[train_idxs], train_df.iloc[val_idxs], batch_size=1)
    model = MultiRegressionModel(lr=1e-4)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_weighted_r2',
        mode='max',
        save_top_k=1,
        filename=f'best_model_fold{fold}',
    )

    trainer = pl.Trainer(
        max_epochs=10,
        devices=CFG.DEVICES,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[checkpoint_callback],
        precision='16-mixed',
    )

    trainer.fit(model, datamodule=datamodule)
    torch.save(model.state_dict(), f'best_model_fold{fold}.pth')

