import random
import numpy as np
import torch
import warnings
import timm
import torch.nn as nn
import pytorch_lightning as pl


from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import pandas as pd
from sklearn.model_selection import KFold
from pytorch_lightning.callbacks import ModelCheckpoint

warnings.filterwarnings('ignore')


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
        image = Image.open(f'/datasets/{row["image_path"]}').convert('RGB')
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
    def __init__(self, model_name='efficientnet_b0', pretrained=True, lr=1e-4, output_dim=5):
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=output_dim)
        self.criterion = nn.SmoothL1Loss()
        self.val_outputs = []

    def forward(self, x):
        return self.model(x) # shape: (N, 5)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_preds = self(x)
        loss = self.criterion(y_preds, y)
        self.log('train_loss', loss, on_setp=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_preds = self(x)
        loss = self.criterion(y_preds, y)
        self.val_outputs.append((y_preds.detach().cpu(), y.detach().cpu()))
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss
    
    def on_validation_epoch_end(self):
        if len(self.val_outputs) == 0:
            self.log("val_weighted_r2", 0.0, prog_bar=True, on_epoch=True)
            for name in ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]:
                self.log(f"val_r2_{name}", 0.0, on_epoch=True)
            self.val_outputs.clear()
            return
        
        y_preds, y_true = zip(*self.val_outputs)
        y_preds = torch.cat(y_preds).numpy()
        y_true = torch.cat(y_true).numpy()
        weighted_r2, r2_scores = weighted_r2_score(y_true, y_preds)
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



# train_df = pd.read_csv('/kaggle/input/csiro-biomass/train.csv')
train_df = pd.read_csv('/datasets/train.csv')
train_df = pd.pivot_table(train_df, index='image_path', columns=['target_name'], values='target').reset_index()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idxs, val_idxs) in enumerate(kf.split(train_df)):
    seed_everything(42)
    datamodule = ImageRegressionDataModule(train_df.iloc[train_idxs], train_df.iloc[val_idxs])
    model = MultiRegressionModel(model_name='efficientnet_b0', pretrained=True, lr=1e-4)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_weighted_r2',
        mode='max',
        save_top_k=1,
        filename=f'best_model_fold{fold}',
    )

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[checkpoint_callback],
        precision='16-mixed',
    )

    trainer.fit(model, datamodule=datamodule)
    torch.save(model.state_dict(), f'best_model_fold{fold}.pth')

