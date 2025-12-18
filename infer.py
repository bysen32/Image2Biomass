import pandas as pd
import os
from PIL import Image
import torch
import timm
import random
import numpy as np
import pytorch_lightning as pl
import warnings
import torch.nn as nn
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

warnings.filterwarnings('ignore')

def seed_everything(seed=42):
    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

ROOT = '/kaggle/input/csiro-biomass/'

# ========== Dataset ==========
class InferenceDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(ROOT, row['image_path'])
        image = Image.open(img_path).convert('RGB')
        if self.transforms:
            image = self.transforms(image)
        return image

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


# ========== Model ==========
class MultiRegressionModel(pl.LightningModule):
    def __init__(self, model_name='efficientnet_b0', pretrained=True, lr=1e-4, output_dim=5):
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=output_dim)
        self.criterion = nn.SmoothL1Loss()
        self.val_outputs = []

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_preds = self(x)
        loss = self.criterion(y_preds, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
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


def tta_inference(model, images):
    preds = model(images)
    preds_lr = model(torch.flip(images, dims=[3]))
    preds_ud = model(torch.flip(images, dims=[2]))
    preds_lrud = model(torch.flip(images, dims=[2, 3]))
    preds_mean = (preds + preds_lr + preds_ud + preds_lrud) / 4
    return preds_mean


def get_id(x):
    return x.split('_')[0]


# Transform
image_size = 1000
infer_transforms = T.Compose([
    T.Resize((image_size, image_size)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# DataLoader
test_df = pd.read_csv(os.path.join(ROOT, "test.csv"))
test_df = test_df[~test_df['image_path'].duplicated()][['sample_id', 'image_path']].reset_index(drop=True)
test_df['sample_id'] = test_df['sample_id'].apply(get_id)
test_dataset = InferenceDataset(test_df, transforms=infer_transforms)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results_dict = {}

for fold in range(5):
    model = MultiRegressionModel(model_name='efficientnet_b0', pretrained=False)
    model.load_state_dict(torch.load(f'/kaggle/input/efficientnet-b0/pytorch/default/1/best_model_fold{fold}.pth'))
    model.eval()
    model.to(device)
    results = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch
            images = images.to(device)
            preds = tta_inference(model, images)
            preds = preds.cpu().numpy()
            results.append(preds)
    results_dict[fold] = np.concatenate(results)
            

result_df = pd.DataFrame(np.mean([results_dict[fold] for fold in range(5)], axis=0),
                         columns=['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g'])
result_df['sample_id'] = test_df['sample_id']
result_df = pd.melt(result_df, id_vars='sample_id', value_vars=["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"], value_name='target')
result_df['sample_id'] = f"{result_df['sample_id']}__{result_df['variable']}"
result_df['target'] = result_df['target'].clip(0, 200)
result_df = result_df[['sample_id', 'target']]
print(result_df)
result_df.to_csv('submission.csv', index=False)