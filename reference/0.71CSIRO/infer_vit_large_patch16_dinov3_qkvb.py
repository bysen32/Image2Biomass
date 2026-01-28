from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GroupKFold, StratifiedGroupKFold
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

import os
from pathlib import Path
import timm
import warnings 

warnings.filterwarnings('ignore')
tqdm.pandas()

class RegressionDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        image = item.image
        targets = [item['Dry_Green_g'], item['Dry_Clover_g'], item['Dry_Dead_g']]
        width, height = image.size
        mid_point = width // 2
        left_image = image.crop((0, 0, mid_point, height))
        right_image = image.crop((mid_point, 0, width, height))

        if self.transform is not None:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)

        return left_image, right_image, targets

def get_test_dataloaders(data, image_size, batch_size):
    res = []
    for trans in [None, T.RandomHorizontalFlip(p=1.0), T.RandomVerticalFlip(p=1.0)]:
        transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if trans:
            transform = T.Compose([
                T.Resize(image_size),
                trans,
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        dataset = RegressionDataset(data, transform=transform)
        res.append(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4))
    return res


class FiLM(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2), 
            nn.ReLU(inplace=True), 
            nn.Linear(feat_dim // 2, feat_dim * 2)
        )

    def forward(self, context):
        gamma_beta = self.mlp(context)
        return torch.chunk(gamma_beta, 2, dim=1)

class CSIROModelRegressor(nn.Module):
    def __init__(self, model_name, pretrained=True, num_classes=3, dropout=0.0, freeze_backbone=False):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')

        self.film = FiLM(self.backbone.num_features)

        self.dropout = nn.Dropout(dropout)

        def make_head():
            return nn.Sequential(
                nn.Linear(self.backbone.num_features * 2, 8),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(8, 1),
            )

        self.head_green = make_head()
        self.head_clover = make_head()
        self.head_dead = make_head()

        self.softplus = nn.Softplus(beta=1.0)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False


    def forward(self, left_img, right_img):
        left_feat = self.backbone(left_img)
        right_feat = self.backbone(right_img)

        context = (left_feat + right_feat) / 2
        gamma, beta = self.film(context)

        left_feat_modulated = left_feat * (1 + gamma) + beta
        right_feat_modulated = right_feat * (1 + gamma) + beta

        combined = torch.cat([left_feat_modulated, right_feat_modulated], dim=1)

        green = self.softplus(self.head_green(combined))   
        clover = self.softplus(self.head_clover(combined))
        dead = self.softplus(self.head_dead(combined)) 

        logits = torch.cat([green, clover, dead], dim=1)

        return logits

def predict(model, dataloader, device):
    model.to(device)
    model.eval()

    all_outputs = []
    with torch.no_grad():
        for left_images, right_images, targets in dataloader:
            left_images = left_images.to(device)
            right_images = right_images.to(device)

            outputs = model(left_images, right_images)
            all_outputs.append(outputs.detach().cpu())

    outputs = torch.cat(all_outputs).numpy()
    return outputs

def predict_loaders(model, dataloaders, device):
    all_outputs = []
    for dataloader in dataloaders:
        outputs = predict(model, dataloader, device)
        all_outputs.append(outputs)
    avg_outputs = np.mean(all_outputs, axis=0)
    return avg_outputs

def predict_folds(dataloaders,models_dir):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    all_preds = []
    for model_file in Path(models_dir).glob('*.pth'):
        model = CSIROModelRegressor(CFG.MODEL_NAME, pretrained=False, num_classes=3)
        model.load_state_dict(torch.load(model_file))
        preds = predict_loaders(model, dataloaders, device)
        all_preds.append(preds)

    avg_preds = np.mean(all_preds, axis=0)
    return avg_preds


class CFG:
    DATA_PATH="/kaggle/input/csiro-biomass/"
    TEST_DATA_PATH="/kaggle/input/csiro-biomass/test.csv"
    MODEL_NAME="vit_large_patch16_dinov3_qkvb"
    MODELS_DIR ='/kaggle/input/modelv3/pytorch/default/1/models_retrained'
    IMG_SIZE=(512,512)


test_df = pd.read_csv(CFG.TEST_DATA_PATH)

test_df['target'] = 0.0
test_df[['sample_id_prefix', 'sample_id_suffix']] = test_df.sample_id.str.split('__', expand=True)

test_data_df = test_df.groupby(['sample_id_prefix', 'image_path']).apply(lambda df: df.set_index('target_name').target)
test_data_df.reset_index(inplace=True)
test_data_df.columns.name = None

test_data_df['image'] = test_data_df.image_path.progress_apply(lambda path: Image.open(CFG.DATA_PATH + path).convert('RGB'))

test_loaders = get_test_dataloaders(test_data_df, CFG.IMG_SIZE, 32)
preds = predict_folds(test_loaders,models_dir=CFG.MODELS_DIR)



test_data_df[['Dry_Green_g', 'Dry_Clover_g', 'Dry_Dead_g']] = preds
test_data_df['Dry_Green_g'] = test_data_df['Dry_Green_g'].apply(lambda x: 0.0 if x < 0.09 else x)
test_data_df['Dry_Clover_g'] = test_data_df['Dry_Clover_g'].apply(lambda x: 0.0 if x < 0.09 else x)
test_data_df['Dry_Dead_g'] = test_data_df['Dry_Dead_g'].apply(lambda x: 0.0 if x < 0.09 else x)
test_data_df['GDM_g'] = test_data_df.Dry_Green_g + test_data_df.Dry_Clover_g
test_data_df['Dry_Total_g'] = test_data_df.GDM_g + test_data_df.Dry_Dead_g

cols = [ 'Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g' , 'Dry_Total_g']
sub_df = test_data_df.set_index('sample_id_prefix')[cols].stack()
sub_df = sub_df.reset_index()
sub_df.columns = ['sample_id_prefix', 'target_name', 'target']

sub_df['sample_id'] = sub_df.sample_id_prefix + '__' + sub_df.target_name

cols = ['sample_id', 'target']
sub_df[cols].to_csv('submission_dinov2026.csv', index=False)

print(sub_df[cols])