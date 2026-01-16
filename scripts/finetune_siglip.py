import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import glob
from pathlib import Path
import sys
from tqdm.auto import tqdm
import json
from copy import deepcopy

import pandas as pd
import numpy as np
import os
import math
import random

import torch
import torch.nn as nn
from torchvision import transforms

from PIL import Image
import cv2

from transformers import AutoProcessor, AutoImageProcessor, AutoModel, Siglip2Model, Siglip2ImageProcessor, SiglipModel, SiglipImageProcessor
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import KFold, GroupKFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.svm import LinearSVR
from sklearn.dummy import DummyRegressor

from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn import preprocessing

from dataclasses import dataclass
# from typing import Optional, Dict

# import matplotlib.pyplot as plt



def seeding(SEED):
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    # pl.seed_everything(SEED)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print('seeding done!!!')


def flush():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

@dataclass
class Config:
    # Data paths
    DATA_PATH: Path = Path(
        "/kaggle/input/csiro-biomass/") if IS_KAGGLE else Path("./datasets/csiro-biomass/")
    TRAIN_DATA_PATH: Path = DATA_PATH/'train'
    TEST_DATA_PATH: Path = DATA_PATH/'test'
    SAVE_DATA_PATH: Path = Path("/kaggle/outputs/working") if IS_KAGGLE else Path("save_data")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42

cfg = Config()
seeding(cfg.seed)


def pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    if 'target' in df.columns.tolist():
        df_pt = pd.pivot_table(
            df,
            values='target',
            index=['image_path', 'Sampling_Date', 'State',
                   'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm'],
            columns='target_name',
            aggfunc='mean'
        ).reset_index()
    else:
        df['target'] = 0
        df_pt = pd.pivot_table(
            df,
            values='target',
            index='image_path',
            columns='target_name',
            aggfunc='mean'
        ).reset_index()
    return df_pt


train_df = pd.read_csv(cfg.DATA_PATH/'train.csv')
test_df = pd.read_csv(cfg.DATA_PATH/'test.csv')
train_df = pivot_table(df=train_df)
test_df = pivot_table(df=test_df)

train_df['image_path'] = train_df['image_path'].apply(lambda p: str(cfg.DATA_PATH / p))
test_df['image_path'] = test_df['image_path'].apply(lambda p: str(cfg.DATA_PATH / p))
# train_df = pd.read_csv("/kaggle/input/csiro-datasplit/csiro_data_split.csv")


def melt_table(df: pd.DataFrame) -> pd.DataFrame:
    TARGET_NAMES = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    melted = df.melt(
        id_vars='image_path',
        value_vars=TARGET_NAMES,
        var_name='target_name',
        value_name='target'
    )

    melted['sample_id'] = (
        melted['image_path'].str.replace(r'^.*/', '', regex=True)  # remove folder path, keep filename
        .str.replace('.jpg', '', regex=False)  # remove extension
        + '__' + melted['target_name']
    )
  
    return melted[['sample_id', 'image_path', 'target_name', 'target']]

# t1 = melt_table(test_df)


def split_image(image, patch_size=520, overlap=16):
    h, w, c = image.shape
    stride = patch_size - overlap
    
    patches = []
    coords  = []   # (y1, x1, y2, x2)
    
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1 = y
            x1 = x
            y2 = y + patch_size
            x2 = x + patch_size
            
            # Pad last patch if needed (very rare with your fixed 1000×2000)
            patch = image[y1:y2, x1:x2, :]
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                pad_h = patch_size - patch.shape[0]
                pad_w = patch_size - patch.shape[1]
                patch = np.pad(patch, ((0,pad_h), (0,pad_w), (0,0)), mode='reflect')
            
            patches.append(patch)
            coords.append((y1, x1, y2, x2))
    
    return patches, coords


def get_model(model_path: str, device: str = 'cpu'):
    model = AutoModel.from_pretrained(model_path,
                                      local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(model_path)
    return model.eval().to(device), processor


def compute_embeddings(model_path, df, patch_size=520):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, processor = get_model(model_path=model_path, device=device)

    IMAGE_PATHS = []
    EMBEDDINGS = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row['image_path']
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        patches, coords = split_image(img, patch_size=patch_size)
        images = [Image.fromarray(p).convert("RGB") for p in patches]

        inputs = processor(images=images, return_tensors="pt").to(model.device)
        with torch.no_grad():
            if 'siglip' in model_path:
                features = model.get_image_features(**inputs)
            elif 'dino' in model_path:
                features = model(**inputs).pooler_output
                # patches = model(**inputs).last_hidden_state
                # features = patches[:, 0, :]
            else:
                raise Exception("Model should be dino or siglip")
        embeds = features.mean(dim=0).detach().cpu().numpy()
        EMBEDDINGS.append(embeds)
        IMAGE_PATHS.append(img_path)

    embeddings = np.stack(EMBEDDINGS, axis=0)
    n_features = embeddings.shape[1]
    emb_columns = [f"emb{i+1}" for i in range(n_features)]
    emb_df = pd.DataFrame(embeddings, columns=emb_columns)
    emb_df['image_path'] = IMAGE_PATHS
    df_final = df.merge(emb_df, on='image_path', how='left')
    flush()
    return df_final 

if IS_KAGGLE:   
    dino_path = "/kaggle/input/dinov2/pytorch/giant/1"
    siglip_path = "/kaggle/input/google-siglip-so400m-patch14-384/transformers/default/1"
else:
    dino_path = "pretrained_models/dinov2-giant"
    siglip_path = "pretrained_models/google-siglip-so400m-patch14-384"

if not os.path.exists(cfg.SAVE_DATA_PATH):
    os.makedirs(cfg.SAVE_DATA_PATH)

if os.path.exists(cfg.SAVE_DATA_PATH/'train_siglip_df.csv'):
    train_siglip_df = pd.read_csv(cfg.SAVE_DATA_PATH/'train_siglip_df.csv')
else:
    train_siglip_df = compute_embeddings(model_path=siglip_path, df=train_df, patch_size=520)
    train_siglip_df.to_csv(cfg.SAVE_DATA_PATH/'train_siglip_df.csv', index=False)

if os.path.exists(cfg.SAVE_DATA_PATH/'test_siglip_df.csv'):
    test_siglip_df = pd.read_csv(cfg.SAVE_DATA_PATH/'test_siglip_df.csv')
else:
    test_siglip_df = compute_embeddings(model_path=siglip_path, df=test_df, patch_size=520)
    test_siglip_df.to_csv(cfg.SAVE_DATA_PATH/'test_siglip_df.csv', index=False)

flush()

############################################################
# 1️⃣Text embeddings
############################################################

# "dense pasture grass",
#         "sparse pasture vegetation",
#         "patchy grass cover",
#         "bare soil patches in grass",
#         "thick tangled grass",
#         "open low-density pasture",
#         "dry cracked soil",
#         "dry canopy",
#         "low moisture vegetation",
#         "dry pasture with yellow tones",
#         "wilted grass"


# def generate_semantic_features(image_embeddings, model_path=siglip_path):
#     print(f"Loading SigLIP Text Encoder from {model_path}...")
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     try:
#         model = AutoModel.from_pretrained(model_path).to(device)
#         tokenizer = AutoTokenizer.from_pretrained(model_path)
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         return None

#     # Prepare Image Tensor
#     if isinstance(image_embeddings, np.ndarray):
#         img_tensor = torch.tensor(image_embeddings, dtype=torch.float32).to(device)
#     else:
#         img_tensor = image_embeddings.to(device)
#     img_tensor = img_tensor / img_tensor.norm(p=2, dim=-1, keepdim=True)

#     # Define Prompts (The Dictionary from above)
#     AGRONOMIC_PROMPTS = {
#         "biomass": {
#             "pos": ["dense tall pasture", "high biomass vegetation", "thick grassy volume"],
#             "neg": ["bare soil", "sparse vegetation", "very short clipped grass"]
#         },
#         "green_vs_brown": {
#             "pos": ["lush green vibrant pasture", "green leaves"],
#             "neg": ["dry brown dead grass", "yellow straw-like vegetation"]
#         },
#         "clover_presence": {
#             "pos": ["white clover patches", "broadleaf clover", "clover flowers"],
#             "neg": ["pure ryegrass", "blade-like grass leaves", "monoculture grass"]
#         },
#         "litter_dead": {
#             "pos": ["accumulated dead plant litter", "mat of dry dead grass"],
#             "neg": ["clean fresh growth", "upright green stalks"]
#         }
#     }

#     feature_store = []
    
#     with torch.no_grad():
#         for axis_name, prompts in AGRONOMIC_PROMPTS.items():
#             # Encode Positive Prompts
#             pos_inputs = tokenizer(prompts["pos"], padding="max_length", return_tensors="pt").to(device)
#             pos_emb = model.get_text_features(**pos_inputs)
#             pos_emb = pos_emb / pos_emb.norm(p=2, dim=-1, keepdim=True)
            
#             # Encode Negative Prompts
#             neg_inputs = tokenizer(prompts["neg"], padding="max_length", return_tensors="pt").to(device)
#             neg_emb = model.get_text_features(**neg_inputs)
#             neg_emb = neg_emb / neg_emb.norm(p=2, dim=-1, keepdim=True)
            
#             # Create Mean Embeddings for the concept groups
#             pos_concept = pos_emb.mean(dim=0, keepdim=True)
#             neg_concept = neg_emb.mean(dim=0, keepdim=True)
            
#             # Calculate Similarity
#             # Shape: (N_imgs, 1)
#             sim_pos = torch.matmul(img_tensor, pos_concept.T).cpu().numpy()
#             sim_neg = torch.matmul(img_tensor, neg_concept.T).cpu().numpy()
            
#             # --- Feature Engineering ---
#             # 1. The Axis Score (The "Ruler"): Pos - Neg
#             axis_score = sim_pos - sim_neg
            
#             # 2. The Raw Activation (Max fit):
#             max_act = np.maximum(sim_pos, sim_neg)
            
#             feature_store.append(axis_score)
#             # Optional: Add raw similarities if you suspect non-linearities
#             # feature_store.append(sim_pos) 
            
#     # Stack features: (N_samples, N_axes)
#     semantic_features = np.hstack(feature_store)
    
#     print(f"Generated {semantic_features.shape[1]} semantic axis features.")
#     return semantic_features

def generate_semantic_features(image_embeddings, model_path=siglip_path):
    """
    Generates 'Concept Scores' by averaging synonyms and calculating biological ratios.
    """
    print(f"Loading SigLIP Text Encoder from {model_path}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        model = AutoModel.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # 1. Define Concept Ensembles (Grouping synonyms reduces noise)
    concept_groups = {
        # Quantity Anchors
        "bare": ["bare soil", "dirt ground", "sparse vegetation", "exposed earth"],
        "sparse": ["low density pasture", "thin grass", "short clipped grass"],
        "medium": ["average pasture cover", "medium height grass", "grazed pasture"],
        "dense": ["dense tall pasture", "thick grassy volume", "high biomass", "overgrown vegetation"],
        
        # State Anchors
        "green": ["lush green vibrant pasture", "photosynthesizing leaves", "fresh growth"],
        "dead": ["dry brown dead grass", "yellow straw", "senesced material", "standing hay"],
        
        # Species Anchors
        "clover": ["white clover", "trifolium repens", "broadleaf legume", "clover flowers"],
        "grass": ["ryegrass", "blade-like leaves", "fescue", "grassy sward"],
        "weeds": ["broadleaf weeds", "thistles", "non-pasture vegetation"]
    }
    
    # 2. Encode and Average Prompts for each Concept
    concept_vectors = {}
    with torch.no_grad():
        for name, prompts in concept_groups.items():
            inputs = tokenizer(prompts, padding="max_length", return_tensors="pt").to(device)
            emb = model.get_text_features(**inputs)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            # Average the embeddings of synonyms to get a stable "Concept Vector"
            concept_vectors[name] = emb.mean(dim=0, keepdim=True)

    # 3. Compute Concept Scores
    if isinstance(image_embeddings, np.ndarray):
        img_tensor = torch.tensor(image_embeddings, dtype=torch.float32).to(device)
    else:
        img_tensor = image_embeddings.to(device)
    img_tensor = img_tensor / img_tensor.norm(p=2, dim=-1, keepdim=True)

    scores = {}
    for name, vec in concept_vectors.items():
        # Dot product
        scores[name] = torch.matmul(img_tensor, vec.T).cpu().numpy().flatten()
    
    # 4. Feature Engineering: Explicit Ratios
    # These help models distinguish between "High Biomass Dead" vs "High Biomass Green"
    
    # Convert dict to DataFrame for easy math
    df_scores = pd.DataFrame(scores)
    
    # A. Greenness Ratio: Green / (Green + Dead)
    df_scores['ratio_greenness'] = df_scores['green'] / (df_scores['green'] + df_scores['dead'] + 1e-6)
    
    # B. Legume Fraction: Clover / (Clover + Grass)
    df_scores['ratio_clover'] = df_scores['clover'] / (df_scores['clover'] + df_scores['grass'] + 1e-6)
    
    # C. Vegetation Cover: (Dense + Medium) / (Bare + Sparse)
    df_scores['ratio_cover'] = (df_scores['dense'] + df_scores['medium']) / (df_scores['bare'] + df_scores['sparse'] + 1e-6)
    
    # D. "Volume": Max of density anchors
    df_scores['max_density'] = df_scores[['bare', 'sparse', 'medium', 'dense']].max(axis=1)

    print(f"Generated {df_scores.shape[1]} semantic features (Ensembles + Ratios).")
    return df_scores.values



############################################################
# 2️⃣Feature Engineering
############################################################

# import numpy as np
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.cross_decomposition import PLSRegression
# from sklearn.mixture import GaussianMixture
# from sklearn.linear_model import BayesianRidge
# from sklearn.metrics.pairwise import cosine_similarity

# class SupervisedEmbeddingEngine(BaseEstimator, TransformerMixin):
#     def __init__(self, 
#                  n_pca=0.95, 
#                  n_pls=10, # Increased slightly to capture specific biomass targets
#                  n_gmm=3,  # Reduced GMM. 5 is risky for N=357 (overfitting clusters)
#                  random_state=42):
        
#         self.n_pca = n_pca
#         self.n_pls = n_pls
#         self.n_gmm = n_gmm
#         self.random_state = random_state
        
#         self.scaler = StandardScaler()
#         self.pca = PCA(n_components=n_pca, random_state=random_state)
#         self.pls = PLSRegression(n_components=n_pls, scale=False)
#         # Using full makes it more robust to singularity
#         self.gmm = GaussianMixture(n_components=n_gmm, covariance_type='full', random_state=random_state)
        
#         self.pls_fitted_ = False

#     def fit(self, X, y=None, X_semantic=None):
#         # 1. Concatenate Embeddings + Semantic Features EARLY
#         # This allows PLS to find correlations between Text Scores and Biomass
#         if X_semantic is not None:
#             # Weight semantic features up slightly so PCA respects them
#             X_combined = np.hstack([X, X_semantic * 2.0]) 
#         else:
#             X_combined = X
            
#         X_scaled = self.scaler.fit_transform(X_combined)
        
#         # 2. Fit Unsupervised
#         self.pca.fit(X_scaled)
#         self.gmm.fit(X_scaled)
        
#         # 3. Fit PLS (Supervised)
#         if y is not None:
#             # Handle multi-output targets (the 5 biomass columns)
#             # Ensure y is (N, 5)
#             y_clean = y.values if hasattr(y, 'values') else y
#             self.pls.fit(X_scaled, y_clean)
#             self.pls_fitted_ = True
            
#         return self

#     def transform(self, X, X_semantic=None):
#         if X_semantic is not None:
#             X_combined = np.hstack([X, X_semantic * 2.0])
#         else:
#             X_combined = X
            
#         X_scaled = self.scaler.transform(X_combined)
#         return self._generate_features(X_scaled)

#     def _generate_features(self, X_scaled):
#         features = []
        
#         # PCA (Structure of data)
#         f_pca = self.pca.transform(X_scaled)
#         features.append(f_pca)
        
#         # PLS (Structure of Targets) - THIS IS CRITICAL
#         if self.pls_fitted_:
#             f_pls = self.pls.transform(X_scaled)
#             features.append(f_pls)
        
#         # GMM (Cluster Probabilities)
#         f_gmm = self.gmm.predict_proba(X_scaled)
#         features.append(f_gmm)
        
#         return np.hstack(features)

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import BayesianRidge
from sklearn.metrics.pairwise import cosine_similarity


class SupervisedEmbeddingEngine(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 n_pca=0.98,  # Slightly higher to keep texture details
                 n_pls=8,     # Keep 8, it worked well
                 n_gmm=5,     
                 random_state=42):
        
        self.n_pca = n_pca
        self.n_pls = n_pls
        self.n_gmm = n_gmm
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_pca, random_state=random_state)
        self.pls = PLSRegression(n_components=n_pls, scale=False)
        self.gmm = GaussianMixture(n_components=n_gmm, covariance_type='diag', random_state=random_state)

        self.pls_fitted_ = False

    def fit(self, X, y=None, X_semantic=None):
        # 1. Standard Scaling on IMAGE embeddings only
        X_scaled = self.scaler.fit_transform(X)
        
        # 2. Fit Unsupervised on IMAGES
        self.pca.fit(X_scaled)
        self.gmm.fit(X_scaled)
        
        # 3. Fit PLS on IMAGES (Supervised)
        if y is not None:
            y_clean = y.values if hasattr(y, 'values') else y
            self.pls.fit(X_scaled, y_clean)
            self.pls_fitted_ = True
        
        return self

    def transform(self, X, X_semantic=None):
        X_scaled = self.scaler.transform(X)
        return self._generate_features(X_scaled, X_semantic)

    def _generate_features(self, X_scaled, X_semantic=None):
        features = []
        
        # A. PCA (Texture/Structure from Images)
        f_pca = self.pca.transform(X_scaled)
        features.append(f_pca)
        
        # B. PLS (Biomass-correlated signals from Images)
        if self.pls_fitted_:
            f_pls = self.pls.transform(X_scaled)
            features.append(f_pls)
        
        # C. GMM (Cluster probs)
        f_gmm = self.gmm.predict_proba(X_scaled)
        features.append(f_gmm)
        
        # D. Semantic Features (LATE FUSION)
        # We append them raw. They are already high-level signals.
        if X_semantic is not None:
            # Normalize semantic scores relative to themselves to match scale of PCA/PLS
            sem_norm = (X_semantic - np.mean(X_semantic, axis=0)) / (np.std(X_semantic, axis=0) + 1e-6)
            features.append(sem_norm)

        return np.hstack(features)



COLUMNS = train_df.filter(like="emb").columns
TARGET_NAMES = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
weights = {
    'Dry_Green_g': 0.1,
    'Dry_Dead_g': 0.1,
    'Dry_Clover_g': 0.1,
    'GDM_g': 0.2,
    'Dry_Total_g': 0.5,
}

TARGET_MAX = {
    "Dry_Clover_g": 71.7865,
    "Dry_Dead_g": 83.8407,
    "Dry_Green_g": 157.9836,
    "Dry_Total_g": 185.70,
    "GDM_g": 157.9836,
}

def competition_metric(y_true, y_pred) -> float:
    y_weighted = 0
    for l, label in enumerate(TARGET_NAMES):
        y_weighted = y_weighted + y_true[:, l].mean() * weights[label]

    ss_res = 0
    ss_tot = 0
    for l, label in enumerate(TARGET_NAMES):
        ss_res = ss_res + ((y_true[:, l] - y_pred[:, l])**2).mean() * weights[label]
        ss_tot = ss_tot + ((y_true[:, l] - y_weighted)**2).mean() * weights[label]

    return 1 - ss_res / ss_tot


def post_process_biomass(df_preds):
    """
    Enforces physical mass balance constraints on biomass predictions.
    
    Constraints enforced:
    1. Dry_Green_g + Dry_Clover_g = GDM_g
    2. GDM_g + Dry_Dead_g = Dry_Total_g
    
    Method:
    Uses Orthogonal Projection. It finds the set of values that satisfy
    the constraints while minimizing the Euclidean distance to the 
    original model predictions.
    
    Args:
        df_preds (pd.DataFrame): DataFrame containing the 5 prediction columns.
        
    Returns:
        pd.DataFrame: A new DataFrame with consistent, non-negative values.
    """
    # 1. Define the specific order required for the math
    # We treat the vector x as: [Green, Clover, Dead, GDM, Total]
    ordered_cols = [
        "Dry_Green_g", 
        "Dry_Clover_g", 
        "Dry_Dead_g", 
        "GDM_g", 
        "Dry_Total_g"
    ]
    
    # Check if columns exist
    if not all(col in df_preds.columns for col in ordered_cols):
        missing = [c for c in ordered_cols if c not in df_preds.columns]
        raise ValueError(f"Input DataFrame is missing columns: {missing}")

    # 2. Extract values in the specific order -> Shape (N_samples, 5)
    Y = df_preds[ordered_cols].values.T  # Transpose to (5, N) for matrix math

    # 3. Define the Constraint Matrix C
    # We want Cx = 0
    # Eq 1: 1*Green + 1*Clover + 0*Dead - 1*GDM + 0*Total = 0
    # Eq 2: 0*Green + 0*Clover + 1*Dead + 1*GDM - 1*Total = 0
    C = np.array([
        [1, 1, 0, -1,  0],
        [0, 0, 1,  1, -1]
    ])

    # 4. Calculate Projection Matrix P
    # P = I - C^T * (C * C^T)^-1 * C
    # This projects any vector onto the null space of C (the valid subspace)
    C_T = C.T
    inv_CCt = np.linalg.inv(C @ C_T)
    P = np.eye(5) - C_T @ inv_CCt @ C

    # 5. Apply Projection
    # Y_new = P * Y
    Y_reconciled = P @ Y

    # 6. Transpose back to (N, 5)
    Y_reconciled = Y_reconciled.T

    # 7. Post-correction for negatives
    # Projection can mathematically create negative values (e.g. if Total was predicted 0)
    # We clip to 0. Note: This might slightly break the sum equality again, 
    # but exact equality with negatives is physically impossible anyway.
    Y_reconciled = Y_reconciled.clip(min=0)

    # 8. Create Output DataFrame
    df_out = df_preds.copy()
    df_out[ordered_cols] = Y_reconciled

    return df_out

# def post_process_biomass(df_preds):
#     """
#     Enforces mass balance constraints hierarchically.
    
#     Philosophy: 
#     - GDM_g is trusted. Green/Clover are scaled to match it.
#     - Dry_Total_g is trusted. Dead is derived as (Total - GDM).
#     - If constraints are physically impossible (e.g. GDM > Total),
#       we assume Total was underestimated and raise it to match GDM.
      
#     Args:
#         df_preds (pd.DataFrame): Predictions
        
#     Returns:
#         pd.DataFrame: Consistently processed dataframe.
#     """
#     # Create a copy to avoid SettingWithCopy warnings
#     df_out = df_preds.copy()
    
#     # ---------------------------------------------------------
#     # 1. Enforce: Dry_Green_g + Dry_Clover_g = GDM_g
#     # ---------------------------------------------------------
#     # We trust the *magnitude* of GDM_g more than the components.
#     # We trust the *ratio* of Green vs Clover from the model.
    
#     # Calculate current component sum
#     comp_sum = df_out["Dry_Green_g"] + df_out["Dry_Clover_g"]
    
#     # Avoid division by zero
#     mask_nonzero = comp_sum > 1e-9
    
#     # Calculate scaling factor so components sum exactly to GDM
#     scale_factor = df_out.loc[mask_nonzero, "GDM_g"] / comp_sum[mask_nonzero]
    
#     # Apply scaling
#     df_out.loc[mask_nonzero, "Dry_Green_g"] *= scale_factor
#     df_out.loc[mask_nonzero, "Dry_Clover_g"] *= scale_factor
    
#     # Edge case: If comp_sum is 0 but GDM is not, we can't scale.
#     # (Optional: You could split GDM evenly, but usually the model predicts 0 GDM here too)
    
#     # ---------------------------------------------------------
#     # 2. Enforce: GDM_g + Dry_Dead_g = Dry_Total_g
#     # ---------------------------------------------------------
#     # You stated Dead is hard to predict. 
#     # Therefore, we discard the direct prediction of Dead and derive it.
    
#     df_out["Dry_Dead_g"] = df_out["Dry_Total_g"] - df_out["GDM_g"]
    
#     # ---------------------------------------------------------
#     # 3. Handle Physical Impossibilities (Negative Dead)
#     # ---------------------------------------------------------
#     # If Dead < 0, it means GDM > Total. This is physically impossible.
#     # Logic: GDM is a sum of living parts (robust). Total is the scan of everything.
#     # If GDM > Total, the model likely underestimated Total.
    
#     neg_dead_mask = df_out["Dry_Dead_g"] < 0
    
#     if neg_dead_mask.any():
#         # Set Dead to 0 (cannot be negative)
#         df_out.loc[neg_dead_mask, "Dry_Dead_g"] = 0
        
#         # Raise Total to match GDM (maintaining balance)
#         df_out.loc[neg_dead_mask, "Dry_Total_g"] = df_out.loc[neg_dead_mask, "GDM_g"]

#     return df_out

def compare_results(oof, train_data):
    y_oof_df = pd.DataFrame(oof, columns=TARGET_NAMES) # ensure columns match
    # 2. Check Score BEFORE Processing
    raw_score = competition_metric(train_data[TARGET_NAMES].values, y_oof_df.values)
    print(f"Raw CV Score: {raw_score:.6f}")
    
    # 3. Apply Post-Processing
    y_oof_proc = post_process_biomass(y_oof_df)
    
    # 4. Check Score AFTER Processing
    proc_score = competition_metric(train_data[TARGET_NAMES].values, y_oof_proc.values)
    print(f"Processed CV Score: {proc_score:.6f}")
    
    print(f"Improvement: {raw_score - proc_score:.6f}")
    
    
# train_df['fold'].nunique()
    

# from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler
# from sklearn.svm import SVR

# def cross_validate(model, train_data, test_data, feature_engine, target_transform='max', seed=42):
#     """
#     target_transform options:
#     - 'max': Linear scaling by max value (Preserves distribution shape)
#     - 'log': np.log1p (Aggressive compression of outliers)
#     - 'sqrt': np.sqrt (Moderate compression, good for biological counts/area)
#     - 'yeo-johnson': PowerTransformer (Makes data Gaussian-like automatically)
#     - 'quantile': QuantileTransformer (Forces strict Normal distribution)
#     """

#     n_splits = train_data['fold'].nunique()
#     target_max_arr = np.array([TARGET_MAX[t] for t in TARGET_NAMES], dtype=float)
#     y_true = train_data[TARGET_NAMES]
    
#     y_pred = pd.DataFrame(0.0, index=train_data.index, columns=TARGET_NAMES)
#     y_pred_test = np.zeros([len(test_data), len(TARGET_NAMES)], dtype=float)

#     for fold in range(n_splits):
#         seeding(seed*(seed//2 + fold))
#         # 1. Split Data
#         train_mask = train_data['fold'] != fold
#         valid_mask = train_data['fold'] == fold
#         val_idx = train_data[valid_mask].index

#         X_train_raw = train_data[train_mask][COLUMNS].values
#         X_valid_raw = train_data[valid_mask][COLUMNS].values
#         X_test_raw = test_data[COLUMNS].values
        
#         y_train = train_data[train_mask][TARGET_NAMES].values
#         y_valid = train_data[valid_mask][TARGET_NAMES].values

#         # ===========================
#         # 2) TARGET TRANSFORMATION
#         # ===========================
#         transformer = None # To store stateful transformers (Yeo/Quantile)
        
#         if target_transform == 'log':
#             y_train_proc = np.log1p(y_train)
            
#         elif target_transform == 'max':
#             y_train_proc = y_train / target_max_arr
            
#         elif target_transform == 'sqrt':
#             # Great for biomass/area data (Variance stabilizing)
#             y_train_proc = np.sqrt(y_train)
            
#         elif target_transform == 'yeo-johnson':
#             # Learns optimal parameter to make data Gaussian
#             transformer = PowerTransformer(method='yeo-johnson', standardize=True)
#             y_train_proc = transformer.fit_transform(y_train)
            
#         elif target_transform == 'quantile':
#             # Forces data into a normal distribution (Robust to outliers)
#             # transformer = QuantileTransformer(output_distribution='uniform', n_quantiles=64, random_state=42)
#             transformer = RobustScaler()
#             y_train_proc = transformer.fit_transform(y_train)
            
#         else:
#             y_train_proc = y_train

#         # ==========================================
#         # 3) FEATURE ENGINEERING
#         # ==========================================
#         engine = deepcopy(feature_engine)
#         # Note: If your engine uses PLS, pass the transformed y!
#         engine.fit(X_train_raw, y=y_train_proc) 
        
#         x_train_eng = engine.transform(X_train_raw)
#         x_valid_eng = engine.transform(X_valid_raw)
#         x_test_eng = engine.transform(X_test_raw)
        
#         # ==========================================
#         # 4) TRAIN & PREDICT
#         # ==========================================
#         fold_valid_pred = np.zeros_like(y_valid)
#         fold_test_pred = np.zeros([len(test_data), len(TARGET_NAMES)])

#         for k in range(len(TARGET_NAMES)):
#             regr = deepcopy(model)
#             regr.fit(x_train_eng, y_train_proc[:, k])
            
#             # Raw Predictions (in transformed space)
#             pred_valid_raw = regr.predict(x_valid_eng)
#             pred_test_raw = regr.predict(x_test_eng)
            
#             # Store raw for inverse transform block below
#             fold_valid_pred[:, k] = pred_valid_raw
#             fold_test_pred[:, k] = pred_test_raw

#         # ===========================
#         # 5) INVERSE TRANSFORM (Apply to full matrix)
#         # ===========================
#         if target_transform == 'log':
#             fold_valid_pred = np.expm1(fold_valid_pred)
#             fold_test_pred = np.expm1(fold_test_pred)
            
#         elif target_transform == 'max':
#             fold_valid_pred = fold_valid_pred * target_max_arr
#             fold_test_pred = fold_test_pred * target_max_arr
            
#         elif target_transform == 'sqrt':
#             # Inverse of sqrt is square
#             fold_valid_pred = np.square(fold_valid_pred)
#             fold_test_pred = np.square(fold_test_pred)
            
#         elif target_transform in ['yeo-johnson', 'quantile']:
#             # Use the fitted transformer to invert
#             fold_valid_pred = transformer.inverse_transform(fold_valid_pred)
#             fold_test_pred = transformer.inverse_transform(fold_test_pred)

#         # # Final Clip (Biomass cannot be negative)
#         # fold_valid_pred = fold_valid_pred.clip(min=0)
#         # fold_test_pred = fold_test_pred.clip(min=0)

#         # Store results
#         y_pred.loc[val_idx] = fold_valid_pred
#         y_pred_test += fold_test_pred / n_splits
        
#         if fold == 0:
#             print(f"  [Fold 0] Target: {target_transform}, Feats: {x_train_eng.shape}")

#     full_cv = competition_metric(y_true.values, y_pred.values)
#     print(f"Full CV Score: {full_cv:.6f}")
    
#     return y_pred.values, y_pred_test

# # Initialize
# seed = 42
# feat_engine = SupervisedEmbeddingEngine(
#     n_pca=0.80,
#     n_pls=10,             # Extract 8 strong supervised signals
#     n_gmm=3,             # 6 Soft clusters
#     random_state=seed
# )

# # print("######## Ridge Regression #######")
# # oof_ridge, pred_test_ri = cross_validate(Ridge(), train_siglip_df, test_siglip_df, feature_engine=feat_engine)
# # compare_results(oof_ridge, train_siglip_df)

# # print("####### Lasso Regression #######")
# # oof_la, pred_test_la = cross_validate(Lasso(), train_siglip_df, test_siglip_df, feature_engine=feat_engine)
# # compare_results(oof_la, train_siglip_df)

# print("\n###### GradientBoosting Regressor #######")
# oof_gb, pred_test_gb = cross_validate(
#     GradientBoostingRegressor(random_state=seed), 
#     train_siglip_df, test_siglip_df, 
#     feature_engine=feat_engine, 
#     target_transform='max')
# compare_results(oof_gb, train_siglip_df)

# print("\n###### Hist Gradient Boosting Regressor ######")
# oof_hb, pred_test_hb = cross_validate(
#     HistGradientBoostingRegressor(random_state=seed), 
#     train_siglip_df, test_siglip_df, 
#     feature_engine=feat_engine, 
#     target_transform='max')
# compare_results(oof_hb, train_siglip_df)

# print("\n##### CAT Regressor ######")
# oof_cat, pred_test_cat = cross_validate(
#     CatBoostRegressor(verbose=0, random_seed=seed), 
#     train_siglip_df, test_siglip_df, 
#     feature_engine=feat_engine
# )
# compare_results(oof_cat, train_siglip_df)

# print("\n######## XGB #######")
# oof_xgb, pred_test_xgb = cross_validate(
#     XGBRegressor(verbosity=0), 
#     train_siglip_df, test_siglip_df, 
#     feature_engine=feat_engine, 
#     target_transform='max')
# compare_results(oof_xgb, train_siglip_df)

# print("\n######## LGBM #######")
# oof_lgbm, pred_test_lgbm = cross_validate(
#     LGBMRegressor(verbose=-1, random_state=seed), 
#     train_siglip_df, test_siglip_df, 
#     feature_engine=feat_engine, 
#     target_transform='max')
# compare_results(oof_lgbm, train_siglip_df)


############################################################
# 3️⃣Semantic probing
############################################################

# --- STEP 1: Generate Semantic Features ---
# We combine Train and Test to generate features in one go, then split them back.
# This ensures the text-projections are consistent.

# Concatenate embeddings
X_all_emb = np.vstack([
    train_siglip_df[COLUMNS].values, 
    test_siglip_df[COLUMNS].values
])

# Generate Semantic Probes (Using the function defined in Part 1)
# Make sure SIGLIP_PATH is correct for your environment
print("Generating Semantic Features via SigLIP Text Encoder...")
try:
    all_semantic_scores = generate_semantic_features(X_all_emb, model_path=siglip_path)
    
    # Split back into Train and Test
    n_train = len(train_siglip_df)
    sem_train_full = all_semantic_scores[:n_train]
    sem_test_full = all_semantic_scores[n_train:]
    print(f"Semantic Features Generated. Train: {sem_train_full.shape}, Test: {sem_test_full.shape}")
    
except Exception as e:
    print(f"Skipping Semantic Features due to error: {e}")
    # Fallback to None if model path is wrong or memory fails
    sem_train_full = None
    sem_test_full = None


# --- STEP 2: Updated Cross-Validation Function ---
def cross_validate(model, train_data, test_data, feature_engine, 
                   semantic_train=None, semantic_test=None, # <--- NEW ARGS
                   target_transform='max', seed=42):

    n_splits = train_data['fold'].nunique()
    # Setup Targets
    target_max_arr = np.array([TARGET_MAX[t] for t in TARGET_NAMES], dtype=float)
    y_true = train_data[TARGET_NAMES]
    
    # Setup Storage
    y_pred = pd.DataFrame(0.0, index=train_data.index, columns=TARGET_NAMES)
    y_pred_test = np.zeros([len(test_data), len(TARGET_NAMES)], dtype=float)

    for fold in range(n_splits):
        seeding(seed*(seed//2 + fold))
        # Create masks
        train_mask = train_data['fold'] != fold
        valid_mask = train_data['fold'] == fold
        val_idx = train_data[valid_mask].index

        # Raw Inputs (Embeddings)
        X_train_raw = train_data[train_mask][COLUMNS].values
        X_valid_raw = train_data[valid_mask][COLUMNS].values
        X_test_raw = test_data[COLUMNS].values
        
        # Semantic Inputs (Slicing)
        # We handle the case where semantic features might be None
        sem_train_fold = semantic_train[train_mask] if semantic_train is not None else None
        sem_valid_fold = semantic_train[valid_mask] if semantic_train is not None else None
        
        # Raw Targets
        y_train = train_data[train_mask][TARGET_NAMES].values
        y_valid = train_data[valid_mask][TARGET_NAMES].values

        # ===========================
        # 1) TRANSFORM TARGETS
        # ===========================
        if target_transform == 'log':
            y_train_proc = np.log1p(y_train)
        elif target_transform == 'max':
            y_train_proc = y_train / target_max_arr
        else:
            y_train_proc = y_train

        # ==========================================
        # 2) FEATURE ENGINEERING
        # ==========================================
        engine = deepcopy(feature_engine)
        
        # FIT: Now passes y (for PLS/RFE) and Semantic Features
        engine.fit(X_train_raw, y=y_train_proc, X_semantic=sem_train_fold)
        
        # TRANSFORM: Pass Semantic Features
        x_train_eng = engine.transform(X_train_raw, X_semantic=sem_train_fold)
        x_valid_eng = engine.transform(X_valid_raw, X_semantic=sem_valid_fold)
        # For test, we use the full test semantic set
        x_test_eng = engine.transform(X_test_raw, X_semantic=semantic_test)
        
        # ==========================================
        # 3) TRAIN & PREDICT
        # ==========================================
        fold_valid_pred = np.zeros_like(y_valid)
        fold_test_pred = np.zeros([len(test_data), len(TARGET_NAMES)])

        for k in range(len(TARGET_NAMES)):
            regr = deepcopy(model)
            
            # Fit model
            regr.fit(x_train_eng, y_train_proc[:, k])
            
            # Predict
            pred_valid_raw = regr.predict(x_valid_eng)
            pred_test_raw = regr.predict(x_test_eng)
            
            # ===========================
            # 4) INVERSE TRANSFORM
            # ===========================
            if target_transform == 'log':
                pred_valid_inv = np.expm1(pred_valid_raw)
                pred_test_inv = np.expm1(pred_test_raw)
            elif target_transform == 'max':
                pred_valid_inv = (pred_valid_raw * target_max_arr[k])
                pred_test_inv = (pred_test_raw * target_max_arr[k])
            else:
                pred_valid_inv = pred_valid_raw
                pred_test_inv = pred_test_raw

            fold_valid_pred[:, k] = pred_valid_inv
            fold_test_pred[:, k] = pred_test_inv

        # Store results
        y_pred.loc[val_idx] = fold_valid_pred
        y_pred_test += fold_test_pred / n_splits
        
        if fold == 0:
            print(f"  [Fold 0 Info] Target: {target_transform}, Feats: {x_train_eng.shape}")

    full_cv = competition_metric(y_true.values, y_pred.values)
    print(f"Full CV Score: {full_cv:.6f}")
    
    return y_pred.values, y_pred_test

# --- STEP 3: Run Models ---

# Initialize the NEW Supervised Engine
feat_engine = SupervisedEmbeddingEngine(
    n_pca=0.80,
    n_pls=8,             # Supervised signals
    n_gmm=6,             # Soft clusters
)

# print("######## Ridge Regression #######")
# oof_ridge, pred_test_ri = cross_validate(
#     Ridge(), 
#     train_siglip_df, test_siglip_df, 
#     feature_engine=feat_engine,
#     semantic_train=sem_train_full, # <--- Pass Semantic
#     semantic_test=sem_test_full    # <--- Pass Semantic
# )
# compare_results(oof_ridge, train_siglip_df)

# print("\n####### Lasso Regression #######")
# # Lasso should perform much better now due to PLS and RFE
# oof_la, pred_test_la = cross_validate(
#     Lasso(alpha=0.015), # Small alpha for normalized feats
#     train_siglip_df, test_siglip_df, 
#     feature_engine=feat_engine,
#     semantic_train=sem_train_full,
#     semantic_test=sem_test_full
# )
# compare_results(oof_la, train_siglip_df)

print("\n###### GradientBoosting Regressor #######")
oof_gb, pred_test_gb = cross_validate(
    GradientBoostingRegressor(), 
    train_siglip_df, test_siglip_df, 
    feature_engine=feat_engine,
    semantic_train=sem_train_full,
    semantic_test=sem_test_full
)
compare_results(oof_gb, train_siglip_df)

print("\n###### Hist Gradient Boosting Regressor ######")
oof_hb, pred_test_hb = cross_validate(
    HistGradientBoostingRegressor(), 
    train_siglip_df, test_siglip_df, 
    feature_engine=feat_engine,
    semantic_train=sem_train_full,
    semantic_test=sem_test_full
)
compare_results(oof_hb, train_siglip_df)

print("\n##### CAT Regressor ######")
oof_cat, pred_test_cat = cross_validate(
    CatBoostRegressor(verbose=0), 
    train_siglip_df, test_siglip_df, 
    feature_engine=feat_engine,
    semantic_train=sem_train_full,
    semantic_test=sem_test_full
)
compare_results(oof_cat, train_siglip_df)

# print("\n######## XGB #######")
# oof_xgb, pred_test_xgb = cross_validate(
#     XGBRegressor(verbosity=0), 
#     train_siglip_df, test_siglip_df, 
#     feature_engine=feat_engine, 
#     semantic_train=sem_train_full,
#     semantic_test=sem_test_full,
#     target_transform='max')
# compare_results(oof_xgb, train_siglip_df)

print("\n######## LGBM #######")
oof_lgbm, pred_test_lgbm = cross_validate(
    LGBMRegressor(verbose=-1), 
    train_siglip_df, test_siglip_df, 
    feature_engine=feat_engine, 
    semantic_train=sem_train_full,
    semantic_test=sem_test_full,
    target_transform='max')
compare_results(oof_lgbm, train_siglip_df)


############################################################
# 4️⃣Hyperparameter tuning
############################################################

# from sklearn.model_selection import RandomizedSearchCV, KFold
# from scipy.stats import uniform, randint

# # ==========================================
# # 1. PARAMETER GRIDS
# # ==========================================

# # HistGradientBoostingRegressor Hyperparameters
# hist_gb_params = {
#     'learning_rate': uniform(0.01, 0.2),      # Continuous distribution
#     'max_iter': [100, 300, 500, 1000],        # Trees
#     'max_leaf_nodes': randint(15, 63),        # Complexity
#     'min_samples_leaf': randint(10, 50),      # Regularization
#     'l2_regularization': uniform(0, 5),       # L2 Reg
#     'max_depth': [None, 5, 10, 15]            # Depth constraint
# }

# # GradientBoostingRegressor Hyperparameters
# # (Standard GBR is slower, so we use slightly smaller ranges)
# gb_params = {
#     'learning_rate': uniform(0.01, 0.2),
#     'n_estimators': [100, 300, 500],
#     'subsample': uniform(0.6, 0.4),           # 0.6 to 1.0
#     'max_depth': randint(3, 8),
#     'min_samples_split': randint(2, 20),
#     'min_samples_leaf': randint(1, 10)
# }

# def tune_on_fold_zero(model_class, param_dist, train_data, feature_engine, 
#                       semantic_train=None, target_transform='max', 
#                       n_iter=20, seed=42):
    
#     print(f"--- Tuning {model_class.__name__} on Fold 0 ---")
    
#     # 1. Extract Fold 0 (Mimicking cross_validate logic)
#     fold = 0
#     train_mask = train_data['fold'] != fold
    
#     # Raw Inputs
#     X_train_raw = train_data[train_mask][COLUMNS].values
    
#     # Semantic Inputs
#     sem_train_fold = semantic_train[train_mask] if semantic_train is not None else None
    
#     # Targets
#     y_train = train_data[train_mask][TARGET_NAMES].values
    
#     # 2. Transform Targets
#     target_max_arr = np.array([TARGET_MAX[t] for t in TARGET_NAMES], dtype=float)
#     if target_transform == 'log':
#         y_train_proc = np.log1p(y_train)
#     elif target_transform == 'max':
#         y_train_proc = y_train / target_max_arr
#     else:
#         y_train_proc = y_train

#     # 3. Feature Engineering (Fit on Fold 0 Train)
#     print("Fitting Feature Engine on Fold 0...")
#     engine = deepcopy(feature_engine)
#     engine.fit(X_train_raw, y=y_train_proc, X_semantic=sem_train_fold)
#     x_train_eng = engine.transform(X_train_raw, X_semantic=sem_train_fold)
    
#     print(f"Features ready: {x_train_eng.shape}. Starting SearchCV per target...")

#     # 4. Tune per Target
#     best_params_per_target = {}
    
#     for k, target_name in enumerate(TARGET_NAMES):
#         print(f"  > Tuning Target: {target_name} ({k+1}/{len(TARGET_NAMES)})")
        
#         # Initialize Base Model
#         base_model = model_class(random_state=seed)
        
#         # Setup Randomized Search
#         # cv=3 is sufficient for tuning to prevent overfitting
#         search = RandomizedSearchCV(
#             estimator=base_model,
#             param_distributions=param_dist,
#             n_iter=n_iter,
#             scoring='neg_mean_squared_error',
#             cv=3, 
#             n_jobs=-1,
#             random_state=seed,
#             verbose=0
#         )
        
#         # Fit on the processed features
#         search.fit(x_train_eng, y_train_proc[:, k])
        
#         best_params_per_target[target_name] = search.best_params_
#         print(f"    Best Score (MSE): {-search.best_score_:.5f}")
#         print(f"    Params: {search.best_params_}")

#     return best_params_per_target

# # Initialize the NEW Supervised Engine
# feat_engine = SupervisedEmbeddingEngine(
#     n_pca=0.80,
#     n_pls=8,             # Supervised signals
#     n_gmm=6,             # Soft clusters
# )

# # --- TUNE HIST GRADIENT BOOSTING ---
# hist_best_params = tune_on_fold_zero(
#     model_class=HistGradientBoostingRegressor,
#     param_dist=hist_gb_params,
#     train_data=train_siglip_df,
#     feature_engine=feat_engine,
#     semantic_train=sem_train_full,
#     target_transform='max',
#     n_iter=15  # Adjust based on your time constraints
# )

# print("\n\n====== RECOMENDED HIST GB PARAMS (Averaged or First Target) ======")
# # Often it's better to pick one set of stable params for all targets 
# # unless variance is huge. Here we look at the first target's result as a proxy:
# print(hist_best_params[TARGET_NAMES[0]])


# # --- TUNE STANDARD GRADIENT BOOSTING ---
# gb_best_params = tune_on_fold_zero(
#     model_class=GradientBoostingRegressor,
#     param_dist=gb_params,
#     train_data=train_siglip_df,
#     feature_engine=feat_engine,
#     semantic_train=sem_train_full,
#     target_transform='max',
#     n_iter=10
# )

# print("\n\n====== RECOMENDED GB PARAMS ======")
# print(gb_best_params[TARGET_NAMES[0]])



# import optuna
# import numpy as np
# import pandas as pd
# from sklearn.base import clone
# from sklearn.decomposition import PCA
# from sklearn.multioutput import MultiOutputRegressor
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor

# # ---------------------------------------------------------
# # 1. Corrected Helper Function
# # ---------------------------------------------------------
# feat_engine = EmbeddingFeatureEngine(
#     n_pca_components=0.90, 
#     n_clusters=25, 
#     use_stats=True, 
#     use_similarity=True,
#     use_anomaly=True,        # Adds Anomaly Score
#     use_entropy=True,        # Adds Entropy
#     use_pca_interactions=True # Adds Poly features on Top 5 PCA
# )

# def get_cv_score(model, train_data, feature_engine, target_transform='max', random_state=42):
#     """
#     Runs CV on ALL folds dynamically to return a single score.
#     Optimized for speed (vectorized target processing).
    
#     Args:
#         model: Estimator (must support Multi-Output or be wrapped in MultiOutputRegressor)
#         train_data: DataFrame containing 'fold' column
#         feature_engine: Transformer with .fit() and .transform()
#         target_transform: 'log', 'max', or None
#     """
#     # 1. Setup global constants
#     target_max_arr = np.array([TARGET_MAX[t] for t in TARGET_NAMES], dtype=float)
#     y_true = train_data[TARGET_NAMES].values
#     y_pred = np.zeros([len(train_data), len(TARGET_NAMES)], dtype=float)
    
#     # 2. Detect Folds dynamically
#     folds = sorted(train_data['fold'].unique())
    
#     # 3. Loop over folds
#     for fold in folds:
#         # -------------------------
#         # Data Slicing
#         # -------------------------
#         train_mask = train_data['fold'] != fold
#         valid_mask = train_data['fold'] == fold
#         val_idx = train_data[valid_mask].index

#         X_train_raw = train_data.loc[train_mask, COLUMNS].values
#         X_valid_raw = train_data.loc[valid_mask, COLUMNS].values
        
#         y_train = train_data.loc[train_mask, TARGET_NAMES].values
#         # y_valid used implicitely via y_true at the end

#         # -------------------------
#         # A) Transform Targets
#         # -------------------------
#         if target_transform == 'log':
#             y_train_proc = np.log1p(y_train)
#         elif target_transform == 'max':
#             y_train_proc = y_train / target_max_arr
#         else:
#             y_train_proc = y_train

#         # -------------------------
#         # B) Feature Engineering
#         # -------------------------
#         # Fit engine only on training split
#         engine = deepcopy(feature_engine)
#         engine.fit(X_train_raw)
        
#         X_train_eng = engine.transform(X_train_raw)
#         X_valid_eng = engine.transform(X_valid_raw)

#         # -------------------------
#         # C) Fit Model (Multi-Output)
#         # -------------------------
#         regr = clone(model)
#         regr.fit(X_train_eng, y_train_proc)

#         # -------------------------
#         # D) Predict & Inverse Transform
#         # -------------------------
#         valid_pred_raw = np.array(regr.predict(X_valid_eng))
        
#         if target_transform == 'log':
#             valid_pred = np.expm1(valid_pred_raw)
#         elif target_transform == 'max':
#             valid_pred = valid_pred_raw * target_max_arr
#         else:
#             valid_pred = valid_pred_raw

#         # Clip and Store
#         y_pred[val_idx] = valid_pred.clip(0)

#     # 4. Calculate Metric
#     score = competition_metric(y_true, y_pred)
    
#     # # Clean output buffer if running in a loop
#     # try:
#     #     from IPython.display import flush_ipython
#     #     flush_ipython()
#     # except ImportError:
#     #     pass
        
#     return score

# # ---------------------------------------------------------
# # 2. Corrected CatBoost Objective
# # ---------------------------------------------------------
# def objective_catboost(trial):
#     params = {
#         # Search Space
#         'iterations': trial.suggest_int('iterations', 800, 2000), # Increased min iterations
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
#         'depth': trial.suggest_int('depth', 4, 8), # Reduced max depth to save memory
#         'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
#         'random_strength': trial.suggest_float('random_strength', 1e-3, 5.0, log=True),
#         'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        
#         # Fixed GPU Params
#         'loss_function': 'MultiRMSE',
#         'task_type': 'GPU',
#         'boosting_type': 'Plain', 
#         'devices': '0',
#         'verbose': 0,
#         'random_state': 42,
#         'allow_writing_files': False # Prevents creating log files
#     }
    
#     model = CatBoostRegressor(**params)
    
#     # Removed n_splits argument; it now uses whatever is in 'train'
#     return get_cv_score(model, train_siglip_df, feature_engine=feat_engine)

# # ---------------------------------------------------------
# # 3. Corrected XGBoost Objective
# # ---------------------------------------------------------
# def objective_xgboost(trial):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 800, 2000),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
#         'max_depth': trial.suggest_int('max_depth', 3, 8),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        
#         # Fixed
#         'tree_method': 'hist',
#         'device': 'cuda',
#         'n_jobs': -1,
#         'random_state': 42,
#         'verbosity': 0
#     }
    
#     model = MultiOutputRegressor(XGBRegressor(**params))
#     return get_cv_score(model, train_siglip_df, feature_engine=feat_engine)

# # ---------------------------------------------------------
# # 4. Corrected LightGBM Objective
# # ---------------------------------------------------------
# def objective_lgbm(trial):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 800, 2000),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
#         'num_leaves': trial.suggest_int('num_leaves', 20, 100),
#         'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        
#         # Fixed
#         'device': 'gpu',
#         'n_jobs': -1,
#         'random_state': 42,
#         'verbose': -1
#     }
    
#     model = MultiOutputRegressor(LGBMRegressor(**params))
#     return get_cv_score(model, train_siglip_df, feature_engine=feat_engine)


# # # --- 1. Tune CatBoost (Highest Priority) ---
# # print("Tuning CatBoost...")
# # study_cat = optuna.create_study(direction='maximize')
# # study_cat.optimize(objective_catboost, n_trials=20)

# # print("Best CatBoost Params:", study_cat.best_params)
# # best_cat_params = study_cat.best_params
# # # Re-add fixed params that Optuna didn't tune
# # best_cat_params.update({
# #     'loss_function': 'MultiRMSE', 
# #     # 'task_type': 'GPU', 
# #     'boosting_type': 'Plain', 
# #     # 'devices': '0', 
# #     'verbose': 0, 
# #     'random_state': 42
# # })


# # # --- 2. Tune XGBoost ---
# # print("\nTuning XGBoost...")
# # study_xgb = optuna.create_study(direction='maximize')
# # study_xgb.optimize(objective_xgboost, n_trials=20)

# # print("Best XGBoost Params:", study_xgb.best_params)
# # # best_xgb_params = study_xgb.best_params
# # # best_xgb_params.update({
# # #     'tree_method': 'hist', 
# # #     'device': 'cuda', 
# # #     'n_jobs': -1, 
# # #     'random_state': 42
# # # })


# # --- 3. Tune LightGBM ---
# print("\nTuning LightGBM...")
# study_lgbm = optuna.create_study(direction='maximize')
# study_lgbm.optimize(objective_lgbm, n_trials=20)

# print("Best LightGBM Params:", study_lgbm.best_params)
# # best_lgbm_params = study_lgbm.best_params
# # best_lgbm_params.update({
# #     'device': 'gpu', 
# #     'n_jobs': -1, 
# #     'random_state': 42, 
# #     'verbose': -1
# # })

############################################################
# 5️⃣Multioutput models
############################################################

# import numpy as np
# import pandas as pd
# from copy import deepcopy
# from sklearn.base import clone
# from sklearn.decomposition import PCA
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.multioutput import RegressorChain

# # Import the specific libraries
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor

# # ---------------------------------------------------------
# # 1. The Optimized Multi-Output GPU CV Function
# # ---------------------------------------------------------
# def cross_validate_multioutput_gpu(model, train_data, test_data, feature_engine, 
#                                    semantic_train=None, semantic_test=None, # <--- NEW ARGS
#                                    target_transform='max', seed=42, n_splits=5):
#     """
#     Performs Cross Validation using a Multi-Output strategy (Vectorized)
#     with support for Semantic Features and Supervised Embedding Engines.
#     """
    
#     # 1. Setup Target Max Array
#     target_max_arr = np.array([TARGET_MAX[t] for t in TARGET_NAMES], dtype=float)
#     y_true = train_data[TARGET_NAMES].values
    
#     # 2. Pre-allocate arrays
#     y_pred = np.zeros([len(train_data), len(TARGET_NAMES)], dtype=float)
#     y_pred_test = np.zeros([len(test_data), len(TARGET_NAMES)], dtype=float)

#     print(f"Starting CV with model: {model.__class__.__name__}")
#     print(f"Target Transform Strategy: {target_transform}")

#     # Ensure n_splits matches the fold column if present
#     if 'fold' in train_data.columns:
#         n_splits = train_data['fold'].nunique()

#     for fold in range(n_splits):
#         seeding(seed*(seed//2 + fold))
#         # ------------------------------
#         # Data Preparation
#         # ------------------------------
#         # Create masks
#         train_mask = train_data['fold'] != fold
#         valid_mask = train_data['fold'] == fold
#         val_idx = train_data[valid_mask].index

#         # Raw Inputs (Embeddings)
#         X_train_raw = train_data.loc[train_mask, COLUMNS].values
#         X_valid_raw = train_data.loc[valid_mask, COLUMNS].values
#         X_test_raw = test_data[COLUMNS].values
        
#         # Semantic Inputs (Slicing) - NEW LOGIC
#         # We handle the case where semantic features might be None
#         sem_train_fold = semantic_train[train_mask] if semantic_train is not None else None
#         sem_valid_fold = semantic_train[valid_mask] if semantic_train is not None else None
        
#         # Raw Targets
#         y_train = train_data.loc[train_mask, TARGET_NAMES].values
#         y_valid = train_data.loc[valid_mask, TARGET_NAMES].values

#         # ------------------------------
#         # 1) Transform Targets (Vectorized)
#         # ------------------------------
#         if target_transform == 'log':
#             y_train_proc = np.log1p(y_train)
#         elif target_transform == 'max':
#             y_train_proc = y_train / target_max_arr
#         else:
#             y_train_proc = y_train
        
#         # ------------------------------
#         # 2) Feature Engineering
#         # ------------------------------
#         engine = deepcopy(feature_engine)
        
#         # FIT: Pass X_train, Y_train (for PLS), and Semantic Train
#         engine.fit(X_train_raw, y=y_train_proc, X_semantic=sem_train_fold)
        
#         # TRANSFORM: Pass corresponding Semantic slices
#         x_train_eng = engine.transform(X_train_raw, X_semantic=sem_train_fold)
#         x_valid_eng = engine.transform(X_valid_raw, X_semantic=sem_valid_fold)
#         # For test, we use the full test semantic set
#         x_test_eng = engine.transform(X_test_raw, X_semantic=semantic_test)

#         # ------------------------------
#         # 3) Train (Multi-Output)
#         # ------------------------------
#         regr = clone(model) 
        
#         # Fit on (N_samples, N_targets)
#         # XGBoost and CatBoost (MultiRMSE) support this natively
#         regr.fit(x_train_eng, y_train_proc)

#         # ------------------------------
#         # 4) Predict & Unscale
#         # ------------------------------
#         valid_pred_raw = np.array(regr.predict(x_valid_eng))
#         test_pred_raw = np.array(regr.predict(x_test_eng))

#         # Inverse Transform
#         if target_transform == 'log':
#             valid_pred = np.expm1(valid_pred_raw)
#             test_pred = np.expm1(test_pred_raw)
#         elif target_transform == 'max':
#             valid_pred = valid_pred_raw * target_max_arr
#             test_pred = test_pred_raw * target_max_arr
#         else:
#             valid_pred = valid_pred_raw
#             test_pred = test_pred_raw

#         # Clip negative predictions
#         valid_pred = valid_pred.clip(0)
#         test_pred = test_pred.clip(0)

#         # Store OOF
#         y_pred[val_idx] = valid_pred
        
#         # Accumulate Test Preds
#         y_pred_test += test_pred / n_splits
            
#         if fold == 0:
#              print(f"  [Fold 0 Debug] Transformed Train Shape: {x_train_eng.shape}")

#     # Global CV score
#     try:
#         full_cv = competition_metric(y_true, y_pred)
#         print(f"Full CV Score: {full_cv:.6f}")
#     except NameError:
#         print("Done (metric function not found)")

#     return y_pred, y_pred_test

# feat_engine = SupervisedEmbeddingEngine(
#     n_pca=0.80,
#     n_pls=8,             # Supervised signals
#     n_gmm=6,             # Soft clusters
# )

# # ---------------------------------------------------------
# # 2. Model Definitions
# # ---------------------------------------------------------
# best_cat_params = {
#     'iterations': 1783, 
#     'learning_rate': 0.0633221588945314, 
#     'depth': 4, 
#     'l2_leaf_reg': 0.1312214556803292, 
#     'random_strength': 0.04403178418151252, 
#     'bagging_temperature': 0.9555074383215754
# }
# best_cat_params.update({
#     'loss_function': 'MultiRMSE', 
#     # 'task_type': 'GPU', 
#     'boosting_type': 'Plain', 
#     # 'devices': '0', 
#     'verbose': 0, 
#     'random_state': 42
# })

# best_xgb_params = {
#     'n_estimators': 1501, 
#     'learning_rate': 0.024461148923117938, 
#     'max_depth': 3, 
#     'subsample': 0.6905614627569726, 
#     'colsample_bytree': 0.895428293256401, 
#     'reg_alpha': 0.4865138988842402, 
#     'reg_lambda': 0.6015849227570268
# }
# best_xgb_params.update({
#     'tree_method': 'hist', 
#     # 'device': 'cuda', 
#     'n_jobs': -1, 
#     'random_state': 42
# })

# best_lgbm_params = {
#     'n_estimators': 1232, 
#     'learning_rate': 0.045467475791811464, 
#     'num_leaves': 32, 
#     'min_child_samples': 38, 
#     'subsample': 0.9389508238313968, 
#     'colsample_bytree': 0.8358504077200445, 
#     'reg_alpha': 0.10126277169074206, 
#     'reg_lambda': 0.1357065010990351
# }
# best_lgbm_params.update({
#     # 'device': 'gpu', 
#     'n_jobs': -1, 
#     'random_state': 42, 
#     'verbose': -1
# })

# best_hgb_params = {
#     'l2_regularization': 0.4424625102595975, 
#     'learning_rate': 0.04919657248382905, 
#     'max_depth': None, 
#     'max_iter': 300, 
#     'max_leaf_nodes': 54, 
#     'min_samples_leaf': 30,
#     'random_state': 42
# }

# best_gbm_params = {
#     'learning_rate': 0.021616722433639893, 
#     'max_depth': 7, 
#     'min_samples_leaf': 4, 
#     'min_samples_split': 9, 
#     'n_estimators': 500, 
#     'subsample': 0.608233797718321,
#     'random_state': 42
# }

# # --- A. XGBoost (Wrapped) ---
# # XGBoost requires MultiOutputRegressor wrapper for multi-target
# xgb_model = MultiOutputRegressor(
#     XGBRegressor(
#         **best_xgb_params
#     )
# )

# # --- B. LightGBM (Wrapped) ---
# # LightGBM requires MultiOutputRegressor wrapper.
# # Note: Ensure you have the GPU-compiled version of LightGBM installed.
# lgbm_model = MultiOutputRegressor(
#     LGBMRegressor(
#         **best_lgbm_params
#     )
# )

# # --- C. CatBoost (Native) ---
# # CatBoost supports "MultiRMSE" natively. No wrapper needed.
# # This is usually the fastest option for multi-target on GPU.
# cat_model = CatBoostRegressor(
#     **best_cat_params
# )

# # ---------------------------------------------------------
# # 3. Usage Example
# # ---------------------------------------------------------
# # Assuming 'train' and 'test' pandas DataFrames exist
# # and TARGET_NAMES / TARGET_MAX / COLUMNS are defined globally

# # # 1. Run XGBoost
# # print("\n--- Running XGBoost ---")
# # oof_xgb, test_xgb = cross_validate_multioutput_gpu(xgb_model, train_siglip_df, test_siglip_df, feature_engine=feat_engine)
# # compare_results(oof_xgb, train_siglip_df)

# # # 2. Run LightGBM
# # print("\n--- Running LightGBM ---")
# # oof_lgbm, test_lgbm = cross_validate_multioutput_gpu(lgbm_model, train_siglip_df, test_siglip_df, feature_engine=feat_engine)
# # compare_results(oof_lgbm, train_siglip_df)

# # # 3. Run CatBoost
# # print("\n--- Running CatBoost ---")
# # oof_cat, test_cat = cross_validate_multioutput_gpu(cat_model, train_siglip_df, test_siglip_df, feature_engine=feat_engine)
# # compare_results(oof_cat, train_siglip_df)

# # print("\n######## Ridge Regression #######")
# # # ridge_model = MultiOutputRegressor(
# # #     Ridge()
# # # )
# # oof_ridge, pred_test_ri = cross_validate_multioutput_gpu(
# #     Ridge(), 
# #     train_siglip_df, test_siglip_df, 
# #     feature_engine=feat_engine,
# # )
# # compare_results(oof_ridge, train_siglip_df)

# # print("\n###### Bayesian Ridge Regressor #######")
# # bayesian_model = MultiOutputRegressor(
# #     BayesianRidge()
# # )
# # oof_bayesian, pred_test_bri = cross_validate_multioutput_gpu(
# #     bayesian_model, 
# #     train_siglip_df, test_siglip_df, 
# #     feature_engine=feat_engine,
# # )
# # compare_results(oof_bayesian, train_siglip_df)

# print("\n###### GradientBoosting Regressor #######")
# gbm_model = MultiOutputRegressor(
#     GradientBoostingRegressor(**best_gbm_params)
# )

# oof_gb, pred_test_gb = cross_validate_multioutput_gpu(
#     gbm_model, 
#     train_siglip_df, test_siglip_df, 
#     feature_engine=feat_engine,
#     semantic_train=sem_train_full,
#     semantic_test=sem_test_full
# )
# compare_results(oof_gb, train_siglip_df)

# print("\n###### Hist Gradient Boosting Regressor ######")
# hist_model = MultiOutputRegressor(
#     HistGradientBoostingRegressor(**best_hgb_params)
# )

# oof_hb, pred_test_hb = cross_validate_multioutput_gpu(
#     hist_model, 
#     train_siglip_df, test_siglip_df, 
#     feature_engine=feat_engine,
#     semantic_train=sem_train_full,
#     semantic_test=sem_test_full
# )
# compare_results(oof_hb, train_siglip_df)



pred_test = (
    pred_test_hb
    + pred_test_gb
    + pred_test_cat
    + pred_test_lgbm
) / 4

# pred_test = (
#     pred_test_hb
#     + pred_test_et
# ) / 2

# pred_test = (
#     pred_test_gb
#     + pred_test_hb
#     + pred_test_et
# ) / 3

# pred_test = (
#     pred_test_ri
#     + pred_test_gb
#     + pred_test_hb
#     + pred_test_et
# ) / 4

# pred_test = (test_xgb + test_lgbm + test_cat + pred_test_ri + pred_test_bri) / 5
# pred_test = 0.6*pred_test_ri + 0.15*pred_test_gb + 0.15*pred_test_hb + 0.1*pred_test_et



test_df[TARGET_NAMES] = pred_test
test_df = post_process_biomass(test_df)
# test_df['GDM_g'] = test_df['Dry_Green_g'] + test_df['Dry_Clover_g']
# test_df['Dry_Total_g'] = test_df['GDM_g'] + test_df['Dry_Dead_g']
sub_df = melt_table(test_df)
sub_df[['sample_id', 'target']].to_csv("submission_siglip.csv", index=False)

############################################################
# 6️⃣Ensemble submission
############################################################

pd.read_csv("submission_siglip.csv")

def ensemble_submission(files=None, weights=None, postprocess=True, output_name="submission.csv"):
    """
    Create ensemble submission from submission_siglip.csv and submission_dinoV3.csv.
    Uses weights (sum normalized) and writes output to CFG.SUBMISSION_DIR/output_name.
    """
    import os
    import numpy as np
    import pandas as pd

    # Defaults
    if files is None:
        files = ["submission_siglip.csv", "submission_dinoV3.csv"]
    if weights is None:
        # Use the Weight variable defined earlier in the notebook if present
        try:
            w = Weight
        except NameError:
            w = [0.55, 0.45]
    else:
        w = weights

    # Validate
    if len(w) != len(files):
        raise ValueError("Number of weights must match number of files")
    w = np.array(w, dtype=float)
    w = w / w.sum()

    # Read submissions
    series_list = []
    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"File not found: {f}")
        s = pd.read_csv(f).set_index("sample_id")["target"]
        series_list.append(s.rename(os.path.splitext(os.path.basename(f))[0]))

    df = pd.concat(series_list, axis=1)  # align by sample_id

    vals = df.values.astype(float)  # (n_samples, n_models)
    mask = ~np.isnan(vals)

    numer = np.nansum(vals * w.reshape(1, -1), axis=1)
    denom = np.nansum(mask * w.reshape(1, -1), axis=1)
    avg = numer / np.where(denom == 0, 1.0, denom)

    out = pd.DataFrame({"sample_id": df.index, "target": avg})

    if postprocess:
        # convert to wide per image, apply post_process_biomass (must exist in notebook)
        tmp = out.copy()
        tmp[["image_id", "target_name"]] = tmp["sample_id"].str.rsplit("__", n=1, expand=True)
        wide = tmp.pivot(index="image_id", columns="target_name", values="target").reset_index()
        # ensure all cols present
        for c in CFG.ALL_TARGET_COLS:
            if c not in wide.columns:
                wide[c] = 0.0
        # use post_process_biomass defined earlier in the notebook
        wide_proc = post_process_biomass(wide)
        long = wide_proc.melt(id_vars="image_id", value_vars=CFG.ALL_TARGET_COLS, var_name="target_name", value_name="target")
        long["sample_id"] = long["image_id"] + "__" + long["target_name"]
        out = long[["sample_id", "target"]].set_index("sample_id").loc[df.index].reset_index()

    # Align to test.csv ordering if available
    try:
        test_df = pd.read_csv(CFG.TEST_CSV)
        if "sample_id" in test_df.columns:
            out = test_df[["sample_id"]].merge(out, on="sample_id", how="left")
    except Exception:
        pass

    out["target"] = out["target"].fillna(0.0)
    save_path = os.path.join(CFG.SUBMISSION_DIR, output_name)
    out.to_csv(save_path, index=False)
    print(f"Saved ensemble submission: {save_path} (rows={len(out)})")
    return out

# Convenience call using your requested weights
# Run this cell to produce submission.csv
ensemble_submission(files=["submission_siglip.csv","submission_dinoV3.csv"], weights=Weight, postprocess=True, output_name="submission.csv")