import os
import gc
import random
import warnings
import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass

# Sklearn & Models
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =========================================================================================
# 1. CONFIGURATION & SEEDING
# =========================================================================================
@dataclass
class Config:
    DATA_PATH: Path = Path("/kaggle/input/csiro-biomass/")
    SPLIT_PATH: Path = Path("/kaggle/input/csiro-datasplit/csiro_data_split.csv")
    SIGLIP_PATH: str = "/kaggle/input/google-siglip-so400m-patch14-384/transformers/default/1"
    
    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    PATCH_SIZE: int = 520
    OVERLAP: int = 16
    
    # Target definitions
    TARGET_NAMES = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    TARGET_MAX = {
        "Dry_Clover_g": 71.7865,
        "Dry_Dead_g": 83.8407,
        "Dry_Green_g": 157.9836,
        "Dry_Total_g": 185.70,
        "GDM_g": 157.9836,
    }

cfg = Config()

def seeding(SEED):
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seeding(cfg.SEED)

# =========================================================================================
# 2. DATA LOADING & PRE-PROCESSING
# =========================================================================================
def pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    if 'target' in df.columns.tolist():
        # Train data
        df_pt = pd.pivot_table(
            df, 
            values='target', 
            index=['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm'], 
            columns='target_name', 
            aggfunc='mean'
        ).reset_index()
    else:
        # Test data
        df['target'] = 0
        df_pt = pd.pivot_table(
            df, 
            values='target', 
            index='image_path', 
            columns='target_name', 
            aggfunc='mean'
        ).reset_index()
    return df_pt

def melt_table(df: pd.DataFrame) -> pd.DataFrame:
    melted = df.melt(
        id_vars='image_path',
        value_vars=cfg.TARGET_NAMES,
        var_name='target_name',
        value_name='target'
    )
    # Create sample_id matching submission format
    melted['sample_id'] = (
        melted['image_path']
        .str.replace(r'^.*/', '', regex=True)
        .str.replace('.jpg', '', regex=False)
        + '__' + melted['target_name']
    )
    return melted[['sample_id', 'target']]

def post_process_biomass(df_preds):
    """
    Derive GDM_g and Dry_Total_g from primary predictions.
    IMPORTANT: Keeps Dry_Clover_g fixed at 0.0 and does NOT modify Dry_Green_g or Dry_Dead_g.
    """
    ordered_cols = ["Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g", "GDM_g", "Dry_Total_g"]
    
    # Ensure cols exist
    for c in ordered_cols:
        if c not in df_preds.columns:
            df_preds[c] = 0.0
    
    df_out = df_preds.copy()
    
    # Keep Dry_Clover_g fixed at 0.0 (should already be 0.0 from model prediction)
    df_out['Dry_Clover_g'] = 0.0
    
    # Keep Dry_Green_g and Dry_Dead_g as predicted (NO MODIFICATION)
    # Only derive GDM_g and Dry_Total_g
    df_out['GDM_g'] = df_out['Dry_Green_g'] + df_out['Dry_Clover_g']
    df_out['Dry_Total_g'] = df_out['GDM_g'] + df_out['Dry_Dead_g']
    
    # Clip only derived targets to non-negative
    df_out['GDM_g'] = df_out['GDM_g'].clip(lower=0.0)
    df_out['Dry_Total_g'] = df_out['Dry_Total_g'].clip(lower=0.0)
    
    return df_out

print("Loading Data...")
# Load Train (Metadata Split)
train_df = pd.read_csv(cfg.SPLIT_PATH) 

# --- FIX: Remove pre-existing embedding columns to prevent duplication ---
cols_to_keep = [c for c in train_df.columns if not c.startswith('emb')]
train_df = train_df[cols_to_keep]
# -----------------------------------------------------------------------

# Ensure train paths match local environment
if not str(train_df['image_path'].iloc[0]).startswith('/'):
     train_df['image_path'] = train_df['image_path'].apply(lambda p: str(cfg.DATA_PATH / 'train' / os.path.basename(p)))

# Load Test
test_df_raw = pd.read_csv(cfg.DATA_PATH / 'test.csv')
test_df = pivot_table(test_df_raw)
test_df['image_path'] = test_df['image_path'].apply(lambda p: str(cfg.DATA_PATH / p))

# =========================================================================================
# 3. FEATURE EXTRACTION: SIGLIP IMAGE EMBEDDINGS
# =========================================================================================
def split_image(image, patch_size=520, overlap=16):
    h, w, c = image.shape
    stride = patch_size - overlap
    patches = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y2 = min(y + patch_size, h)
            x2 = min(x + patch_size, w)
            y1 = max(0, y2 - patch_size) # Ensure fixed size
            x1 = max(0, x2 - patch_size)
            patch = image[y1:y2, x1:x2, :]
            patches.append(patch)
    return patches

def compute_embeddings(model_path, df):
    print(f"Computing Embeddings for {len(df)} images...")
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval().to(cfg.DEVICE)
    processor = AutoImageProcessor.from_pretrained(model_path)
    
    EMBEDDINGS = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            img = cv2.imread(row['image_path'])
            if img is None: raise ValueError("Image not found")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            patches = split_image(img, patch_size=cfg.PATCH_SIZE, overlap=cfg.OVERLAP)
            images = [Image.fromarray(p) for p in patches]
            
            # Batch process patches
            inputs = processor(images=images, return_tensors="pt").to(cfg.DEVICE)
            with torch.no_grad():
                features = model.get_image_features(**inputs)
            
            # Average pooling of patches
            avg_embed = features.mean(dim=0).cpu().numpy()
            EMBEDDINGS.append(avg_embed)
        except Exception as e:
            print(f"Error processing {row['image_path']}: {e}")
            # Fallback zero embedding
            EMBEDDINGS.append(np.zeros(1152))
        
    torch.cuda.empty_cache()
    return np.stack(EMBEDDINGS)

# Compute Features
train_embeddings = compute_embeddings(cfg.SIGLIP_PATH, train_df)
test_embeddings = compute_embeddings(cfg.SIGLIP_PATH, test_df)

# Create Feature DataFrames
emb_cols = [f"emb{i}" for i in range(train_embeddings.shape[1])]
train_feat_df = pd.concat([train_df, pd.DataFrame(train_embeddings, columns=emb_cols)], axis=1)
test_feat_df = pd.concat([test_df, pd.DataFrame(test_embeddings, columns=emb_cols)], axis=1)

# Double check column counts
print(f"Train Features Shape: {train_feat_df.shape}")
print(f"Test Features Shape: {test_feat_df.shape}")

# =========================================================================================
# 4. FEATURE EXTRACTION: SEMANTIC FEATURES (TEXT PROBING)
# =========================================================================================
def generate_semantic_features(image_embeddings_np, model_path):
    print("Generating Semantic Features...")
    model = AutoModel.from_pretrained(model_path).to(cfg.DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Anchors
    concept_groups = {
        "bare": ["bare soil", "dirt ground", "sparse vegetation", "exposed earth"],
        "sparse": ["low density pasture", "thin grass", "short clipped grass"],
        "medium": ["average pasture cover", "medium height grass", "grazed pasture"],
        "dense": ["dense tall pasture", "thick grassy volume", "high biomass", "overgrown vegetation"],
        "green": ["lush green vibrant pasture", "photosynthesizing leaves", "fresh growth"],
        "dead": ["dry brown dead grass", "yellow straw", "senesced material", "standing hay"],
        "clover": ["white clover", "trifolium repens", "broadleaf legume", "clover flowers"],
        "grass": ["ryegrass", "blade-like leaves", "fescue", "grassy sward"]
    }
    
    # Encode Concepts
    concept_vectors = {}
    with torch.no_grad():
        for name, prompts in concept_groups.items():
            inputs = tokenizer(prompts, padding="max_length", return_tensors="pt").to(cfg.DEVICE)
            emb = model.get_text_features(**inputs)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            concept_vectors[name] = emb.mean(dim=0, keepdim=True)
            
    # Compute Scores
    img_tensor = torch.tensor(image_embeddings_np, dtype=torch.float32).to(cfg.DEVICE)
    img_tensor = img_tensor / img_tensor.norm(p=2, dim=-1, keepdim=True)
    
    scores = {}
    for name, vec in concept_vectors.items():
        scores[name] = torch.matmul(img_tensor, vec.T).cpu().numpy().flatten()
        
    df_scores = pd.DataFrame(scores)
    # Ratios
    df_scores['ratio_greenness'] = df_scores['green'] / (df_scores['green'] + df_scores['dead'] + 1e-6)
    df_scores['ratio_clover'] = df_scores['clover'] / (df_scores['clover'] + df_scores['grass'] + 1e-6)
    df_scores['ratio_cover'] = (df_scores['dense'] + df_scores['medium']) / (df_scores['bare'] + df_scores['sparse'] + 1e-6)
    
    torch.cuda.empty_cache()
    return df_scores.values

# Combine for semantic generation to ensure consistency
all_emb = np.vstack([train_embeddings, test_embeddings])
all_semantic = generate_semantic_features(all_emb, cfg.SIGLIP_PATH)

sem_train = all_semantic[:len(train_df)]
sem_test = all_semantic[len(train_df):]

# =========================================================================================
# 5. SUPERVISED EMBEDDING ENGINE
# =========================================================================================
class SupervisedEmbeddingEngine:
    def __init__(self, n_pca=0.80, n_pls=8, n_gmm=6, random_state=42):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_pca, random_state=random_state)
        self.pls = PLSRegression(n_components=n_pls, scale=False)
        self.gmm = GaussianMixture(n_components=n_gmm, covariance_type='diag', random_state=random_state)
        self.pls_fitted_ = False

    def fit(self, X, y=None, X_semantic=None):
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        
        # Unsupervised
        self.pca.fit(X_scaled)
        self.gmm.fit(X_scaled)
        
        # Supervised
        if y is not None:
            self.pls.fit(X_scaled, y)
            self.pls_fitted_ = True
        return self

    def transform(self, X, X_semantic=None):
        X_scaled = self.scaler.transform(X)
        
        features = [self.pca.transform(X_scaled)]
        
        if self.pls_fitted_:
            features.append(self.pls.transform(X_scaled))
            
        features.append(self.gmm.predict_proba(X_scaled))
        
        if X_semantic is not None:
            # Normalize semantic
            sem_norm = (X_semantic - np.mean(X_semantic, axis=0)) / (np.std(X_semantic, axis=0) + 1e-6)
            features.append(sem_norm)
            
        return np.hstack(features)

# =========================================================================================
# 6. TRAINING & INFERENCE (5-FOLD CV)
# =========================================================================================
def cross_validate_predict(model_cls, model_params, train_data, test_data, sem_tr, sem_te, feature_engine):
    target_max_arr = np.array([cfg.TARGET_MAX[t] for t in cfg.TARGET_NAMES], dtype=float)
    y_pred_test_accum = np.zeros([len(test_data), len(cfg.TARGET_NAMES)], dtype=float)
    
    # Ensure n_splits is integer
    n_splits = int(train_data['fold'].nunique())
    
    # Pre-extract raw columns to avoid indexing overhead
    # Force float32 to save memory and ensure compatibility
    X_train_full = train_data[emb_cols].values.astype(np.float32)
    X_test_raw = test_data[emb_cols].values.astype(np.float32)
    y_train_full = train_data[cfg.TARGET_NAMES].values.astype(np.float32)
    
    for fold in range(n_splits):
        print(f"Processing Fold {fold}...")
        # Split
        train_mask = train_data['fold'] != fold
        
        X_tr = X_train_full[train_mask]
        y_tr = y_train_full[train_mask] / target_max_arr # Max Scaling
        
        sem_tr_fold = sem_tr[train_mask]
        
        # Feature Engineering (Fit on fold train)
        engine = deepcopy(feature_engine)
        engine.fit(X_tr, y=y_tr, X_semantic=sem_tr_fold)
        
        x_tr_eng = engine.transform(X_tr, X_semantic=sem_tr_fold)
        x_te_eng = engine.transform(X_test_raw, X_semantic=sem_te)
        
        # Train & Predict per target
        fold_test_pred = np.zeros([len(test_data), len(cfg.TARGET_NAMES)])
        
        for k in range(len(cfg.TARGET_NAMES)):
            target_name = cfg.TARGET_NAMES[k]
            
            # Dry_Clover_g: Always predict 0.0 (no training needed)
            if target_name == 'Dry_Clover_g':
                fold_test_pred[:, k] = 0.0
            else:
                # Train and predict for other targets
                model = model_cls(**model_params)
                model.fit(x_tr_eng, y_tr[:, k])
                pred_raw = model.predict(x_te_eng)
                fold_test_pred[:, k] = pred_raw * target_max_arr[k] # Inverse Scale
            
        y_pred_test_accum += fold_test_pred
        
    return y_pred_test_accum / n_splits

# Model Parameters (Optimized)
params_cat = {
    'iterations': 1900, 'learning_rate': 0.045, 'depth': 4, 'l2_leaf_reg': 0.56, 
    'random_strength': 0.045, 'bagging_temperature': 0.98, 'verbose': 0, 'random_state': 42,
    'allow_writing_files': False
}
params_xgb = { # Using GradientBoostingRegressor as proxy
    'n_estimators': 1354, 'learning_rate': 0.010, 'max_depth': 3, 'subsample': 0.60, 
    'random_state': 42
}
params_lgbm = {
    'n_estimators': 807, 'learning_rate': 0.014, 'num_leaves': 48, 'min_child_samples': 19, 
    'subsample': 0.745, 'colsample_bytree': 0.745, 'reg_alpha': 0.21, 'reg_lambda': 3.78,
    'verbose': -1, 'random_state': 42
}
params_hist = {
    'max_iter': 300, 'learning_rate': 0.05, 'max_depth': None, 'l2_regularization': 0.44,
    'random_state': 42
}

feat_engine = SupervisedEmbeddingEngine(n_pca=0.80, n_pls=8, n_gmm=6)

print("Training & Inferring Models...")

# 1. HistGradientBoosting
print("Model: HistGradientBoosting")
pred_hist = cross_validate_predict(
    HistGradientBoostingRegressor, params_hist, 
    train_feat_df, test_feat_df, sem_train, sem_test, feat_engine
)

# 2. GradientBoosting
print("Model: GradientBoosting")
pred_gb = cross_validate_predict(
    GradientBoostingRegressor, params_xgb, 
    train_feat_df, test_feat_df, sem_train, sem_test, feat_engine
)

# 3. CatBoost
print("Model: CatBoost")
pred_cat = cross_validate_predict(
    CatBoostRegressor, params_cat, 
    train_feat_df, test_feat_df, sem_train, sem_test, feat_engine
)

# 4. LightGBM
print("Model: LightGBM")
pred_lgbm = cross_validate_predict(
    LGBMRegressor, params_lgbm, 
    train_feat_df, test_feat_df, sem_train, sem_test, feat_engine
)

# =========================================================================================
# 7. ENSEMBLING & SUBMISSION
# =========================================================================================
print("Ensembling and Post-processing...")
# Simple Average Ensemble
final_pred = (pred_hist + pred_gb + pred_cat + pred_lgbm) / 4.0

# Assign to dataframe
test_feat_df[cfg.TARGET_NAMES] = final_pred

# Post-process (Mass Balance)
test_processed = post_process_biomass(test_feat_df)

# Create Submission File
sub_df = melt_table(test_processed)
output_path = "submission_siglip.csv"
sub_df.to_csv(output_path, index=False)

print(f"✓ Siglip submission generated: {output_path}")
print(sub_df.head())