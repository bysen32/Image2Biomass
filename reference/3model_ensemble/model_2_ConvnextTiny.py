from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from collections import OrderedDict
import os
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import cv2
from tqdm import tqdm


# ============================================================================
# Configuration Management
# ============================================================================

@dataclass
class InferenceConfig:
    """
    Data class for managing inference pipeline configuration.

    The following items must match the training configuration:
    - model_name
    - img_size
    - target column names
    """

    # --- Path settings ---
    base_path: Path = Path('/kaggle/input/csiro-biomass')
    test_csv: Path = field(init=False)
    test_image_dir: Path = field(init=False)
    model_dir: Path = Path('/kaggle/input/csiro-exp3/convnext_exp3') # Directory where trained models are stored
    submission_file: str = 'submission_ConvnextTiny.csv'

    # --- Model settings (must match training) ---
    model_name: str = 'convnext_small' # Backbone model to use
    img_size: int = 1000 # Input image size

    # --- Device settings ---
    device: torch.device = field(default_factory=lambda: torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    ))

    # --- Inference settings ---
    batch_size: int = 1
    num_workers: int = 1
    n_folds: int = 5 # Number of folds for ensemble

    # --- Target settings (must match training) ---
    # The 3 targets the model directly predicts
    train_target_cols: list[str] = field(default_factory=lambda: [
        'Dry_Total_g', 'GDM_g', 'Dry_Green_g'
    ])

    # All 5 targets required for submission
    # 全部的5个目标都要预测
    all_target_cols: list[str] = field(default_factory=lambda: [
        'Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g'
    ])

    def __post_init__(self) -> None:
        """Construct paths after initialization"""
        self.test_csv = self.base_path / 'test.csv'
        self.test_image_dir = self.base_path / 'test'

    def display_info(self) -> None:
        """Display configuration information"""
        print(f"{'='*70}")
        print(f"Inference Configuration")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Backbone: {self.model_name}")
        print(f"Image Size: {self.img_size}x{self.img_size}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Ensemble: {self.n_folds}-Fold")
        print(f"TTA: 3 Views (Original, Horizontal Flip, Vertical Flip)")
        print(f"{'='*70}\n")


# ============================================================================
# TTA (Test-Time Augmentation) Transforms
# 测试时增强
# ============================================================================

class TTATransformFactory:
    """
    Factory class for generating Test Time Augmentation transforms.

    Provides 3 different views:
    1. Original (no augmentation)
    2. Horizontal flip
    3. Vertical flip
    """

    def __init__(self, img_size: int):
        """
        Args:
            img_size: Image size after resizing
        """
        self.img_size = img_size

        # Base transforms common to all views
        self.base_transforms = [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Standard ImageNet normalization
            ToTensorV2() # Convert to PyTorch tensor format
        ]

    def get_tta_transforms(self) -> list[A.Compose]:
        """
        Generate 3 transform pipelines for TTA.

        Returns:
            List of 3 Albumentations.Compose objects

        Why not add more TTA variations?
            → Considering the trade-off with inference time.
        """
        # View 1: Original
        original = A.Compose([
            A.Resize(self.img_size, self.img_size),
            *self.base_transforms
        ])

        # View 2: Horizontal flip
        hflip = A.Compose([
            A.HorizontalFlip(p=1.0), # Apply horizontal flip with 100% probability
            A.Resize(self.img_size, self.img_size),
            *self.base_transforms
        ])

        # View 3: Vertical flip
        vflip = A.Compose([
            A.VerticalFlip(p=1.0), # Apply vertical flip with 100% probability
            A.Resize(self.img_size, self.img_size),
            *self.base_transforms
        ])

        # 返回3个增强后的图像：原始、水平翻转、垂直翻转
        return [original, hflip, vflip]


# ============================================================================
# Dataset
# ============================================================================

class TestBiomassDataset(Dataset):
    """
    Two-stream dataset for testing. 
    # 两个流：左图和右图
    Accepts a specific transform pipeline for TTA and applies
    the same augmentation to both left and right images.
    # 接受一个特定的增强管道用于TTA，并应用相同的增强到左右两张图像

    Returns:
        tuple: (img_left, img_right) (left image tensor, right image tensor)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform_pipeline: A.Compose,
        image_dir: Path
    ):
        """
        Args:
            df: DataFrame containing image paths
            transform_pipeline: Augmentation pipeline to apply
            image_dir: Path to the image directory
        """
        self.df = df
        self.transform = transform_pipeline
        self.image_dir = image_dir
        self.image_paths = df['image_path'].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get one sample.

        Args:
            idx: Sample index

        Returns:
            (left_image, right_image): Tuple of left and right image tensors

        Why not apply different augmentations to left/right as in training?
            → During TTA, apply the same transform to both images to preserve symmetry.
        """
        img_path = self.image_paths[idx]
        full_path = self.image_dir / Path(img_path).name

        # Load image (return black image on error)
        # 加载图像，如果失败则返回黑色图像
        image = cv2.imread(str(full_path))

        if image is None:
            print(f"Warning: Failed to load image: {full_path} -> Returning black image")
            # 图像尺寸：宽度2000，高度1000，3个通道（RGB）
            image = np.zeros((1000, 2000, 3), dtype=np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 将图像从BGR格式转换为RGB格式

        # Split into left and right
        # 将图像分割为左图和右图
        height, width = image.shape[:2]
        mid_point = width // 2
        img_left = image[:, :mid_point]
        img_right = image[:, mid_point:]
        # 左图和右图的尺寸：宽度1000，高度1000，3个通道（RGB）

        # Apply same transform to both
        # 应用相同的增强到左右两张图像
        img_left_tensor = self.transform(image=img_left)['image']
        img_right_tensor = self.transform(image=img_right)['image']

        return img_left_tensor, img_right_tensor


# ============================================================================
# Model
# ============================================================================

class BiomassModel(nn.Module):
    """
    Two-stream, three-head regression model.
    # 双流，三个回归头模型

    Uses the exact same architecture as during training.
    """

    def __init__(self, model_name: str, pretrained: bool = False):
        """
        Args:
            model_name: timm model name
            pretrained: Whether to use pretrained weights (False for inference, as custom weights are loaded later)
        """
        super().__init__()

        # Shared backbone for both streams
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,       # Classifier layer is not needed 不需要分类器层
            global_pool='avg'    # Use GAP (Global Average Pooling) 使用全局平均池化
        )

        self.n_features = self.backbone.num_features # Number of output features from the backbone
        self.n_combined = self.n_features * 2        # Number of features after concatenating left and right streams 连接左图和右图后的特征数量

        # Dedicated prediction heads for each of the three targets
        self.head_total = self._create_head() # Head for Dry_Total_g
        self.head_gdm = self._create_head()   # Head for GDM_g
        self.head_green = self._create_head() # Head for Dry_Green_g

    def _create_head(self) -> nn.Sequential:
        """Helper function to generate the MLP structure for a single head"""
        # 创建线性层、ReLU激活函数、Dropout层和输出层
        return nn.Sequential(
            nn.Linear(self.n_combined, self.n_combined // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.n_combined // 2, 1) # Output is a single continuous value 输出是一个连续值
        )

    def forward(
        self,
        img_left: torch.Tensor,
        img_right: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            img_left: Left image tensor [B, C, H, W]
            img_right: Right image tensor [B, C, H, W]

        Returns:
            (total_pred, gdm_pred, green_pred): Tuple of predictions (each [B, 1])
        """
        feat_left = self.backbone(img_left)   # Extract features from the left image
        feat_right = self.backbone(img_right) # Extract features from the right image
        combined = torch.cat([feat_left, feat_right], dim=1) # Concatenate features

        # Calculate predictions with each head
        # 计算每个目标的预测值
        out_total = self.head_total(combined)
        out_gdm = self.head_gdm(combined)
        out_green = self.head_green(combined)

        return out_total, out_gdm, out_green


# ============================================================================
# Model Loader
# ============================================================================

class ModelLoader:
    """
    Class for loading trained models.

    Handles weights saved with DataParallel.
    """

    def __init__(self, config: InferenceConfig):
        """
        Args:
            config: Configuration object
        """
        self.config = config

    def load_fold_models(self) -> list[nn.Module]:
        """
        Load all 5-Fold trained models.

        Returns:
            List of models (each in eval mode on the specified device)

        Raises:
            FileNotFoundError: If a model file is not found
        """
        print(f"\nLoading {self.config.n_folds} trained models...")

        models = []

        for fold in range(self.config.n_folds):
            model_path = self.config.model_dir / f'best_model_fold{fold}.pth'

            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Initialize model
            model = BiomassModel(self.config.model_name, pretrained=False)

            # Load weights
            state_dict = torch.load(model_path, map_location=self.config.device)

            # Remove 'module.' prefix from DataParallel
            # 移除'module.'前缀
            state_dict = self._remove_dataparallel_prefix(state_dict)

            model.load_state_dict(state_dict)
            model.eval()  # Set to evaluation mode
            model.to(self.config.device) # Move model to GPU/CPU

            models.append(model)
            print(f"  ✓ Fold {fold} model loaded")

        print(f"✓ Successfully loaded {len(models)} models\n")
        return models

    @staticmethod
    def _remove_dataparallel_prefix(state_dict: dict) -> dict:
        """
        Remove the 'module.' prefix from keys in a state_dict saved with DataParallel.

        Args:
            state_dict: Model weight dictionary

        Returns:
            Weight dictionary with the prefix removed

        Why not use try-except with a direct load_state_dict call?
            → Explicitly handling the prefix presence improves readability.
        """
        if not any(k.startswith('module.') for k in state_dict.keys()):
            return state_dict  # Return as is if no prefix is found

        # Create a new dictionary with modified keys
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = value

        return new_state_dict


# ============================================================================
# Inference Engine
# ============================================================================

class InferenceEngine:
    """
    Engine for executing TTA + Ensemble inference.
    TTA + 集成推理引擎
    """

    def __init__(
        self,
        models: list[nn.Module],
        config: InferenceConfig
    ):
        """
        Args:
            models: List of trained models (for 5 folds)
            config: Configuration object
        """
        self.models = models
        self.config = config

    def predict_single_view(
        self,
        loader: DataLoader
    ) -> dict[str, np.ndarray]:
        """
        Predict with 5-Fold Ensemble for one TTA view.

        Args:
            loader: DataLoader (with a specific TTA transform applied)

        Returns:
            Dictionary of predictions in the format {'total': [N], 'gdm': [N], 'green': [N]}
        """
        view_preds = {'total': [], 'gdm': [], 'green': []}

        with torch.no_grad(): # Disable gradient calculation
            for img_left, img_right in tqdm(loader, desc="  Predicting", leave=False):
                img_left = img_left.to(self.config.device)
                img_right = img_right.to(self.config.device)

                # Collect predictions from 5 folds
                fold_preds = {'total': [], 'gdm': [], 'green': []}

                for model in self.models:
                    # 预测每个目标的值
                    pred_total, pred_gdm, pred_green = model(img_left, img_right)
                    # 收集5个fold的预测值
                    fold_preds['total'].append(pred_total.cpu())
                    fold_preds['gdm'].append(pred_gdm.cpu())
                    fold_preds['green'].append(pred_green.cpu())

                # Average predictions across 5 folds
                # 平均5个fold的预测值
                avg_total = torch.mean(torch.stack(fold_preds['total']), dim=0)
                avg_gdm = torch.mean(torch.stack(fold_preds['gdm']), dim=0)
                avg_green = torch.mean(torch.stack(fold_preds['green']), dim=0)

                # 将平均后的预测值添加到view_preds中
                view_preds['total'].append(avg_total.numpy())
                view_preds['gdm'].append(avg_gdm.numpy())
                view_preds['green'].append(avg_green.numpy())

        # Concatenate results from all batches
        # 将所有批次的预测结果连接起来
        return {
            k: np.concatenate(v).flatten()
            for k, v in view_preds.items()
        }

    def predict_with_tta(
        self,
        test_df: pd.DataFrame,
        tta_transforms: list[A.Compose]
    ) -> dict[str, np.ndarray]:
        """
        Execute final prediction with TTA + Ensemble.

        Args:
            test_df: Test data DataFrame
            tta_transforms: List of transforms for TTA

        Returns:
            Dictionary of final predictions after TTA averaging
        """
        print(f"\nStarting TTA inference: {len(tta_transforms)} Views × {self.config.n_folds} Folds")

        all_view_preds: list[dict[str, np.ndarray]] = []

        for i, transform in enumerate(tta_transforms):
            print(f"--- TTA View {i+1}/{len(tta_transforms)} ---")

            # Create a dedicated Dataset and DataLoader for this view
            # 创建一个专门用于这个视图的Dataset和DataLoader
            dataset = TestBiomassDataset(
                test_df,
                transform,
                self.config.test_image_dir
            )

            loader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True
            )

            # Perform 5-Fold Ensemble prediction
            view_preds = self.predict_single_view(loader)
            all_view_preds.append(view_preds)

            print(f"  ✓ View {i+1} completed")

        # TTA Ensemble (average across all views)
        # 计算TTA Ensemble（平均所有视图）
        print("\nCalculating TTA Ensemble (averaging all views)...")
        final_preds = {
            'total': np.mean([p['total'] for p in all_view_preds], axis=0),
            'gdm': np.mean([p['gdm'] for p in all_view_preds], axis=0),
            'green': np.mean([p['green'] for p in all_view_preds], axis=0)
        }

        print("✓ Inference completed\n")
        return final_preds


# ============================================================================
# Submission Creation
# ============================================================================

class SubmissionCreator:
    """
    Class for creating the Kaggle submission CSV from predictions.
    """

    def __init__(self, config: InferenceConfig):
        """
        Args:
            config: Configuration object
        """
        self.config = config

    def create(
        self,
        predictions: dict[str, np.ndarray],
        test_df_long: pd.DataFrame,
        test_df_unique: pd.DataFrame
    ) -> None:
        """
        Create and save the submission CSV from predictions.

        Args:
            predictions: Predictions in the format {'total': [...], 'gdm': [...], 'green': [...]}
            test_df_long: Original test.csv (long format)
            test_df_unique: DataFrame with only unique images

        Processing flow:
        1. Calculate 5 targets from 3 predictions
        2. Create a wide-format DataFrame
        3. Convert to long format (melt)
        4. Merge with sample_id
        5. Save as CSV
        """
        print("Creating submission CSV...")

        # 1. Get the 3 predictions output by the model
        pred_total = predictions['total']
        pred_gdm = predictions['gdm']
        pred_green = predictions['green']

        # 2. Calculate the remaining 2 targets using relationships (clip negative values to 0)
        # 计算剩余2个目标的值（将负值截断为0）
        pred_clover = np.maximum(0, pred_gdm - pred_green)
        pred_dead = np.maximum(0, pred_total - pred_gdm)

        # 3. Create a wide-format DataFrame
        # 创建一个宽格式DataFrame
        preds_wide = pd.DataFrame({
            'image_path': test_df_unique['image_path'],
            'Dry_Green_g': pred_green,
            'Dry_Dead_g': pred_dead,
            'Dry_Clover_g': pred_clover,
            'GDM_g': pred_gdm,
            'Dry_Total_g': pred_total
        })

        # 4. Convert to long format (unpivot)
        # 将宽格式转换为长格式
        preds_long = preds_wide.melt(
            id_vars=['image_path'],
            value_vars=self.config.all_target_cols,
            var_name='target_name',
            value_name='target'
        )

        # 5. Merge with the original test.csv to get sample_id
        # 将原始test.csv和预测结果合并
        submission = pd.merge(
            test_df_long[['sample_id', 'image_path', 'target_name']],
            preds_long,
            on=['image_path', 'target_name'],
            how='left'
        )

        # 6. Format and save
        submission = submission[['sample_id', 'target']]
        submission.to_csv(self.config.submission_file, index=False)

        print(f"\n🎉 Submission saved to: {self.config.submission_file}")
        print("\n--- First 5 rows ---")
        print(submission.head())
        print("\n--- Last 5 rows ---")
        print(submission.tail())


# ============================================================================
# Inference Pipeline
# ============================================================================

class InferencePipeline:
    """
    Class that orchestrates the entire inference pipeline.
    """

    def __init__(self, config: InferenceConfig):
        """
        Args:
            config: Configuration object
        """
        self.config = config
        self.model_loader = ModelLoader(config)
        self.tta_factory = TTATransformFactory(config.img_size)
        self.submission_creator = SubmissionCreator(config)

    def run(self) -> None:
        """
        Execute the entire inference pipeline.

        Processing flow:
        1. Load test data
        2. Load models (5-Fold)
        3. Run TTA inference (3 Views × 5 Folds)
        4. Create submission file
        """
        print(f"\n{'='*70}")
        print(f"🚀 Starting Inference Pipeline")
        print(f"{'='*70}")

        try:
            # 1. Load test data
            test_df_long, test_df_unique = self._load_test_data()

            # 2. Load models
            models = self.model_loader.load_fold_models()

            # 3. Run TTA inference
            engine = InferenceEngine(models, self.config)
            tta_transforms = self.tta_factory.get_tta_transforms()
            predictions = engine.predict_with_tta(test_df_unique, tta_transforms)

            # 4. Create submission file
            self.submission_creator.create(
                predictions,
                test_df_long,
                test_df_unique
            )

            print("\n✨ Inference Pipeline Completed Successfully ✨")

        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            raise

        finally:
            # Free up memory
            del models, engine, predictions
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _load_test_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load test data.

        Returns:
            (test_df_long, test_df_unique)
            - test_df_long: Original long-format DataFrame (with sample_id)
            - test_df_unique: DataFrame filtered to unique images only

        Raises:
            FileNotFoundError: If test.csv is not found
        """
        print(f"\nLoading test data: {self.config.test_csv}")

        if not self.config.test_csv.exists():
            raise FileNotFoundError(f"test.csv not found: {self.config.test_csv}")

        test_df_long = pd.read_csv(self.config.test_csv)
        # Since predictions are made per image, create a DataFrame with duplicate image paths removed
        test_df_unique = test_df_long.drop_duplicates(
            subset=['image_path']
        ).reset_index(drop=True)

        print(f"  Long format data: {len(test_df_long)} rows")
        print(f"  Unique images: {len(test_df_unique)} images\n")

        return test_df_long, test_df_unique


# ============================================================================
# Main Execution Block
# ============================================================================

if __name__ == '__main__':
    # Initialize configuration
    config = InferenceConfig()
    config.display_info()

    # Run the pipeline
    pipeline = InferencePipeline(config)
    pipeline.run()

    print("\n" + "="*70)
    print("🎊 All inference processes have completed!")
    print("="*70)