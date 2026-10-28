"""Controlled candidate training pipeline for MobileNetV2 fine-tuning."""

import json
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from edgeloop.common.config import get_config
from edgeloop.training.evaluator import ModelEvaluator, EvaluationReport

logger = logging.getLogger("edgeloop.training.trainer")


class TrashDataset(Dataset):
    """PyTorch Dataset loading images and class labels."""

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = Path(row["image_path"])

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback zero tensor if image missing
            logger.warning("Failed to open image %s: %s", img_path, e)
            img = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform:
            tensor = self.transform(img)
        else:
            tensor = transforms.ToTensor()(img)

        target = self.class_to_idx[row["label"]]
        return tensor, target


class CandidateTrainer:
    """Trains candidate MobileNetV2 models under strict leakage-free splits."""

    def __init__(self, seed: int = 42, device: str = "auto"):
        self.config = get_config()
        self.seed = seed
        self.set_seed(seed)

        self.candidates_dir = self.config.candidates_dir
        self.candidates_dir.mkdir(parents=True, exist_ok=True)

        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def group_train_val_split(
        self,
        df: pd.DataFrame,
        val_fraction: float = 0.20
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train and validation sets by unique `sample_id` groups.
        This guarantees zero data leakage of identical physical source objects.
        """
        unique_samples = list(df["sample_id"].unique())
        rng = np.random.default_rng(self.seed)
        rng.shuffle(unique_samples)

        val_count = int(round(len(unique_samples) * val_fraction))
        if val_count == 0 and len(unique_samples) >= 2:
            val_count = 1
        elif val_count >= len(unique_samples) and len(unique_samples) > 1:
            val_count = len(unique_samples) - 1

        val_samples = set(unique_samples[:val_count])
        train_samples = set(unique_samples[val_count:])

        train_df = df[df["sample_id"].isin(train_samples)].copy()
        val_df = df[df["sample_id"].isin(val_samples)].copy()

        logger.info(
            "Group split: %d total unique images -> %d train (%d rows), %d val (%d rows)",
            len(unique_samples), len(train_samples), len(train_df), len(val_samples), len(val_df)
        )
        return train_df, val_df

    def train_candidate(
        self,
        dataset_df: pd.DataFrame,
        candidate_name: str,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        base_checkpoint: Optional[Path] = None
    ) -> Tuple[Path, Dict[str, Any]]:
        """Fine-tune MobileNetV2 on reviewed data and save candidate checkpoint."""
        train_df, val_df = self.group_train_val_split(dataset_df, val_fraction=0.20)

        train_loader = DataLoader(
            TrashDataset(train_df, self.train_transform),
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            TrashDataset(val_df, self.val_transform),
            batch_size=batch_size,
            shuffle=False
        )

        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(1280, 6)

        # Initialize from base checkpoint or champion
        init_p = base_checkpoint or self.config.baseline_checkpoint_path
        if init_p.exists():
            ckpt = torch.load(str(init_p), map_location=self.device)
            state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state)
            logger.info("Initialized candidate weights from %s", init_p)

        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        history = []
        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for images, targets in train_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(targets)

            train_loss /= max(1, len(train_df))

            # Validation pass
            model.eval()
            val_loss = 0.0
            correct = 0
            with torch.no_grad():
                for images, targets in val_loader:
                    images, targets = images.to(self.device), targets.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * len(targets)
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == targets).sum().item()

            val_loss /= max(1, len(val_df))
            val_acc = correct / max(1, len(val_df))

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            }
            history.append(epoch_record)
            logger.info("Epoch %d/%d - Train Loss: %.4f - Val Loss: %.4f - Val Acc: %.4f", epoch, epochs, train_loss, val_loss, val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc

        # Save checkpoint
        now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        version = f"{candidate_name}_{now_str}"
        ckpt_filename = f"{version}.pth"
        save_path = self.candidates_dir / ckpt_filename

        metadata = {
            "model_version": version,
            "architecture": "mobilenet_v2",
            "classes": self.config.classes,
            "class_to_idx": {c: i for i, c in enumerate(self.config.classes)},
            "seed": self.seed,
            "best_val_accuracy": best_val_acc,
            "training_history": history,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "model_state_dict": model.state_dict()
        }
        torch.save(metadata, str(save_path))
        logger.info("Saved candidate checkpoint to %s", save_path)

        return save_path, metadata
