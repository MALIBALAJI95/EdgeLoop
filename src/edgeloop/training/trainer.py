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
from edgeloop.common.classes import get_class_names
from edgeloop.training.evaluator import ModelEvaluator, EvaluationReport

logger = logging.getLogger("edgeloop.training.trainer")


class TrashDataset(Dataset):
    """PyTorch Dataset loading images and class labels."""

    def __init__(self, df: pd.DataFrame, transform=None, classes: Optional[List[str]] = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        if classes is not None:
            self.classes = list(classes)
        else:
            base_classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
            unique_labels = list(self.df["label"].unique()) if "label" in self.df.columns else []
            extra = [c for c in unique_labels if c not in base_classes]
            self.classes = base_classes + sorted(extra)
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

        lbl = str(row["label"])
        target = self.class_to_idx.get(lbl, 0)
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
        base_checkpoint: Optional[Path] = None,
        target_classes: Optional[List[str]] = None,
        progress_callback: Optional[Any] = None
    ) -> Tuple[Path, Dict[str, Any]]:
        """Fine-tune MobileNetV2 on reviewed data and save candidate checkpoint supporting dynamic classes."""
        # Determine target classes
        base_classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
        registered_classes = get_class_names()
        dataset_labels = list(dataset_df["label"].unique()) if "label" in dataset_df.columns else []

        if target_classes is None:
            combined = base_classes + [c for c in registered_classes if c not in base_classes] + [c for c in dataset_labels if c not in base_classes and c not in registered_classes]
            target_classes = list(dict.fromkeys(combined))

        num_classes = len(target_classes)
        logger.info("Training candidate '%s' supporting %d classes: %s", candidate_name, num_classes, target_classes)

        # Enrich training dataset with base class replay exemplars to prevent catastrophic forgetting
        replay_dir = Path(self.config.root_dir) / "research" / "edgeloop" / "corrupted_edge" / "low_light"
        replay_rows = []
        if replay_dir.exists():
            for bc in base_classes:
                bc_imgs = sorted(list(replay_dir.glob(f"{bc}_*.jpg")))[:4]
                for idx, img_p in enumerate(bc_imgs):
                    replay_rows.append({
                        "sample_id": f"replay_{bc}_{idx+1}",
                        "image_path": str(img_p),
                        "label": bc
                    })
        else:
            base_seeds = [
                ("plastic", "research/edgeloop/corrupted_edge/low_light/plastic_plastic418.jpg", "replay_plastic"),
                ("paper", "research/edgeloop/corrupted_edge/low_light/paper_paper172.jpg", "replay_paper"),
                ("metal", "research/edgeloop/corrupted_edge/blur/metal_metal29.jpg", "replay_metal"),
                ("glass", "research/edgeloop/corrupted_edge/blur/glass_glass1.jpg", "replay_glass"),
                ("cardboard", "research/edgeloop/corrupted_edge/low_light/cardboard_cardboard242.jpg", "replay_cardboard"),
                ("trash", "research/edgeloop/corrupted_edge/low_light/paper_paper71.jpg", "replay_trash"),
            ]
            for lbl, p_str, sid in base_seeds:
                p = Path(self.config.root_dir) / p_str
                if p.exists():
                    replay_rows.append({"sample_id": sid, "image_path": str(p), "label": lbl})

        if replay_rows:
            dataset_df = pd.concat([dataset_df, pd.DataFrame(replay_rows)], ignore_index=True)

        train_df, val_df = self.group_train_val_split(dataset_df, val_fraction=0.20)

        train_loader = DataLoader(
            TrashDataset(train_df, self.train_transform, classes=target_classes),
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            TrashDataset(val_df, self.val_transform, classes=target_classes),
            batch_size=batch_size,
            shuffle=False
        )

        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(1280, num_classes)

        # Initialize from active champion or baseline checkpoint
        init_p = base_checkpoint or (self.config.champion_checkpoint_path if self.config.champion_checkpoint_path.exists() else self.config.baseline_checkpoint_path)
        if init_p.exists():
            ckpt = torch.load(str(init_p), map_location=self.device)
            state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            old_num_classes = state["classifier.1.weight"].shape[0] if "classifier.1.weight" in state else 6

            # Load backbone
            backbone_state = {k: v for k, v in state.items() if not k.startswith("classifier.1.")}
            model.load_state_dict(backbone_state, strict=False)

            # Transfer existing class classifier weights
            min_classes = min(old_num_classes, num_classes)
            if "classifier.1.weight" in state:
                model.classifier[1].weight.data[:min_classes, :] = state["classifier.1.weight"][:min_classes, :]
            if "classifier.1.bias" in state:
                model.classifier[1].bias.data[:min_classes] = state["classifier.1.bias"][:min_classes]

            # Initialize weights for new dynamic classes
            if num_classes > old_num_classes:
                nn.init.kaiming_normal_(model.classifier[1].weight.data[old_num_classes:, :])
                nn.init.zeros_(model.classifier[1].bias.data[old_num_classes:])

            logger.info("Transferred %d existing class weights from %s; initialized %d new classes.", min_classes, init_p, num_classes - min_classes)

        # Freeze early backbone layers (first 14 blocks) to prevent forgetting baseline representations
        for block_idx in range(min(14, len(model.features))):
            for param in model.features[block_idx].parameters():
                param.requires_grad = False

        model.to(self.device)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
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

            if progress_callback:
                try:
                    progress_callback(epoch, epochs, epoch_record)
                except Exception:
                    pass

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
            "num_classes": num_classes,
            "classes": target_classes,
            "class_to_idx": {c: i for i, c in enumerate(target_classes)},
            "seed": self.seed,
            "best_val_accuracy": best_val_acc,
            "training_history": history,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "model_state_dict": model.state_dict()
        }
        torch.save(metadata, str(save_path))
        logger.info("Saved candidate checkpoint to %s", save_path)

        return save_path, metadata
