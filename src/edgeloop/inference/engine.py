"""Inference engine for EdgeLoop waste classification."""

import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
from torchvision import models, transforms

from edgeloop.common.config import get_config
from edgeloop.common.database import get_db

logger = logging.getLogger("edgeloop.inference")


class InvalidImageError(ValueError):
    """Raised when an input image fails integrity or dimension checks."""
    pass


class CheckpointLoadError(RuntimeError):
    """Raised when model weights cannot be loaded safely."""
    pass


class PredictionResult(BaseModel):
    """Structured result of an inference execution."""
    sample_id: str
    predicted_class: str
    class_probabilities: Dict[str, float]
    confidence: float = Field(..., ge=0.0, le=1.0)
    entropy: float = Field(..., ge=0.0)
    inference_time_ms: float
    model_version: str
    image_path: Optional[str] = None


class InferenceEngine:
    """MobileNetV2 waste classification inference service."""

    DEFAULT_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        model_version: Optional[str] = None,
        log_predictions: bool = True,
    ):
        self.config = get_config()
        self.log_predictions = log_predictions

        # Resolve device (Apple Silicon mps, CUDA, or CPU fallback)
        self.device = self._resolve_device(device)
        logger.info("Initializing InferenceEngine on device: %s", self.device)

        # Resolve checkpoint path
        if checkpoint_path is None:
            # Check if an active champion exists in artifacts/champion
            champ_p = self.config.champion_checkpoint_path
            if champ_p.exists():
                self.checkpoint_path = champ_p
                self.model_version = model_version or "champion"
            else:
                self.checkpoint_path = self.config.baseline_checkpoint_path
                self.model_version = model_version or "mobilenetv2_baseline"
        else:
            self.checkpoint_path = Path(checkpoint_path)
            self.model_version = model_version or self.checkpoint_path.stem

        self.classes = self.DEFAULT_CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for i, c in enumerate(self.classes)}

        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.get("model.mean", [0.485, 0.456, 0.406]),
                std=self.config.get("model.std", [0.229, 0.224, 0.225])
            )
        ])

        # Load model
        self.model = self._load_model()

    def _resolve_device(self, requested_device: Optional[str]) -> torch.device:
        if requested_device and requested_device != "auto":
            return torch.device(requested_device)

        env_dev = os.getenv("EDGELOOP_DEVICE", "auto").lower()
        if env_dev == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        elif env_dev == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif env_dev == "cpu":
            return torch.device("cpu")

        # Auto detection
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_model(self) -> nn.Module:
        if not self.checkpoint_path.exists():
            raise CheckpointLoadError(f"Checkpoint not found at: {self.checkpoint_path}")

        try:
            model = models.mobilenet_v2(weights=None)
            num_classes = len(self.classes)
            model.classifier[1] = nn.Linear(1280, num_classes)

            ckpt = torch.load(str(self.checkpoint_path), map_location=self.device)

            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
                if "classes" in ckpt and isinstance(ckpt["classes"], list):
                    self.classes = ckpt["classes"]
                    self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
                    self.idx_to_class = {i: c for i, c in enumerate(self.classes)}
            elif isinstance(ckpt, dict):
                state_dict = ckpt
            else:
                raise CheckpointLoadError(f"Unrecognized checkpoint object type: {type(ckpt)}")

            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            logger.info("Successfully loaded MobileNetV2 from %s", self.checkpoint_path)
            return model
        except Exception as e:
            if isinstance(e, CheckpointLoadError):
                raise
            raise CheckpointLoadError(f"Failed to load checkpoint from {self.checkpoint_path}: {e}") from e

    def validate_image_file(self, image_path: Union[str, Path], max_size_bytes: int = 15 * 1024 * 1024) -> Path:
        """Validate file existence, extension, size, and image decode integrity."""
        p = Path(image_path)
        if not p.exists() or not p.is_file():
            raise InvalidImageError(f"File not found: {p}")

        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        if p.suffix.lower() not in valid_exts:
            raise InvalidImageError(f"Unsupported image extension '{p.suffix}'. Must be one of {valid_exts}")

        file_size = p.stat().st_size
        if file_size == 0:
            raise InvalidImageError(f"Image file is empty (0 bytes): {p}")
        if file_size > max_size_bytes:
            raise InvalidImageError(f"Image file exceeds maximum size {max_size_bytes} bytes: {file_size}")

        try:
            with Image.open(p) as img:
                img.verify()
            with Image.open(p) as img:
                w, h = img.size
                if w < 16 or h < 16:
                    raise InvalidImageError(f"Image dimensions too small ({w}x{h}). Minimum is 16x16.")
                if w > 4096 or h > 4096:
                    raise InvalidImageError(f"Image dimensions too large ({w}x{h}). Maximum is 4096x4096.")
        except (UnidentifiedImageError, OSError, SyntaxError) as e:
            raise InvalidImageError(f"Corrupted or invalid image file: {e}") from e

        return p

    def compute_entropy(self, probs: np.ndarray) -> float:
        """Compute Shannon entropy H(P) = -sum(p * ln(p))."""
        eps = 1e-12
        clipped = np.clip(probs, eps, 1.0)
        return float(-np.sum(clipped * np.log(clipped)))

    def predict_image(
        self,
        image_input: Union[str, Path, Image.Image],
        sample_id: Optional[str] = None,
        condition: str = "unknown"
    ) -> PredictionResult:
        """Execute single-image inference with entropy uncertainty and latency measurement."""
        image_path_str: Optional[str] = None

        if isinstance(image_input, (str, Path)):
            validated_path = self.validate_image_file(image_input)
            image_path_str = str(validated_path.resolve())
            sample_id = sample_id or validated_path.stem
            try:
                pil_img = Image.open(validated_path).convert("RGB")
            except Exception as e:
                raise InvalidImageError(f"Failed to decode image: {e}") from e
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            sample_id = sample_id or f"sample_{int(time.time() * 1000)}"
        else:
            raise InvalidImageError(f"Unsupported input type: {type(image_input)}")

        # Transform and forward pass
        start_time = time.perf_counter()
        tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs_t = torch.softmax(logits, dim=1)[0]
            probs = probs_t.detach().cpu().numpy()

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        pred_idx = int(np.argmax(probs))
        predicted_class = self.idx_to_class[pred_idx]
        confidence = float(probs[pred_idx])
        entropy = self.compute_entropy(probs)

        prob_dict = {
            cls_name: float(probs[self.class_to_idx[cls_name]])
            for cls_name in self.classes
        }

        # Log prediction to database if enabled
        if self.log_predictions:
            try:
                db = get_db()
                db.log_inference(
                    sample_id=sample_id,
                    image_path=image_path_str or "memory_buffer",
                    predicted_class=predicted_class,
                    confidence=confidence,
                    entropy=entropy,
                    latency_ms=latency_ms,
                    model_version=self.model_version,
                    condition=condition
                )
            except Exception as e:
                logger.warning("Failed to log inference to database: %s", e)

        return PredictionResult(
            sample_id=sample_id,
            predicted_class=predicted_class,
            class_probabilities=prob_dict,
            confidence=confidence,
            entropy=entropy,
            inference_time_ms=latency_ms,
            model_version=self.model_version,
            image_path=image_path_str
        )

    def predict_batch(
        self,
        image_inputs: List[Union[str, Path, Image.Image]],
        sample_ids: Optional[List[str]] = None,
        condition: str = "unknown"
    ) -> List[PredictionResult]:
        """Execute batch inference."""
        results = []
        for i, item in enumerate(image_inputs):
            sid = sample_ids[i] if sample_ids and i < len(sample_ids) else None
            res = self.predict_image(item, sample_id=sid, condition=condition)
            results.append(res)
        return results
