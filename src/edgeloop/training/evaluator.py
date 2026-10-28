"""Comprehensive model evaluation and champion-versus-candidate comparison."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import torch

from edgeloop.common.config import get_config
from edgeloop.inference.engine import InferenceEngine

logger = logging.getLogger("edgeloop.training.evaluator")


class ClassMetrics(BaseModel):
    precision: float
    recall: float
    f1_score: float
    support: int


class EvaluationReport(BaseModel):
    model_version: str
    checkpoint_path: str
    eval_dataset: str
    num_samples: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: Dict[str, ClassMetrics]
    confusion_matrix: List[List[int]]
    mean_latency_ms: float
    evaluated_at: str


class ComparisonReport(BaseModel):
    candidate_version: str
    champion_version: str
    passes_all_gates: bool
    macro_f1_delta: float
    accuracy_delta: float
    trash_f1_delta: float
    gate_checks: Dict[str, bool]
    reasons: List[str]
    candidate_metrics: EvaluationReport
    champion_metrics: EvaluationReport


class ModelEvaluator:
    """Evaluates MobileNetV2 models and executes promotion gate checks."""

    def __init__(self):
        self.config = get_config()
        self.classes = self.config.classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def evaluate_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        eval_manifest_or_df: Union[str, Path, pd.DataFrame],
        model_version: Optional[str] = None,
        device: str = "cpu",
        max_samples: Optional[int] = None
    ) -> EvaluationReport:
        """Run evaluation over a dataset and compute accuracy, macro-F1, per-class F1, and confusion matrix."""
        ckpt_p = Path(checkpoint_path)
        if not ckpt_p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_p}")

        engine = InferenceEngine(checkpoint_path=ckpt_p, device=device, log_predictions=False)

        # Load evaluation data
        if isinstance(eval_manifest_or_df, (str, Path)):
            manifest_path = Path(eval_manifest_or_df)
            dataset_name = manifest_path.name
            df = pd.read_csv(manifest_path)
        else:
            dataset_name = "in_memory_dataframe"
            df = eval_manifest_or_df.copy()

        if "label" not in df.columns:
            raise ValueError("Evaluation dataset must contain a 'label' column.")

        # If split column exists, prefer clean_holdout if images exist on disk; otherwise use edge_stream
        if "split" in df.columns:
            has_clean = (df["split"] == "clean_holdout").any()
            if has_clean:
                clean_df = df[df["split"] == "clean_holdout"].copy()
                # Test whether clean_holdout image files exist
                first_sid = clean_df.iloc[0].get("sample_id", "")
                if list((Path(self.config.root_dir) / "research" / "edgeloop").glob(f"**/{first_sid}.jpg")):
                    df = clean_df
                elif (df["split"] == "edge_stream").any():
                    df = df[df["split"] == "edge_stream"].copy()
            elif (df["split"] == "edge_stream").any():
                df = df[df["split"] == "edge_stream"].copy()

        if max_samples is not None and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42).copy()

        # Find image path column
        path_col = None
        for col in ["image_path", "path"]:
            if col in df.columns:
                path_col = col
                break
        if not path_col:
            raise ValueError("Evaluation dataset must contain 'image_path' or 'path' column.")

        y_true = []
        y_pred = []
        latencies = []

        from datetime import datetime, timezone

        for _, row in df.iterrows():
            img_p_str = row[path_col]
            img_p = Path(img_p_str)
            sid = row.get("sample_id", "")

            # If path from kaggle archive is relative or not found, locate in research/edgeloop
            if not img_p.exists():
                # 1. Search by sample_id in research/edgeloop
                if sid:
                    matches = list((Path(self.config.root_dir) / "research" / "edgeloop").glob(f"**/{sid}.jpg"))
                    if matches:
                        img_p = matches[0]

                # 2. Check research corrupted_edge with condition
                if not img_p.exists():
                    fname = img_p.name
                    cond = row.get("condition", "")
                    alt_p = Path(self.config.root_dir) / "research" / "edgeloop" / "corrupted_edge" / cond / fname
                    if alt_p.exists():
                        img_p = alt_p
                    else:
                        matches = list((Path(self.config.root_dir) / "research" / "edgeloop").glob(f"**/{fname}"))
                        if matches:
                            img_p = matches[0]

            if not img_p.exists():
                continue

            try:
                res = engine.predict_image(img_p)
                y_true.append(row["label"])
                y_pred.append(res.predicted_class)
                latencies.append(res.inference_time_ms)
            except Exception as e:
                logger.warning("Skipping unreadable evaluation image %s: %s", img_p, e)

        if not y_true:
            raise RuntimeError("Evaluation failed: No valid images could be evaluated.")

        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        # Per-class metrics
        prec, rec, f1, supp = precision_recall_fscore_support(
            y_true, y_pred, labels=self.classes, zero_division=0
        )
        per_class = {}
        for i, cls_name in enumerate(self.classes):
            per_class[cls_name] = ClassMetrics(
                precision=float(prec[i]),
                recall=float(rec[i]),
                f1_score=float(f1[i]),
                support=int(supp[i])
            )

        cm = confusion_matrix(y_true, y_pred, labels=self.classes).tolist()
        mean_lat = float(np.mean(latencies)) if latencies else 0.0

        return EvaluationReport(
            model_version=model_version or ckpt_p.stem,
            checkpoint_path=str(ckpt_p),
            eval_dataset=dataset_name,
            num_samples=len(y_true),
            accuracy=acc,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            per_class=per_class,
            confusion_matrix=cm,
            mean_latency_ms=mean_lat,
            evaluated_at=datetime.now(timezone.utc).isoformat()
        )

    def compare_against_champion(
        self,
        candidate_metrics: EvaluationReport,
        champion_metrics: EvaluationReport,
        min_macro_f1_improvement: Optional[float] = None,
        max_per_class_drop: Optional[float] = None
    ) -> ComparisonReport:
        """
        Enforce strict model governance promotion gates:
        1. Macro-F1 must improve over champion.
        2. No class (especially 'trash') may suffer regression greater than max_drop.
        """
        min_imp = min_macro_f1_improvement if min_macro_f1_improvement is not None else float(self.config.get("governance.min_macro_f1_improvement", 0.001))
        max_drop = max_per_class_drop if max_per_class_drop is not None else float(self.config.get("governance.max_per_class_f1_drop", 0.02))

        macro_f1_delta = candidate_metrics.macro_f1 - champion_metrics.macro_f1
        acc_delta = candidate_metrics.accuracy - champion_metrics.accuracy

        candidate_trash_f1 = candidate_metrics.per_class.get("trash", ClassMetrics(precision=0, recall=0, f1_score=0, support=0)).f1_score
        champion_trash_f1 = champion_metrics.per_class.get("trash", ClassMetrics(precision=0, recall=0, f1_score=0, support=0)).f1_score
        trash_f1_delta = candidate_trash_f1 - champion_trash_f1

        gates = {}
        reasons = []

        # Gate 1: Macro-F1 Improvement
        gate_f1 = bool(macro_f1_delta >= min_imp)
        gates["macro_f1_improvement"] = gate_f1
        if not gate_f1:
            reasons.append(
                f"Candidate macro-F1 ({candidate_metrics.macro_f1:.4f}) failed to improve over champion ({champion_metrics.macro_f1:.4f}) by at least {min_imp} (delta: {macro_f1_delta:+.4f})."
            )

        # Gate 2: Sensitive Class ('trash') Regression Check
        gate_trash = bool(trash_f1_delta >= -max_drop)
        gates["trash_class_regression_check"] = gate_trash
        if not gate_trash:
            reasons.append(
                f"Candidate trash F1 ({candidate_trash_f1:.4f}) regressed beyond acceptable limit (-{max_drop}) compared to champion ({champion_trash_f1:.4f}) (delta: {trash_f1_delta:+.4f})."
            )

        # Gate 3: General Class Regression Check across all classes
        gate_all_classes = True
        for cls_name in self.classes:
            cand_cls_f1 = candidate_metrics.per_class.get(cls_name).f1_score
            champ_cls_f1 = champion_metrics.per_class.get(cls_name).f1_score
            if cand_cls_f1 - champ_cls_f1 < -max_drop:
                gate_all_classes = False
                reasons.append(
                    f"Class '{cls_name}' regressed by {(cand_cls_f1 - champ_cls_f1):+.4f} (limit: -{max_drop})."
                )
        gates["all_classes_within_tolerance"] = gate_all_classes

        passes_all = gate_f1 and gate_trash and gate_all_classes

        return ComparisonReport(
            candidate_version=candidate_metrics.model_version,
            champion_version=champion_metrics.model_version,
            passes_all_gates=passes_all,
            macro_f1_delta=macro_f1_delta,
            accuracy_delta=acc_delta,
            trash_f1_delta=trash_f1_delta,
            gate_checks=gates,
            reasons=reasons,
            candidate_metrics=candidate_metrics,
            champion_metrics=champion_metrics
        )
