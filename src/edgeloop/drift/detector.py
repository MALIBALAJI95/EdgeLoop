"""Drift detection using Kolmogorov-Smirnov test and reference distributions."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from edgeloop.common.config import get_config
from edgeloop.common.database import get_db
from edgeloop.drift.features import DriftFeatureExtractor

logger = logging.getLogger("edgeloop.drift")


class DriftDetector:
    """Statistical drift detector comparing edge streams to clean reference distributions."""

    FEATURE_KEYS = [
        "brightness",
        "contrast",
        "edge_strength",
        "high_frequency_std",
        "high_frequency_mean_abs"
    ]

    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        ks_p_value_threshold: float = 0.01,
        z_score_threshold: float = 3.0,
        reference_version: str = "trashnet_clean_reference_v1"
    ):
        self.config = get_config()
        self.ks_p_value_threshold = ks_p_value_threshold or self.config.get("drift.ks_p_value_threshold", 0.01)
        self.z_score_threshold = z_score_threshold
        self.reference_version = reference_version
        self.extractor = DriftFeatureExtractor()

        if reference_data is not None:
            self.reference_df = reference_data
        else:
            self.reference_df = self._load_default_reference()

        self._compute_reference_stats()

    def _load_default_reference(self) -> pd.DataFrame:
        """Load clean reference feature baseline from research artifacts."""
        research_csv = Path(self.config.root_dir) / "research" / "edgeloop" / "stream_features.csv"
        if research_csv.exists():
            df = pd.read_csv(research_csv)
            # Filter clean reference data
            clean_df = df[df["condition"] == "clean"]
            if not clean_df.empty:
                logger.info("Loaded reference distribution from %s (n=%d)", research_csv, len(clean_df))
                return clean_df

        # Fallback synthetic reference distribution if file missing
        logger.warning("Reference feature file not found. Initializing fallback reference baseline.")
        np.random.seed(42)
        n = 500
        return pd.DataFrame({
            "brightness": np.random.normal(0.65, 0.10, n),
            "contrast": np.random.normal(0.25, 0.05, n),
            "edge_strength": np.random.normal(0.18, 0.04, n),
            "high_frequency_std": np.random.normal(10.0, 2.5, n),
            "high_frequency_mean_abs": np.random.normal(3.5, 1.0, n)
        })

    def _compute_reference_stats(self) -> None:
        """Precompute reference means and standard deviations for z-score outlier detection."""
        self.ref_stats: Dict[str, Dict[str, float]] = {}
        for feat in self.FEATURE_KEYS:
            if feat in self.reference_df.columns:
                vals = self.reference_df[feat].dropna().values
                mean = float(np.mean(vals))
                std = float(np.std(vals))
                self.ref_stats[feat] = {
                    "mean": mean,
                    "std": std if std > 1e-6 else 1e-6,
                    "q25": float(np.percentile(vals, 25)),
                    "q75": float(np.percentile(vals, 75))
                }

    def check_sample(
        self,
        features: Dict[str, float],
        inference_id: Optional[int] = None,
        log_to_db: bool = True
    ) -> Tuple[bool, List[str]]:
        """Instant single-sample check based on standard deviations from reference mean."""
        triggered = []
        for feat, stats in self.ref_stats.items():
            val = features.get(feat)
            if val is not None:
                z = abs(val - stats["mean"]) / stats["std"]
                if z > self.z_score_threshold:
                    triggered.append(feat)

        is_drift = len(triggered) > 0

        if log_to_db:
            try:
                db = get_db()
                db.log_drift(
                    inference_id=inference_id,
                    features=features,
                    is_drift=is_drift,
                    triggered_features=triggered
                )
            except Exception as e:
                logger.warning("Failed to log drift event to database: %s", e)

        return is_drift, triggered

    def check_window_drift(
        self,
        window_features: Union[pd.DataFrame, List[Dict[str, float]]],
        p_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Perform two-sample Kolmogorov-Smirnov test comparing observation window to reference."""
        thresh = p_threshold if p_threshold is not None else self.ks_p_value_threshold

        if isinstance(window_features, list):
            window_df = pd.DataFrame(window_features)
        else:
            window_df = window_features

        results = {
            "is_drift": False,
            "reference_version": self.reference_version,
            "window_size": len(window_df),
            "p_value_threshold": thresh,
            "triggered_features": [],
            "ks_results": {}
        }

        if len(window_df) < 5:
            # Need minimum observation window
            return results

        triggered = []
        for feat in self.FEATURE_KEYS:
            if feat in self.reference_df.columns and feat in window_df.columns:
                ref_vals = self.reference_df[feat].dropna().values
                win_vals = window_df[feat].dropna().values

                if len(win_vals) > 0 and len(ref_vals) > 0:
                    ks_stat, p_val = ks_2samp(ref_vals, win_vals)
                    is_flagged = bool(p_val < thresh)

                    results["ks_results"][feat] = {
                        "ks_statistic": float(ks_stat),
                        "p_value": float(p_val),
                        "flagged": is_flagged
                    }
                    if is_flagged:
                        triggered.append(feat)

        results["triggered_features"] = triggered
        results["is_drift"] = len(triggered) > 0
        return results
