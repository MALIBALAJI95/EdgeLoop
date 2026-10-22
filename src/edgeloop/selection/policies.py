"""Interchangeable selective upload policies and budget management."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class StreamRecord(BaseModel):
    """Metadata for a stream record subject to selective upload."""
    sample_id: str
    image_path: str
    predicted_class: str
    confidence: float
    entropy: float
    condition: str = "unknown"
    drift_flag: bool = False
    file_size_bytes: int = 0
    # True label is optional and only used by the evaluation utility, never exposed to policies
    ground_truth: Optional[str] = None


class SelectionReport(BaseModel):
    """Diagnostic report summarizing a selective upload run."""
    strategy: str
    total_candidates: int
    budget_allocated: int
    selected_records_count: int
    unique_source_images_count: int
    estimated_transfer_bytes: int
    condition_distribution: Dict[str, int]
    upload_reasons: Dict[str, int]
    # Evaluation-only metrics (populated only when ground-truth is available)
    errors_captured: Optional[int] = None
    error_rate: Optional[float] = None
    unique_error_images: Optional[int] = None


class BaseSelectionPolicy(ABC):
    """Abstract base class for budget-constrained selective upload policies."""

    def __init__(self, name: str, seed: int = 42):
        self.name = name
        self.seed = seed

    @abstractmethod
    def select(
        self,
        records: List[StreamRecord],
        budget_records: int
    ) -> Tuple[List[StreamRecord], List[str]]:
        """
        Select records within budget.
        Returns:
            Tuple of (selected_records, selection_reasons)
        """
        pass


class RandomSelectionPolicy(BaseSelectionPolicy):
    """Baseline policy selecting records uniformly at random."""

    def __init__(self, seed: int = 42):
        super().__init__("random", seed)

    def select(
        self,
        records: List[StreamRecord],
        budget_records: int
    ) -> Tuple[List[StreamRecord], List[str]]:
        if not records or budget_records <= 0:
            return [], []

        k = min(budget_records, len(records))
        rng = np.random.default_rng(self.seed)
        indices = rng.choice(len(records), size=k, replace=False)
        # Sort indices to preserve reproducible deterministic order
        indices.sort()

        selected = [records[i] for i in indices]
        reasons = ["random_selection"] * len(selected)
        return selected, reasons


class UncertaintySelectionPolicy(BaseSelectionPolicy):
    """Selects records with highest predictive uncertainty (Shannon entropy)."""

    def __init__(self, seed: int = 42):
        super().__init__("uncertainty", seed)

    def select(
        self,
        records: List[StreamRecord],
        budget_records: int
    ) -> Tuple[List[StreamRecord], List[str]]:
        if not records or budget_records <= 0:
            return [], []

        k = min(budget_records, len(records))
        # Sort descending by entropy; secondary sort by confidence ascending; third by sample_id
        sorted_records = sorted(
            records,
            key=lambda r: (-r.entropy, r.confidence, r.sample_id)
        )

        selected = sorted_records[:k]
        reasons = ["uncertainty_entropy"] * len(selected)
        return selected, reasons


class UniqueAwareSelectionPolicy(BaseSelectionPolicy):
    """
    Entropy-ranked selection that avoids selecting redundant variants of the
    same physical source image when alternative unique sources are available.
    """

    def __init__(self, seed: int = 42):
        super().__init__("unique_aware", seed)

    def select(
        self,
        records: List[StreamRecord],
        budget_records: int
    ) -> Tuple[List[StreamRecord], List[str]]:
        if not records or budget_records <= 0:
            return [], []

        k = min(budget_records, len(records))
        # Sort all records by uncertainty descending
        sorted_records = sorted(
            records,
            key=lambda r: (-r.entropy, r.confidence, r.sample_id)
        )

        selected: List[StreamRecord] = []
        reasons: List[str] = []
        seen_source_ids: Set[str] = set()
        deferred_records: List[StreamRecord] = []

        # Pass 1: Select at most one variant per unique source image
        for rec in sorted_records:
            if rec.sample_id not in seen_source_ids:
                seen_source_ids.add(rec.sample_id)
                selected.append(rec)
                reasons.append("unique_source_uncertainty")
                if len(selected) == k:
                    break
            else:
                deferred_records.append(rec)

        # Pass 2: If budget remains, fill from deferred records
        if len(selected) < k:
            remaining_needed = k - len(selected)
            for rec in deferred_records[:remaining_needed]:
                selected.append(rec)
                reasons.append("duplicate_source_fallback")

        return selected, reasons


class DriftAwareSelectionPolicy(BaseSelectionPolicy):
    """
    Prioritizes records triggering image-level drift alerts, ranked by uncertainty,
    then fills remaining budget with general uncertain records.
    """

    def __init__(self, seed: int = 42):
        super().__init__("drift_aware", seed)

    def select(
        self,
        records: List[StreamRecord],
        budget_records: int
    ) -> Tuple[List[StreamRecord], List[str]]:
        if not records or budget_records <= 0:
            return [], []

        k = min(budget_records, len(records))

        # Split into drifted and non-drifted
        drift_records = [r for r in records if r.drift_flag]
        non_drift_records = [r for r in records if not r.drift_flag]

        # Sort both descending by entropy
        drift_sorted = sorted(drift_records, key=lambda r: (-r.entropy, r.confidence, r.sample_id))
        non_drift_sorted = sorted(non_drift_records, key=lambda r: (-r.entropy, r.confidence, r.sample_id))

        selected: List[StreamRecord] = []
        reasons: List[str] = []

        # Take up to k from drifted
        for rec in drift_sorted[:k]:
            selected.append(rec)
            reasons.append("drift_alert_uncertainty")

        # Fill remainder from non-drifted
        if len(selected) < k:
            remaining = k - len(selected)
            for rec in non_drift_sorted[:remaining]:
                selected.append(rec)
                reasons.append("non_drift_uncertainty")

        return selected, reasons


class SelectionEngine:
    """Manages policy registry, budget execution, and diagnostic reporting."""

    def __init__(self):
        self.policies: Dict[str, BaseSelectionPolicy] = {
            "random": RandomSelectionPolicy(),
            "uncertainty": UncertaintySelectionPolicy(),
            "unique_aware": UniqueAwareSelectionPolicy(),
            "drift_aware": DriftAwareSelectionPolicy()
        }

    def register_policy(self, name: str, policy: BaseSelectionPolicy) -> None:
        self.policies[name] = policy

    def run_selection(
        self,
        records: List[StreamRecord],
        strategy: str = "unique_aware",
        budget_records: Optional[int] = None,
        budget_fraction: Optional[float] = None
    ) -> Tuple[List[StreamRecord], SelectionReport]:
        if strategy not in self.policies:
            raise ValueError(f"Unknown selection strategy '{strategy}'. Available: {list(self.policies.keys())}")

        policy = self.policies[strategy]

        # Resolve budget
        if budget_records is not None:
            budget = budget_records
        elif budget_fraction is not None:
            budget = int(round(len(records) * budget_fraction))
        else:
            budget = min(203, len(records))

        selected, reasons = policy.select(records, budget)

        # Build diagnostic report
        cond_dist: Dict[str, int] = {}
        reason_dist: Dict[str, int] = {}
        unique_sources: Set[str] = set()
        transfer_bytes = 0

        errors_captured = 0
        has_labels = False
        unique_error_sources: Set[str] = set()

        for rec, reason in zip(selected, reasons):
            cond_dist[rec.condition] = cond_dist.get(rec.condition, 0) + 1
            reason_dist[reason] = reason_dist.get(reason, 0) + 1
            unique_sources.add(rec.sample_id)
            transfer_bytes += rec.file_size_bytes or 25000  # Default ~25KB if 0

            if rec.ground_truth is not None:
                has_labels = True
                if rec.ground_truth != rec.predicted_class:
                    errors_captured += 1
                    unique_error_sources.add(rec.sample_id)

        report = SelectionReport(
            strategy=strategy,
            total_candidates=len(records),
            budget_allocated=budget,
            selected_records_count=len(selected),
            unique_source_images_count=len(unique_sources),
            estimated_transfer_bytes=transfer_bytes,
            condition_distribution=cond_dist,
            upload_reasons=reason_dist,
            errors_captured=errors_captured if has_labels else None,
            error_rate=float(errors_captured / len(selected)) if has_labels and selected else None,
            unique_error_images=len(unique_error_sources) if has_labels else None
        )

        return selected, report
