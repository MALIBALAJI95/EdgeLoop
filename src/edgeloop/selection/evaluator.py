"""Evaluation utility for selective upload policies on historical research streams."""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from edgeloop.common.config import get_config
from edgeloop.selection.policies import (
    SelectionEngine,
    StreamRecord,
    SelectionReport,
)


class SelectionPolicyEvaluator:
    """Evaluates selection policies against labeled research datasets."""

    def __init__(self, research_dir: Optional[Path] = None):
        self.config = get_config()
        self.research_dir = research_dir or (Path(self.config.root_dir) / "research" / "edgeloop")
        self.engine = SelectionEngine()

    def load_research_stream(self) -> List[StreamRecord]:
        """
        Merge corruption stream, model edge predictions, and drift flags into StreamRecords.
        True labels are attached to StreamRecord.ground_truth for post-selection evaluation only.
        """
        pred_csv = self.research_dir / "edge_predictions.csv"
        drift_csv = self.research_dir / "stream_features_with_drift.csv"
        manifest_csv = self.research_dir / "corruption_manifest.csv"

        if not pred_csv.exists() or not manifest_csv.exists():
            raise FileNotFoundError(f"Research files not found in {self.research_dir}")

        preds_df = pd.read_csv(pred_csv)
        manifest_df = pd.read_csv(manifest_csv)

        # Merge predictions with corruption manifest to get ground truth
        merged = pd.merge(
            preds_df,
            manifest_df[["sample_id", "condition", "label"]],
            on=["sample_id", "condition"],
            how="left"
        )

        # Merge drift flag if available
        if drift_csv.exists():
            drift_df = pd.read_csv(drift_csv)
            if "drift_flag" in drift_df.columns:
                merged = pd.merge(
                    merged,
                    drift_df[["sample_id", "condition", "drift_flag"]],
                    on=["sample_id", "condition"],
                    how="left"
                )

        records: List[StreamRecord] = []
        for _, row in merged.iterrows():
            drift_val = bool(row.get("drift_flag", False))
            records.append(
                StreamRecord(
                    sample_id=str(row["sample_id"]),
                    image_path=str(row["path"]),
                    predicted_class=str(row["predicted_class"]),
                    confidence=float(row["confidence"]),
                    entropy=float(row["entropy"]),
                    condition=str(row.get("condition", "unknown")),
                    drift_flag=drift_val,
                    file_size_bytes=25000,
                    ground_truth=str(row["label"]) if pd.notna(row.get("label")) else None
                )
            )

        return records

    def compare_all_policies(self, budget_records: int = 203) -> Dict[str, SelectionReport]:
        """Run all registered policies on the research stream and compare metrics."""
        records = self.load_research_stream()
        results: Dict[str, SelectionReport] = {}

        for strategy in ["random", "uncertainty", "unique_aware", "drift_aware"]:
            _, report = self.engine.run_selection(
                records=records,
                strategy=strategy,
                budget_records=budget_records
            )
            results[strategy] = report

        return results

    def to_dataframe(self, results: Dict[str, SelectionReport]) -> pd.DataFrame:
        """Format comparison results into a Pandas DataFrame."""
        rows = []
        for strat, rep in results.items():
            rows.append({
                "strategy": strat,
                "selected_records": rep.selected_records_count,
                "unique_source_images": rep.unique_source_images_count,
                "errors_captured": rep.errors_captured,
                "error_rate": f"{rep.error_rate * 100:.2f}%" if rep.error_rate else "N/A",
                "unique_error_images": rep.unique_error_images
            })
        return pd.DataFrame(rows)
