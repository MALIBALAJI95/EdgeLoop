"""Unit tests for EdgeLoop selective upload policies."""

import pytest
from edgeloop.selection.policies import (
    SelectionEngine,
    StreamRecord,
    RandomSelectionPolicy,
    UncertaintySelectionPolicy,
    UniqueAwareSelectionPolicy,
    DriftAwareSelectionPolicy
)


@pytest.fixture
def mock_stream_records():
    records = []
    # 5 unique images, each with 3 variants (clean, blur, noise)
    for img_id in range(10):
        sample_id = f"image_{img_id:03d}"
        for cond, entropy in [("clean", 0.2 + img_id * 0.05), ("blur", 0.5 + img_id * 0.05), ("noise", 0.8 + img_id * 0.05)]:
            drift = cond in ["blur", "noise"] and img_id < 5
            records.append(
                StreamRecord(
                    sample_id=sample_id,
                    image_path=f"/path/{sample_id}_{cond}.jpg",
                    predicted_class="glass" if img_id % 2 == 0 else "metal",
                    confidence=0.9 - (entropy * 0.4),
                    entropy=entropy,
                    condition=cond,
                    drift_flag=drift,
                    file_size_bytes=20000,
                    ground_truth="glass"
                )
            )
    return records


def test_budget_enforcement(mock_stream_records):
    engine = SelectionEngine()
    budget = 7
    selected, report = engine.run_selection(
        mock_stream_records,
        strategy="random",
        budget_records=budget
    )
    assert len(selected) == budget
    assert report.selected_records_count == budget
    assert report.budget_allocated == budget


def test_budget_fraction_enforcement(mock_stream_records):
    engine = SelectionEngine()
    selected, report = engine.run_selection(
        mock_stream_records,
        strategy="uncertainty",
        budget_fraction=0.20
    )
    expected_count = int(round(len(mock_stream_records) * 0.20))
    assert len(selected) == expected_count


def test_uncertainty_policy_ranking(mock_stream_records):
    policy = UncertaintySelectionPolicy()
    selected, reasons = policy.select(mock_stream_records, budget_records=5)
    assert len(selected) == 5

    # Check that entropy is non-increasing
    entropies = [r.entropy for r in selected]
    for i in range(len(entropies) - 1):
        assert entropies[i] >= entropies[i + 1]


def test_unique_aware_deduplication(mock_stream_records):
    policy = UniqueAwareSelectionPolicy()
    budget = 8
    selected, reasons = policy.select(mock_stream_records, budget_records=budget)
    assert len(selected) == budget

    # All selected records must come from distinct sample_ids since there are 10 unique images
    sample_ids = [r.sample_id for r in selected]
    assert len(set(sample_ids)) == budget


def test_random_policy_determinism(mock_stream_records):
    policy1 = RandomSelectionPolicy(seed=123)
    policy2 = RandomSelectionPolicy(seed=123)

    sel1, _ = policy1.select(mock_stream_records, budget_records=10)
    sel2, _ = policy2.select(mock_stream_records, budget_records=10)

    assert [r.sample_id for r in sel1] == [r.sample_id for r in sel2]
    assert [r.condition for r in sel1] == [r.condition for r in sel2]


def test_drift_aware_selection_prioritizes_drift(mock_stream_records):
    policy = DriftAwareSelectionPolicy()
    selected, reasons = policy.select(mock_stream_records, budget_records=5)
    assert len(selected) == 5
    for r in selected:
        assert r.drift_flag is True
