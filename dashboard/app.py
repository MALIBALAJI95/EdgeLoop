"""EdgeLoop Streamlit Operational Monitoring and Human Review Dashboard."""

import io
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Add src to pythonpath
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir / "src"))

from edgeloop.common.config import get_config
from edgeloop.common.database import get_db
from edgeloop.drift.detector import DriftDetector
from edgeloop.drift.features import DriftFeatureExtractor
from edgeloop.inference.engine import InferenceEngine
from edgeloop.registry.manager import ModelRegistry, PromotionRejectionError
from edgeloop.review.service import ReviewService, ReviewSubmission
from edgeloop.selection.policies import SelectionEngine, StreamRecord

st.set_page_config(
    page_title="EdgeLoop — MLOps Waste Classification",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #28a745;
        margin-bottom: 10px;
    }
    .gov-banner {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_components():
    config = get_config()
    db = get_db()
    reg = ModelRegistry()
    rev_svc = ReviewService()
    detector = DriftDetector()
    extractor = DriftFeatureExtractor()
    return config, db, reg, rev_svc, detector, extractor


config, db, reg, rev_svc, detector, extractor = load_components()

# Sidebar Navigation
st.sidebar.image("https://raw.githubusercontent.com/feathericons/feather/master/icons/cpu.svg", width=40)
st.sidebar.title("EdgeLoop MLOps")
st.sidebar.caption("Drift-Aware Edge-to-Cloud System")

nav = st.sidebar.radio(
    "Navigation",
    [
        "System Overview & Champion",
        "Live Inference & Ingestion",
        "Drift Monitoring",
        "Selective Upload Budget",
        "Human Review Queue",
        "Candidate vs Champion Governance",
        "Model Promotion & Rollback"
    ]
)

champion = reg.get_champion()

# -------------------------------------------------------------
# 1. System Overview & Champion
# -------------------------------------------------------------
if nav == "System Overview & Champion":
    st.header("♻️ EdgeLoop System Overview")
    st.write(
        "EdgeLoop coordinates edge inference, sensor drift detection, budget-constrained uploads, "
        "and human-in-the-loop validation for waste classification."
    )

    # Top KPI Metrics
    col1, col2, col3, col4 = st.columns(4)

    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM inferences")
        total_inf = cursor.fetchone()["cnt"]

        cursor.execute("SELECT COUNT(*) as cnt FROM drift_records WHERE is_drift = 1")
        total_drift = cursor.fetchone()["cnt"]

        cursor.execute("SELECT COUNT(*) as cnt FROM selection_queue")
        total_queued = cursor.fetchone()["cnt"]

        cursor.execute("SELECT COUNT(*) as cnt FROM review_records")
        total_reviewed = cursor.fetchone()["cnt"]

    col1.metric("Total Inferences", total_inf)
    col2.metric("Drift Alerts Triggered", total_drift)
    col3.metric("Selected Uploads", total_queued)
    col4.metric("Verified Ground Truth", total_reviewed)

    st.markdown("---")
    st.subheader("🏆 Active Champion Model")

    if champion:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model Version", champion.version)
        c2.metric("Stage", champion.stage.upper())
        c3.metric("Holdout Macro-F1", f"{champion.macro_f1:.4f}" if champion.macro_f1 else "N/A")
        c4.metric("Trash Class F1", f"{champion.trash_f1:.4f}" if champion.trash_f1 else "N/A")

        st.info(f"**Promotion Reason**: {champion.promotion_reason} | **Approved By**: {champion.approved_by}")

        if champion.metrics_json and "classification_report" in champion.metrics_json:
            st.write("##### Baseline Classification Report (Clean Holdout - 506 Images)")
            rep = champion.metrics_json["classification_report"]
            report_df = pd.DataFrame(rep).transpose()
            st.dataframe(report_df.style.format(precision=4), use_container_width=True)
    else:
        st.warning("No active champion found in registry.")

    st.markdown("""
    <div class="gov-banner">
        <strong>⚠️ Model Governance Rule:</strong> Baseline MobileNetV2 is the active champion.
        Retraining experiments on synthetic streams (uncertainty candidate and unique-aware candidate)
        regressed on the fixed clean holdout (Macro-F1 0.885 and 0.880 vs baseline 0.894).
        Automatic promotion is blocked by governance policy.
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------
# 2. Live Inference & Ingestion
# -------------------------------------------------------------
elif nav == "Live Inference & Ingestion":
    st.header("📸 Live Image Inference & Ingestion")

    st.markdown("""
    Test edge inference on any waste item. Note that EdgeLoop classifies into the **6 TrashNet recycling categories**:
    `paper` (books, pages, magazines), `cardboard`, `glass`, `metal`, `plastic`, and `trash` (general non-recyclables).
    """)

    tab_cam, tab_up, tab_sample = st.tabs(["📷 Live Camera Capture", "📁 Upload Image File", "🧪 Select Research Sample"])

    def display_inference_result(img: Image.Image, sample_id: str, condition: str):
        """Helper to run inference and display telemetry immediately."""
        engine = InferenceEngine(log_predictions=True)
        res = engine.predict_image(img, sample_id=sample_id, condition=condition)

        # Drift features
        feats = extractor.extract_from_image(img)
        is_drift, triggered = detector.check_sample(feats, log_to_db=True)

        col_img, col_res = st.columns([1, 1.5])
        with col_img:
            st.image(img, caption=f"Analyzed Frame: {sample_id} ({condition})", use_container_width=True)

        with col_res:
            st.success(f"### Classification: `{res.predicted_class.upper()}` (Confidence: {res.confidence * 100:.1f}%)")

            # Recycling & category context
            if res.predicted_class in ["paper", "cardboard"]:
                st.info("📚 **Category Context**: Books, notebooks, and cardboard covers are classified as **`PAPER`** or **`CARDBOARD`** for recycling.")
            elif res.predicted_class == "plastic":
                st.info("🧴 **Category Context**: Plastic containers, covers, bottles, or packaging.")
            elif res.predicted_class == "metal":
                st.info("🥫 **Category Context**: Aluminum cans, tin foil, or metal lids.")
            elif res.predicted_class == "glass":
                st.info("🍾 **Category Context**: Glass bottles, jars, or glassware.")
            elif res.predicted_class == "trash":
                st.info("🗑️ **Category Context**: General non-recyclable solid refuse.")

            m1, m2, m3 = st.columns(3)
            m1.metric("Uncertainty (Entropy)", f"{res.entropy:.4f}")
            m2.metric("Inference Latency", f"{res.inference_time_ms:.1f} ms")
            m3.metric("Sensor Drift", "⚠️ DRIFT ALERT" if is_drift else "✅ Clean")

            if is_drift:
                st.warning(f"Drift triggered on: {', '.join(triggered)}")

            # Probabilities chart
            st.write("##### Class Probability Distribution")
            prob_df = pd.DataFrame(list(res.class_probabilities.items()), columns=["Category", "Probability"])
            st.bar_chart(prob_df.set_index("Category"))

            # One-click add to Human Review Queue
            if st.button("📥 Send to Human Review Queue", key=f"queue_{sample_id}"):
                with db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT id FROM inferences WHERE sample_id = ? ORDER BY id DESC LIMIT 1", (sample_id,))
                    row = cursor.fetchone()
                    if row:
                        inf_id = row["id"]
                        db.add_to_selection_queue(
                            inference_id=inf_id,
                            sample_id=sample_id,
                            image_path=f"camera_capture_{sample_id}",
                            strategy="manual_operator_selection",
                            upload_reason="operator_flagged_for_audit",
                            entropy=res.entropy
                        )
                        st.success(f"Sample {sample_id} added to the Review Queue! Check the 'Human Review Queue' tab.")

    with tab_cam:
        st.write("Point your webcam or phone camera at an object. **Inference runs automatically upon capture:**")
        camera_file = st.camera_input("Capture frame from camera", key="webcam_capture")
        if camera_file is not None:
            cam_img = Image.open(camera_file).convert("RGB")
            sample_id = f"camera_{int(time.time())}"
            display_inference_result(cam_img, sample_id=sample_id, condition="live_camera")

        st.caption("💡 For continuous 30 FPS video streaming with a live HUD overlay, run in terminal: `.venv/bin/python scripts/live_camera.py`")

    with tab_up:
        uploaded_file = st.file_uploader("Upload an image file (JPEG/PNG)", type=["jpg", "jpeg", "png"], key="file_upload")
        if uploaded_file is not None:
            up_img = Image.open(uploaded_file).convert("RGB")
            sample_id = Path(uploaded_file.name).stem
            display_inference_result(up_img, sample_id=sample_id, condition="user_upload")

    with tab_sample:
        sample_choice = st.selectbox(
            "Choose a pre-packaged research sample:",
            [
                "research/edgeloop/corrupted_edge/blur/glass_glass1.jpg",
                "research/edgeloop/corrupted_edge/low_light/glass_glass1.jpg",
                "research/edgeloop/corrupted_edge/noise/glass_glass1.jpg",
                "research/edgeloop/corrupted_edge/blur/metal_metal29.jpg",
                "research/edgeloop/corrupted_edge/low_light/paper_paper71.jpg"
            ],
            key="sample_select"
        )
        if sample_choice and Path(sample_choice).exists():
            samp_img = Image.open(sample_choice).convert("RGB")
            sample_id = Path(sample_choice).stem
            condition = Path(sample_choice).parent.name
            display_inference_result(samp_img, sample_id=sample_id, condition=condition)

    st.markdown("---")
    st.subheader("Recent Inferences Feed")
    with db.get_connection() as conn:
        df_inf = pd.read_sql("SELECT id, sample_id, predicted_class, confidence, entropy, latency_ms, condition, created_at FROM inferences ORDER BY id DESC LIMIT 20", conn)
        st.dataframe(df_inf, use_container_width=True)


# -------------------------------------------------------------
# 3. Drift Monitoring
# -------------------------------------------------------------
elif nav == "Drift Monitoring":
    st.header("📉 Drift Monitoring & Image Quality Shifts")

    with db.get_connection() as conn:
        df_drift = pd.read_sql("SELECT * FROM drift_records ORDER BY id DESC LIMIT 100", conn)

    if not df_drift.empty:
        c1, c2, c3 = st.columns(3)
        total = len(df_drift)
        drift_cnt = int(df_drift["is_drift"].sum())
        c1.metric("Recent Monitored", total)
        c2.metric("Alerts Triggered", drift_cnt)
        c3.metric("Drift Alert Rate", f"{(drift_cnt / total) * 100:.1f}%")

        st.write("##### Feature Values Across Edge Stream")
        feature_cols = ["brightness", "contrast", "edge_strength", "high_freq_std", "high_freq_mean_abs"]
        st.line_chart(df_drift[feature_cols])

        st.write("##### Recent Drift Metric Logs")
        st.dataframe(df_drift, use_container_width=True)
    else:
        st.info("No drift records logged yet. Run live inference to generate drift events.")

    st.write("##### Kolmogorov-Smirnov Statistical Two-Sample Drift Test")
    st.write(
        "EdgeLoop runs two-sample Kolmogorov-Smirnov tests comparing observation windows against clean reference baselines. "
        "Unlike fixed thresholds, this detects shifts in sensor noise and blur without clean false-alarms."
    )


# -------------------------------------------------------------
# 4. Selective Upload Budget
# -------------------------------------------------------------
elif nav == "Selective Upload Budget":
    st.header("📡 Selective Upload & Budget Management")

    st.write("Configure edge upload budget and policy constraints:")

    c1, c2 = st.columns(2)
    budget = c1.slider("Upload Budget (Record Count)", min_value=10, max_value=500, value=203)
    policy_name = c2.selectbox("Active Selection Strategy", ["unique_aware", "uncertainty", "random", "drift_aware"])

    # Load research policy comparison
    evaluator_csv = Path(config.root_dir) / "research" / "edgeloop" / "final_selection_comparison.csv"
    if evaluator_csv.exists():
        st.write("##### Historical Policy Comparison on 2,024-Record Research Stream (Budget = 203)")
        comp_df = pd.read_csv(evaluator_csv)
        st.dataframe(comp_df, use_container_width=True)

    st.markdown("""
    - **Random**: Uniform random baseline (captured only 46 errors).
    - **Uncertainty**: Captures the most errors (120), but heavily oversamples multiple corruptions of the same object (only 134 unique images).
    - **Unique-Aware**: Enforces diversity constraint (203 unique images, 102 errors).
    - **Drift-Aware**: Targets sensor degradation (strongly biased toward low-light condition).
    """)


# -------------------------------------------------------------
# 5. Human Review Queue
# -------------------------------------------------------------
elif nav == "Human Review Queue":
    st.header("✍️ Human-in-the-Loop Review Queue")
    st.write("Inspect edge-selected predictions, verify or correct ground-truth labels, and audit provenance.")

    pending_items = rev_svc.get_pending_queue(limit=20)

    if not pending_items:
        st.info("No pending items in review queue. Run selective upload to populate the queue.")
    else:
        st.write(f"**Pending Verification Tasks**: {len(pending_items)} items")

        item = pending_items[0]

        col_img, col_form = st.columns([1, 1.5])

        with col_img:
            img_p = Path(item.image_path)
            if img_p.exists():
                st.image(str(img_p), caption=f"Sample: {item.sample_id}", use_container_width=True)
            else:
                st.warning(f"Image file not found at: {img_p}")

        with col_form:
            st.write(f"**Predicted Class**: `{item.predicted_class}`")
            st.write(f"**Confidence**: `{item.confidence:.4f}` | **Entropy**: `{item.entropy:.4f}`")
            st.write(f"**Upload Reason**: `{item.upload_reason}` ({item.strategy})")
            if item.is_drift:
                st.warning(f"Drift Triggered: {item.triggered_features}")

            classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
            pred_idx = classes.index(item.predicted_class) if item.predicted_class in classes else 0
            true_label = st.selectbox("Ground Truth Label", classes, index=pred_idx)
            reviewer = st.text_input("Reviewer Name", value="operator_1")
            notes = st.text_area("Audit Notes", placeholder="e.g., Clean glass bottle, clear lighting")

            if st.button("Submit Verified Label", type="primary"):
                sub = ReviewSubmission(
                    queue_id=item.queue_id,
                    true_label=true_label,
                    reviewer=reviewer,
                    notes=notes
                )
                rev_svc.submit_review(sub)
                st.success(f"Verified label for {item.sample_id} recorded successfully!")
                st.rerun()

    st.markdown("---")
    st.subheader("Reviewed Ground Truth Export")
    c1, c2 = st.columns([3, 1])
    desc = c1.text_input("Dataset Version Description", "Verified edge samples batch")
    if c2.button("Export Dataset Version"):
        try:
            ver = rev_svc.export_dataset_version(description=desc)
            st.success(f"Exported version `{ver.version}` with {ver.record_count} records!")
        except Exception as e:
            st.error(f"Export error: {e}")


# -------------------------------------------------------------
# 6. Candidate vs Champion Governance
# -------------------------------------------------------------
elif nav == "Candidate vs Champion Governance":
    st.header("⚖️ Candidate vs Champion Governance Comparison")

    st.write(
        "Candidate models must strictly outperform the active champion on holdout Macro-F1 and pass "
        "class-level regression checks (especially for `trash`) before being eligible for promotion."
    )

    models_df = pd.DataFrame([m.model_dump() for m in reg.list_models()])
    st.write("##### Model Registry Entries")
    st.dataframe(models_df[["version", "stage", "accuracy", "macro_f1", "trash_f1", "approved_by"]], use_container_width=True)

    # Detailed comparison table from research holdout
    st.write("##### Fixed Clean Holdout Benchmark (506 Images)")
    holdout_json_path = Path(config.root_dir) / "research" / "edgeloop" / "retraining_holdout_detailed.json"
    if holdout_json_path.exists():
        with open(holdout_json_path) as f:
            holdout_data = json.load(f)

        summary_rows = []
        for item in holdout_data:
            summary_rows.append({
                "Model": item["model"],
                "Holdout Accuracy": f"{item['accuracy'] * 100:.2f}%",
                "Holdout Macro-F1": f"{item['macro_f1']:.4f}",
                "Trash Class F1": f"{item['classification_report']['trash']['f1-score']:.4f}",
                "Gate Status": "CHAMPION" if item["model"] == "baseline" else "REJECTED (REGRESSION)"
            })
        st.table(pd.DataFrame(summary_rows))


# -------------------------------------------------------------
# 7. Model Promotion & Rollback
# -------------------------------------------------------------
elif nav == "Model Promotion & Rollback":
    st.header("🛡️ Safe Model Promotion & Rollback Console")

    st.write(
        "This console executes safe, atomic checkpoint updates and rollbacks. "
        "All promotion actions require explicit approver credentials and documented justification."
    )

    tab_promote, tab_rollback = st.tabs(["Promote Candidate", "Rollback Champion"])

    with tab_promote:
        st.subheader("Promote Candidate to Active Champion")
        all_models = reg.list_models()
        candidate_versions = [m.version for m in all_models if m.stage in ["candidate", "evaluated", "approved"]]

        if not candidate_versions:
            st.info("No candidates currently waiting for promotion.")
        else:
            cand_to_promote = st.selectbox("Select Candidate Version", candidate_versions)
            approver = st.text_input("Approver Identity", placeholder="e.g. Lead MLOps Engineer")
            reason = st.text_area("Promotion Reason", placeholder="Document why this candidate meets production standards")

            if st.button("Execute Promotion", type="primary"):
                try:
                    promoted = reg.promote_candidate(
                        candidate_version=cand_to_promote,
                        approver=approver,
                        promotion_reason=reason
                    )
                    st.success(f"Successfully promoted {promoted.version} to Champion!")
                    st.rerun()
                except PromotionRejectionError as e:
                    st.error(f"Promotion Gate Blocked: {e}")
                except Exception as e:
                    st.error(f"Promotion Failed: {e}")

    with tab_rollback:
        st.subheader("Emergency Rollback to Previous Champion")
        st.warning("Rolling back immediately restores the backup champion checkpoint.")
        rb_approver = st.text_input("Rollback Authorized By", placeholder="Incident Commander")
        rb_reason = st.text_area("Rollback Justification", placeholder="Reason for reverting champion")

        if st.button("Execute Rollback", type="secondary"):
            try:
                reverted = reg.rollback_champion(
                    approver=rb_approver,
                    reason=rb_reason
                )
                st.success(f"Rollback successful! Active champion is now {reverted.version}")
                st.rerun()
            except Exception as e:
                st.error(f"Rollback failed: {e}")
