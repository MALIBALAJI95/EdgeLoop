"""EdgeLoop: Interactive AI Waste Sorting System & MLOps Operations Dashboard.

Features:
- Live multi-object detection (SSDLite320) + waste classification (MobileNetV2 champion)
- Visual bounding boxes, confidence scores, and recycling bin recommendations
- Dark modern interface with responsive cards and real-time category counters
- Dedicated MLOps backend console: drift monitoring, human review, candidate training & promotion
"""

import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import av
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# Add src to pythonpath
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir / "src"))

from edgeloop.common.config import get_config
from edgeloop.common.database import get_db
from edgeloop.drift.detector import DriftDetector
from edgeloop.drift.features import DriftFeatureExtractor
from edgeloop.inference.detector import WasteObjectDetector, DetectionFrameResult, BIN_MAPPINGS
from edgeloop.inference.engine import InferenceEngine
from edgeloop.registry.manager import ModelRegistry, PromotionRejectionError
from edgeloop.review.service import ReviewService, ReviewSubmission
from edgeloop.selection.policies import SelectionEngine, StreamRecord
from edgeloop.training.trainer import CandidateTrainer

# Configure Streamlit Page
st.set_page_config(
    page_title="EdgeLoop — AI Waste Sorter & MLOps",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme Custom CSS
st.markdown("""
<style>
    /* Dark Theme Core */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Custom Header Bar */
    .top-nav {
        background: linear-gradient(90deg, #161b22 0%, #21262d 100%);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 12px 20px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* Metric Cards */
    .metric-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #58a6ff;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        color: #8b949e;
        letter-spacing: 0.5px;
    }

    /* Bin Recommendation Card */
    .bin-card {
        border-radius: 12px;
        padding: 16px 20px;
        color: #ffffff;
        margin-bottom: 16px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.3);
    }
    .bin-title {
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .bin-subtitle {
        font-size: 13px;
        opacity: 0.9;
    }

    /* Category Badge Grid */
    .cat-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 13px;
        margin: 3px;
    }
    
    /* MLOps Alert Banners */
    .gov-alert {
        background-color: #2e1a1a;
        border-left: 4px solid #f85149;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 10px 0;
        color: #ff7b72;
    }
    .gov-success {
        background-color: #1b2f1e;
        border-left: 4px solid #3fb950;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 10px 0;
        color: #7ee787;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_services():
    config = get_config()
    db = get_db()
    reg = ModelRegistry()
    rev_svc = ReviewService()
    detector = WasteObjectDetector(device="auto")
    trainer = CandidateTrainer(device="auto")
    return config, db, reg, rev_svc, detector, trainer


config, db, reg, rev_svc, detector, trainer = load_all_services()
champion = reg.get_champion()

# Initialize Session State Counters
if "session_detections_count" not in st.session_state:
    st.session_state.session_detections_count = 0
if "last_detection_result" not in st.session_state:
    st.session_state.last_detection_result = None

# Top Bar Status
with db.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as cnt FROM inferences")
    total_lifetime_inf = cursor.fetchone()["cnt"]
    cursor.execute("SELECT COUNT(*) as cnt FROM selection_queue WHERE status = 'pending'")
    pending_queue_cnt = cursor.fetchone()["cnt"]
    cursor.execute("SELECT COUNT(*) as cnt FROM review_records")
    verified_labels_cnt = cursor.fetchone()["cnt"]

champ_ver = champion.version if champion else "baseline"
champ_f1 = f"{champion.macro_f1:.4f}" if champion and champion.macro_f1 else "0.8943"

st.markdown(f"""
<div class="top-nav">
    <div style="font-size: 22px; font-weight: 700; color: #58a6ff;">
        ♻️ EdgeLoop <span style="font-size: 14px; font-weight: 400; color: #8b949e;">| Autonomous Waste Sorter & MLOps</span>
    </div>
    <div style="font-size: 13px; color: #8b949e;">
        <span style="color: #3fb950; font-weight: 600;">● CHAMPION</span>: {champ_ver} (F1: {champ_f1}) &nbsp;|&nbsp;
        <span style="color: #58a6ff;">QUEUE</span>: {pending_queue_cnt} &nbsp;|&nbsp;
        <span style="color: #d29922;">VERIFIED</span>: {verified_labels_cnt}
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation
view_mode = st.radio(
    "Application Mode",
    ["🎯 Live Sorting Console", "🔬 MLOps & Model Operations"],
    horizontal=True,
    label_visibility="collapsed"
)

# ==============================================================================
# SCREEN 1: LIVE SORTING CONSOLE (PRIMARY INTERACTIVE UI)
# ==============================================================================
if view_mode == "🎯 Live Sorting Console":
    # 4 Quick Metric Cards
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    curr_res: Optional[DetectionFrameResult] = st.session_state.last_detection_result
    now_detected = curr_res.total_objects if curr_res else 0
    now_class = curr_res.dominant_class.upper() if (curr_res and curr_res.dominant_class) else "READY"
    now_bin = curr_res.dominant_bin.split(" (")[0] if (curr_res and curr_res.dominant_bin) else "Awaiting Input"

    with m_col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Objects in View</div>
            <div class="metric-value">{now_detected}</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Current Dominant Category</div>
            <div class="metric-value" style="color: #3fb950;">{now_class}</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Assigned Recycling Bin</div>
            <div class="metric-value" style="color: #d29922;">{now_bin}</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Total Processed (Lifetime)</div>
            <div class="metric-value">{total_lifetime_inf}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # 2-Column Main Workspace
    col_camera, col_telemetry = st.columns([1.35, 1.0])

    with col_camera:
        st.subheader("📹 Real-Time Detection Feed")

        input_type = st.radio(
            "Camera Mode",
            ["📷 Camera Snapshot & Analysis", "📁 File Upload Fallback", "🧪 Research Sample", "🎥 Continuous WebRTC Stream"],
            horizontal=True
        )

        input_image: Optional[Image.Image] = None
        sample_tag: str = "frame"

        if input_type == "📷 Camera Snapshot & Analysis":
            st.write("Point your webcam at waste items (e.g. bottles, books, cans, cardboard):")
            cam_capture = st.camera_input("Capture live frame", key="camera_snap")
            if cam_capture is not None:
                input_image = Image.open(cam_capture).convert("RGB")
                sample_tag = f"camera_{int(time.time())}"

        elif input_type == "📁 File Upload Fallback":
            uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="file_up")
            if uploaded_file is not None:
                input_image = Image.open(uploaded_file).convert("RGB")
                sample_tag = Path(uploaded_file.name).stem

        elif input_type == "🧪 Research Sample":
            sample_path = st.selectbox(
                "Choose an edge benchmark sample:",
                [
                    "research/edgeloop/corrupted_edge/blur/glass_glass1.jpg",
                    "research/edgeloop/corrupted_edge/low_light/glass_glass1.jpg",
                    "research/edgeloop/corrupted_edge/noise/glass_glass1.jpg",
                    "research/edgeloop/corrupted_edge/blur/metal_metal29.jpg",
                    "research/edgeloop/corrupted_edge/low_light/paper_paper71.jpg"
                ]
            )
            if sample_path and Path(sample_path).exists():
                input_image = Image.open(sample_path).convert("RGB")
                sample_tag = Path(sample_path).stem

        elif input_type == "🎥 Continuous WebRTC Stream":
            st.info("Continuous WebRTC stream running. Detected objects are outlined dynamically below:")

            class VideoProcessor:
                def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                    img = frame.to_ndarray(format="bgr24")
                    res = detector.detect_and_classify(img)
                    annotated = detector.draw_overlays(img, res)
                    return av.VideoFrame.from_ndarray(annotated, format="bgr24")

            webrtc_streamer(
                key="waste-webrtc",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True
            )

        # Process Single Frame if Captured or Uploaded
        if input_image is not None:
            # 1. Run Multi-Object Detection + Waste Classification
            det_result = detector.detect_and_classify(input_image)
            st.session_state.last_detection_result = det_result
            st.session_state.session_detections_count += det_result.total_objects

            # 2. Draw Visual Overlays (Bounding boxes + pills)
            annotated_frame = detector.draw_overlays(input_image, det_result)

            # Display annotated image
            st.image(annotated_frame, caption=f"Analyzed: {det_result.total_objects} object(s) localized in {det_result.processing_time_ms:.1f}ms", use_container_width=True)

            # Action Buttons Row
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("📥 Log Telemetry to Registry DB", type="primary", use_container_width=True):
                    # Save frame
                    save_dir = Path("data/incoming")
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_p = save_dir / f"{sample_tag}.jpg"
                    input_image.save(save_p)

                    # Log each detected object to database
                    for i, obj in enumerate(det_result.objects):
                        db.log_inference(
                            sample_id=f"{sample_tag}_obj{i+1}",
                            image_path=str(save_p.resolve()),
                            predicted_class=obj.waste_class,
                            confidence=obj.waste_confidence,
                            entropy=obj.entropy,
                            latency_ms=det_result.processing_time_ms,
                            model_version=champ_ver,
                            condition="camera_live"
                        )
                    st.success("✅ Logged detection and telemetry to SQLite database!")

            with btn_col2:
                if st.button("🚩 Queue for Human Verification", use_container_width=True):
                    save_dir = Path("data/incoming")
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_p = save_dir / f"{sample_tag}.jpg"
                    input_image.save(save_p)

                    dom = det_result.objects[0]
                    inf_id = db.log_inference(
                        sample_id=sample_tag,
                        image_path=str(save_p.resolve()),
                        predicted_class=dom.waste_class,
                        confidence=dom.waste_confidence,
                        entropy=dom.entropy,
                        latency_ms=det_result.processing_time_ms,
                        model_version=champ_ver,
                        condition="flagged_manual"
                    )
                    db.add_to_selection_queue(
                        inference_id=inf_id,
                        sample_id=sample_tag,
                        image_path=str(save_p.resolve()),
                        strategy="operator_flagged",
                        upload_reason="uncertainty_review",
                        entropy=dom.entropy
                    )
                    st.warning("⚠️ Added item to the MLOps Human Review Queue!")

    # Right Column: Suggested Bin & Live Statistics
    with col_telemetry:
        st.subheader("♻️ Suggested Recycling Bin")

        if curr_res and curr_res.objects:
            dom_obj = curr_res.objects[0]
            bin_data = BIN_MAPPINGS.get(dom_obj.waste_class, BIN_MAPPINGS["trash"])
            r, g, b = bin_data["color_rgb"]

            st.markdown(f"""
            <div class="bin-card" style="background: linear-gradient(135deg, rgba({r},{g},{b},0.9) 0%, rgba({r},{g},{b},0.6) 100%);">
                <div class="bin-title">{bin_data['bin_name']}</div>
                <div style="font-size: 16px; font-weight: 600; margin: 6px 0;">
                    Target Material: {dom_obj.waste_class.upper()} ({dom_obj.waste_confidence * 100:.1f}% Confidence)
                </div>
                <div class="bin-subtitle">{bin_data['guidance']}</div>
            </div>
            """, unsafe_allow_html=True)

            # If multiple objects detected, display list of all
            if len(curr_res.objects) > 1:
                st.write(f"##### All Detected Items ({len(curr_res.objects)} items):")
                for idx, o in enumerate(curr_res.objects):
                    st.markdown(f"- **Object {idx+1}**: `{o.detector_label}` → **{o.waste_class.upper()}** ({o.waste_confidence*100:.0f}%) → *{o.suggested_bin}*")

            # Explanation of architecture
            with st.expander("ℹ️ How Detection & Classification Work Together"):
                st.markdown("""
                **Two-Stage Edge Pipeline**:
                1. **SSDLite320 Object Localizer**: Pretrained lightweight detector proposes spatial bounding boxes for visible items (bottles, cups, books, boxes, etc.).
                2. **MobileNetV2 Waste Classifier (Champion)**: Each cropped object region is classified by the trained TrashNet champion into material categories (`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`).
                3. **Books & Paper**: In waste facilities, books and notebooks are recycled as **PAPER** (or **CARDBOARD** for hardcovers).
                """)
        else:
            st.info("Awaiting camera input. Capture or upload a frame to view bin recommendations.")

        st.markdown("---")
        st.subheader("📊 Category Distribution")

        cat_counts = db.get_category_counts()
        total_cats = sum(cat_counts.values()) or 1

        c_badges = {
            "paper": ("#1e88e5", "📚 Paper"),
            "cardboard": ("#8d6e63", "📦 Cardboard"),
            "plastic": ("#fbc02d", "🧴 Plastic"),
            "glass": ("#43a047", "🍾 Glass"),
            "metal": ("#78909c", "🥫 Metal"),
            "trash": ("#e53935", "🗑️ Trash")
        }

        cols_badges = st.columns(3)
        for i, (cat_name, (badge_color, label)) in enumerate(c_badges.items()):
            cnt = cat_counts.get(cat_name, 0)
            col_idx = i % 3
            with cols_badges[col_idx]:
                st.markdown(f"""
                <div style="background-color: #21262d; border-radius: 8px; padding: 10px; margin-bottom: 8px; text-align: center; border-left: 4px solid {badge_color};">
                    <div style="font-size: 11px; color: #8b949e;">{label}</div>
                    <div style="font-size: 18px; font-weight: 700; color: #c9d1d9;">{cnt}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("🕒 Recent Detections Feed")
        recent_dets = db.get_recent_detections(limit=5)
        if recent_dets:
            for r in recent_dets:
                st.markdown(f"**{r['sample_id']}**: `{r['predicted_class'].upper()}` ({r['confidence']*100:.1f}%) • *{r['latency_ms']:.1f}ms* <span style='color: #8b949e; font-size: 11px;'>({r['created_at'][:19]})</span>", unsafe_allow_html=True)
        else:
            st.write("No inferences logged yet.")


# ==============================================================================
# SCREEN 2: MLOPS & MODEL OPERATIONS
# ==============================================================================
elif view_mode == "🔬 MLOps & Model Operations":
    st.header("🔬 MLOps Backend & Continuous Learning Pipeline")
    st.write(
        "Audit sensor drift, manage selective upload budgets, verify human-in-the-loop labels, "
        "and execute controlled model retraining with safe promotion gates."
    )

    mlops_tab1, mlops_tab2, mlops_tab3, mlops_tab4 = st.tabs([
        "📈 Sensor Drift Telemetry",
        "✍️ Human Review Queue & Dataset Store",
        "🚀 Candidate Retraining & Promotion Gates",
        "🏆 Model Registry & Rollback History"
    ])

    # -------------------------------------------------------------
    # TAB 1: Sensor Drift Telemetry
    # -------------------------------------------------------------
    with mlops_tab1:
        st.subheader("Drift Monitoring & Quality Shift Analysis")

        with db.get_connection() as conn:
            df_drift = pd.read_sql("SELECT * FROM drift_records ORDER BY id DESC LIMIT 100", conn)

        if not df_drift.empty:
            d_col1, d_col2, d_col3 = st.columns(3)
            total_d = len(df_drift)
            alerts_cnt = int(df_drift["is_drift"].sum())
            d_col1.metric("Monitored Frames", total_d)
            d_col2.metric("Drift Alerts Triggered", alerts_cnt)
            d_col3.metric("Drift Alert Rate", f"{(alerts_cnt / total_d) * 100:.1f}%")

            st.write("##### Image Feature Fluctuations Over Time")
            feat_cols = ["brightness", "contrast", "edge_strength", "high_freq_std", "high_freq_mean_abs"]
            st.line_chart(df_drift[feat_cols])

            st.write("##### Detailed Drift Event Log")
            st.dataframe(df_drift[["id", "inference_id", "brightness", "contrast", "edge_strength", "is_drift", "triggered_features", "timestamp"]], use_container_width=True)
        else:
            st.info("No drift records logged yet. Process frames in the Live Sorting Console to generate telemetry.")

        st.markdown("""
        **Kolmogorov-Smirnov (KS) Statistical Two-Sample Drift Detection**:
        EdgeLoop avoids naive static thresholding by evaluating observation windows against clean reference distributions.
        This prevents low-light over-alerting while capturing real sensor defocus and camera lens contamination.
        """)

    # -------------------------------------------------------------
    # TAB 2: Human Review Queue & Verified Ground Truth
    # -------------------------------------------------------------
    with mlops_tab2:
        st.subheader("Human-in-the-Loop Review Queue")
        st.write("Edge devices upload budget-selected uncertain or drift-flagged images. Operators audit and verify ground truth:")

        pending = rev_svc.get_pending_queue(limit=25)

        if not pending:
            st.info("Review queue is currently empty. Flag items from the Live Sorting Console or run selective upload.")
        else:
            st.write(f"**Pending Audit Items**: {len(pending)} tasks")
            item = pending[0]

            rev_col1, rev_col2 = st.columns([1, 1.3])
            with rev_col1:
                img_p = Path(item.image_path)
                if img_p.exists():
                    st.image(str(img_p), caption=f"Sample: {item.sample_id}", use_container_width=True)
                else:
                    st.warning(f"File at {img_p} is archived.")

            with rev_col2:
                st.write(f"**Model Predicted**: `{item.predicted_class.upper()}` ({item.confidence*100:.1f}%)")
                st.write(f"**Entropy (Uncertainty)**: `{item.entropy:.4f}`")
                st.write(f"**Selection Strategy**: `{item.strategy}` (`{item.upload_reason}`)")
                if item.is_drift:
                    st.warning(f"⚠️ Sensor Drift Alert: {item.triggered_features}")

                classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
                default_idx = classes.index(item.predicted_class) if item.predicted_class in classes else 0
                true_label = st.selectbox("Verified Ground Truth Label", classes, index=default_idx)
                reviewer = st.text_input("Reviewer Name", value="operator_1")
                audit_notes = st.text_area("Audit Notes", placeholder="e.g. Corrected to paper; glossy book cover confused classifier.")

                if st.button("Submit Verified Label & Commit to Ground Truth", type="primary"):
                    sub = ReviewSubmission(
                        queue_id=item.queue_id,
                        true_label=true_label,
                        reviewer=reviewer,
                        notes=audit_notes
                    )
                    rev_svc.submit_review(sub)
                    st.success(f"Committed verification for {item.sample_id}! Item marked reviewed.")
                    st.rerun()

        st.markdown("---")
        st.subheader("Verified Ground Truth Dataset Store")
        st.write(f"**Total Verified Labels Available**: {verified_labels_cnt}")

        exp_col1, exp_col2 = st.columns([2, 1])
        export_desc = exp_col1.text_input("Dataset Manifest Description", "Verified edge samples batch")
        if exp_col2.button("Export Immutable Dataset Manifest"):
            try:
                exp_res = rev_svc.export_dataset_version(description=export_desc)
                st.success(f"Exported `{exp_res.version}` with {exp_res.record_count} verified samples!")
            except Exception as e:
                st.error(f"Export failed: {e}")

    # -------------------------------------------------------------
    # TAB 3: Candidate Retraining & Safe Promotion Gates
    # -------------------------------------------------------------
    with mlops_tab3:
        st.subheader("Controlled Candidate Retraining Pipeline")
        st.write(
            "Train candidate models exclusively when enough verified new labels exist. "
            "Enforce leak-free group splitting by physical source ID to eliminate data contamination."
        )

        st.write(f"**Verified Ground Truth Labels Available**: `{verified_labels_cnt}` samples")

        # Check if enough verified samples exist
        min_required = 4
        can_train = verified_labels_cnt >= min_required

        if not can_train:
            st.warning(f"⚠️ Need at least {min_required} verified review records to trigger candidate fine-tuning. Currently have {verified_labels_cnt}.")
        else:
            st.success(f"✅ Sufficient verified data available ({verified_labels_cnt} samples). Ready for candidate training.")

        tr_col1, tr_col2, tr_col3 = st.columns(3)
        epochs = tr_col1.slider("Training Epochs", min_value=1, max_value=10, value=2)
        lr = tr_col2.select_slider("Learning Rate", options=[1e-5, 5e-5, 1e-4, 5e-4], value=1e-4)
        cand_name = tr_col3.text_input("Candidate Version Name", value=f"candidate_v{int(time.time())}")

        if st.button("🚀 Launch Candidate Fine-Tuning Job", disabled=not can_train, type="primary"):
            with st.spinner("Executing leak-free fine-tuning on verified datasets..."):
                try:
                    with db.get_connection() as conn:
                        df_rev = pd.read_sql("SELECT sample_id, image_path, true_label as label FROM review_records", conn)

                    # Train candidate
                    save_path, metadata = trainer.train_candidate(
                        dataset_df=df_rev,
                        candidate_name=cand_name,
                        epochs=epochs,
                        learning_rate=lr
                    )

                    # Register candidate in registry
                    reg.register_candidate(
                        version=cand_name,
                        checkpoint_path=save_path
                    )
                    st.success(f"Candidate {cand_name} trained successfully and registered!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {e}")

        st.markdown("---")
        st.subheader("Candidate vs Champion Governance Evaluation")

        candidates = [m.version for m in reg.list_models() if m.stage in ["candidate", "evaluated"]]

        if not candidates:
            st.info("No candidate models awaiting evaluation.")
        else:
            selected_cand = st.selectbox("Select Candidate to Evaluate", candidates)
            eval_manifest = "research/edgeloop/dataset_manifest.csv"

            if st.button("⚖️ Run Comparative Holdout Evaluation Gate"):
                with st.spinner("Evaluating candidate and champion on benchmark holdout..."):
                    try:
                        comparison = reg.evaluate_candidate(
                            candidate_version=selected_cand,
                            eval_manifest_path=eval_manifest
                        )

                        st.write("##### Gate Evaluation Results:")
                        g1, g2, g3 = st.columns(3)
                        g1.metric("Macro-F1 Delta", f"{comparison.macro_f1_delta:+.4f}", delta_color="normal")
                        g2.metric("Accuracy Delta", f"{comparison.accuracy_delta:+.4f}", delta_color="normal")
                        g3.metric("Trash F1 Delta", f"{comparison.trash_f1_delta:+.4f}", delta_color="normal")

                        if comparison.passes_all_gates:
                            st.markdown(f"""
                            <div class="gov-success">
                                <strong>✅ PROMOTION APPROVED:</strong> Candidate {selected_cand} passed all statistical acceptance criteria.
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="gov-alert">
                                <strong>❌ PROMOTION REJECTED:</strong> Candidate {selected_cand} failed governance gates. Baseline champion retained.
                                <ul>
                                    {''.join(f'<li>{r}</li>' for r in comparison.reasons)}
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Evaluation error: {e}")

    # -------------------------------------------------------------
    # TAB 4: Model Registry & Rollback History
    # -------------------------------------------------------------
    with mlops_tab4:
        st.subheader("Model Registry & Version Lifecycle")
        st.write("Lifecycle: `candidate → evaluated → approved → champion → archived`")

        all_models = reg.list_models()
        m_rows = []
        for m in all_models:
            m_rows.append({
                "Version": m.version,
                "Stage": m.stage.upper(),
                "Accuracy": f"{m.accuracy*100:.2f}%" if m.accuracy else "N/A",
                "Macro-F1": f"{m.macro_f1:.4f}" if m.macro_f1 else "N/A",
                "Trash F1": f"{m.trash_f1:.4f}" if m.trash_f1 else "N/A",
                "Approved By": m.approved_by or "System",
                "Promotion Reason": m.promotion_reason or "N/A",
                "Updated At": m.updated_at[:19]
            })
        st.dataframe(pd.DataFrame(m_rows), use_container_width=True)

        st.markdown("---")
        st.subheader("Emergency Rollback Console")
        st.write("Revert champion checkpoint to previous version with atomic file swap:")

        rb_col1, rb_col2 = st.columns([1, 1])
        rb_user = rb_col1.text_input("Rollback Operator", "Lead MLOps Engineer")
        rb_reason = rb_col2.text_input("Rollback Reason", "Performance regression detected in field sorting")

        if st.button("⏮️ Execute Emergency Champion Rollback", type="secondary"):
            try:
                reverted = reg.rollback_champion(approver=rb_user, reason=rb_reason)
                st.success(f"Rollback successful! Active champion is now `{reverted.version}`.")
                st.rerun()
            except Exception as e:
                st.error(f"Rollback failed: {e}")
