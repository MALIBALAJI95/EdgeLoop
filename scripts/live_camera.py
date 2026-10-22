#!/usr/bin/env python3
"""EdgeLoop Live Camera Inference & Real-Time Drift HUD.

Runs real-time classification, entropy uncertainty estimation, and drift
monitoring directly from a webcam video stream with visual HUD overlays.
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edgeloop.common.database import get_db
from edgeloop.drift.detector import DriftDetector
from edgeloop.drift.features import DriftFeatureExtractor
from edgeloop.inference.engine import InferenceEngine


def draw_hud(
    frame: np.ndarray,
    predicted_class: str,
    confidence: float,
    entropy: float,
    latency_ms: float,
    fps: float,
    is_drift: bool,
    triggered_features: list,
    model_version: str,
    saved_badge_timer: float
) -> np.ndarray:
    """Draw interactive HUD telemetry overlays onto video frame."""
    h, w, _ = frame.shape
    overlay = frame.copy()

    # Semi-transparent top dashboard banner
    cv2.rectangle(overlay, (0, 0), (w, 110), (20, 20, 25), -1)
    # Semi-transparent bottom status banner
    cv2.rectangle(overlay, (0, h - 45), (w, h), (20, 20, 25), -1)

    alpha = 0.75
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Choose color based on confidence and drift
    if is_drift:
        theme_color = (0, 140, 255)  # Orange for drift
    elif confidence >= 0.70:
        theme_color = (0, 220, 80)   # Green for confident
    else:
        theme_color = (0, 215, 255)  # Yellow for uncertain

    # Top Title
    cv2.putText(
        frame,
        f"EDGELOOP LIVE EDGE INFERENCE | {model_version.upper()}",
        (15, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )

    # Predicted Class
    class_label = f"CLASS: {predicted_class.upper()}"
    cv2.putText(
        frame,
        class_label,
        (15, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        theme_color,
        2,
        cv2.LINE_AA
    )

    # Confidence & Uncertainty
    conf_str = f"Confidence: {confidence * 100:.1f}%"
    entropy_str = f"Entropy: {entropy:.3f}"
    cv2.putText(
        frame,
        f"{conf_str}  |  {entropy_str}",
        (15, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )

    # Drift Status (Top Right)
    drift_badge = "DRIFT: ALERT" if is_drift else "DRIFT: NORMAL"
    drift_color = (0, 70, 255) if is_drift else (0, 200, 70)
    cv2.putText(
        frame,
        drift_badge,
        (w - 220, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        drift_color,
        2,
        cv2.LINE_AA
    )

    if is_drift and triggered_features:
        trig_str = f"[{', '.join(triggered_features[:2])}]"
        cv2.putText(
            frame,
            trig_str,
            (w - 220, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 140, 255),
            1,
            cv2.LINE_AA
        )

    # Latency and FPS (Top Right)
    perf_str = f"{fps:.1f} FPS | {latency_ms:.1f} ms"
    cv2.putText(
        frame,
        perf_str,
        (w - 220, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )

    # Bottom Banner instructions
    controls_str = "[SPACE] Capture & Save Sample to Review Queue   |   [Q] Quit"
    cv2.putText(
        frame,
        controls_str,
        (15, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (190, 190, 190),
        1,
        cv2.LINE_AA
    )

    # Snapshot Saved Feedback
    if saved_badge_timer > 0:
        cv2.putText(
            frame,
            "SNAPSHOT LOGGED TO DB & QUEUE!",
            (w // 2 - 190, h - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    return frame


def main():
    parser = argparse.ArgumentParser(description="EdgeLoop Live Camera Inference")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (cpu, mps, cuda, auto)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional model checkpoint override")
    parser.add_argument("--throttle-fps", type=float, default=25.0, help="Target FPS limit")
    args = parser.parse_args()

    print("==========================================================")
    print("♻️  Starting EdgeLoop Real-Time Camera Stream")
    print("==========================================================")
    print(f"Connecting to Camera ID: {args.camera_id}...")

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"❌ Error: Unable to access camera device {args.camera_id}.")
        print("Tip: If on macOS, grant Terminal/IDE camera permissions in System Settings -> Privacy & Security -> Camera.")
        print("Alternatively, open the Streamlit dashboard at http://127.0.0.1:8501 and use the browser Camera tab!")
        sys.exit(1)

    # Set frame dimensions
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Initializing MobileNetV2 Inference Engine and Drift Detector...")
    engine = InferenceEngine(checkpoint_path=args.checkpoint, device=args.device, log_predictions=False)
    detector = DriftDetector()
    extractor = DriftFeatureExtractor()
    db = get_db()

    print("Camera feed ready! Press SPACE to capture a sample, or Q to exit.")

    fps_history = []
    saved_timer = 0.0
    last_frame_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to grab frame from camera.")
            time.sleep(0.05)
            continue

        now = time.perf_counter()
        dt = now - last_frame_time
        last_frame_time = now
        fps = 1.0 / dt if dt > 0 else 30.0

        # Convert OpenCV BGR to RGB PIL Image for inference
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        # 1. Run inference
        res = engine.predict_image(pil_img, sample_id=f"live_{int(now * 1000)}", condition="live_camera")

        # 2. Extract drift features
        feats = extractor.extract_from_image(pil_img)
        is_drift, triggered = detector.check_sample(feats, log_to_db=False)

        # Draw HUD overlays
        annotated = draw_hud(
            frame=frame,
            predicted_class=res.predicted_class,
            confidence=res.confidence,
            entropy=res.entropy,
            latency_ms=res.inference_time_ms,
            fps=fps,
            is_drift=is_drift,
            triggered_features=triggered,
            model_version=res.model_version,
            saved_badge_timer=saved_timer
        )

        if saved_timer > 0:
            saved_timer -= dt

        cv2.imshow("EdgeLoop — Live Waste Classification & Drift Monitor", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 'q' or ESC
            print("\nExiting live camera stream.")
            break
        elif key == ord(" "):  # SPACE: save to DB and review queue
            sample_id = f"camera_snap_{int(time.time())}"
            # Save frame to data/incoming
            save_dir = Path("data/incoming")
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{sample_id}.jpg"
            cv2.imwrite(str(save_path), frame)

            # Log inference to DB
            inf_id = db.log_inference(
                sample_id=sample_id,
                image_path=str(save_path.resolve()),
                predicted_class=res.predicted_class,
                confidence=res.confidence,
                entropy=res.entropy,
                latency_ms=res.inference_time_ms,
                model_version=res.model_version,
                condition="live_camera_snapshot"
            )
            # Log drift to DB
            db.log_drift(
                inference_id=inf_id,
                features=feats,
                is_drift=is_drift,
                triggered_features=triggered
            )
            # Queue for human review
            db.add_to_selection_queue(
                inference_id=inf_id,
                sample_id=sample_id,
                image_path=str(save_path.resolve()),
                strategy="camera_manual_capture",
                upload_reason="live_operator_snapshot",
                entropy=res.entropy
            )
            print(f"📸 Saved snapshot {sample_id} to database and human review queue!")
            saved_timer = 2.0  # Show badge for 2 seconds

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
