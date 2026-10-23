#!/usr/bin/env python3
"""EdgeLoop Live Camera Multi-Object Waste Sorter & Real-Time Drift HUD.

Continuously analyzes video frames from webcam, detects multiple visible
waste objects, classifies waste materials using the MobileNetV2 champion,
and overlays bounding boxes with suggested recycling bins at 25-30 FPS.
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
from edgeloop.inference.detector import WasteObjectDetector, BIN_MAPPINGS


def main():
    parser = argparse.ArgumentParser(description="EdgeLoop Live Camera Waste Sorter")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (cpu, mps, cuda, auto)")
    parser.add_argument("--detection-thresh", type=float, default=0.25, help="Object detection threshold")
    args = parser.parse_args()

    print("=================================================================")
    print("♻️  Starting EdgeLoop Live Multi-Object Waste Sorting Stream")
    print("=================================================================")
    print(f"Connecting to Camera ID: {args.camera_id}...")

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"❌ Error: Unable to access camera device {args.camera_id}.")
        print("Tip: If on macOS, grant Terminal/IDE camera permissions in System Settings -> Privacy & Security -> Camera.")
        print("Alternatively, open the Streamlit dashboard at http://127.0.0.1:8501 and use the web camera tab!")
        sys.exit(1)

    # Set frame dimensions
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Initializing Multi-Object Detector (SSDLite320) + MobileNetV2 Champion...")
    detector = WasteObjectDetector(detection_threshold=args.detection_thresh, device=args.device)
    db = get_db()

    print("Camera feed ready! Press SPACE to capture a sample, or Q to exit.")

    saved_timer = 0.0
    last_frame_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        now = time.perf_counter()
        dt = now - last_frame_time
        last_frame_time = now
        fps = 1.0 / dt if dt > 0 else 30.0

        # Run multi-object detection and waste classification
        result = detector.detect_and_classify(frame)

        # Draw bounding boxes and bin overlays
        annotated = detector.draw_overlays(frame, result)
        # Convert back to BGR for OpenCV display
        bgr_display = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        h, w, _ = bgr_display.shape

        # Top status banner
        overlay = bgr_display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (15, 20, 25), -1)
        cv2.rectangle(overlay, (0, h - 35), (w, h), (15, 20, 25), -1)
        cv2.addWeighted(overlay, 0.7, bgr_display, 0.3, 0, bgr_display)

        # Telemetry text
        title_str = f"EDGELOOP LIVE AI SORTER | {result.total_objects} OBJ(S) DETECTED | {fps:.1f} FPS"
        cv2.putText(bgr_display, title_str, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (88, 166, 255), 2, cv2.LINE_AA)

        # Drift badge
        if result.frame_drift:
            cv2.putText(bgr_display, "DRIFT ALERT", (w - 150, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(bgr_display, "CLEAN", (w - 100, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (63, 185, 80), 2, cv2.LINE_AA)

        # Controls text
        cv2.putText(bgr_display, "[SPACE] Log Snapshot to Review Queue   |   [Q] Quit", (14, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)

        # Snapshot feedback banner
        if saved_timer > 0:
            saved_timer -= dt
            cv2.putText(bgr_display, "SNAPSHOT SAVED TO REVIEW QUEUE!", (w // 2 - 180, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2, cv2.LINE_AA)

        cv2.imshow("EdgeLoop — Live Multi-Object Waste Sorter", bgr_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord(" "):
            sample_id = f"camera_snap_{int(time.time())}"
            save_dir = Path("data/incoming")
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{sample_id}.jpg"
            cv2.imwrite(str(save_path), frame)

            dom = result.objects[0]
            inf_id = db.log_inference(
                sample_id=sample_id,
                image_path=str(save_path.resolve()),
                predicted_class=dom.waste_class,
                confidence=dom.waste_confidence,
                entropy=dom.entropy,
                latency_ms=result.processing_time_ms,
                model_version="champion",
                condition="camera_live_snapshot"
            )
            db.add_to_selection_queue(
                inference_id=inf_id,
                sample_id=sample_id,
                image_path=str(save_path.resolve()),
                strategy="operator_manual",
                upload_reason="live_camera_capture",
                entropy=dom.entropy
            )
            saved_timer = 2.0
            print(f"📸 Saved snapshot {sample_id} to database and human review queue!")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
