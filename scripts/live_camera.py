#!/usr/bin/env python3
"""EdgeLoop Live Camera Multi-Object Waste Sorter & Real-Time Drift HUD.

Continuously analyzes video frames from webcam at 16:9 widescreen HD (1280x720),
detects and tracks non-living waste objects (bottles, cans, paper, books, etc.),
classifies materials using the MobileNetV2 champion, suggests color-coded recycling bins,
and automatically synchronizes live detections to the EdgeLoop database.
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
from edgeloop.inference.tracker import StreamObjectTracker


def main():
    parser = argparse.ArgumentParser(description="EdgeLoop Live Camera Waste Sorter")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (cpu, mps, cuda, auto)")
    parser.add_argument("--detection-thresh", type=float, default=0.25, help="Object detection threshold")
    parser.add_argument("--width", type=int, default=1280, help="Camera width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Camera height (default: 720)")
    args = parser.parse_args()

    print("=================================================================")
    print("♻️  EdgeLoop Live AI Waste Sorter (Widescreen HD 1280x720)")
    print("=================================================================")
    print(f"Connecting to Camera ID: {args.camera_id}...")

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"❌ Error: Unable to access camera device {args.camera_id}.")
        print("Tip: Ensure Terminal / IDE has Camera permission in macOS System Settings -> Privacy & Security -> Camera.")
        sys.exit(1)

    # Set 16:9 widescreen HD dimensions
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("Loading Two-Stage Object Localizer (SSDLite320) + MobileNetV2 Champion...")
    detector = WasteObjectDetector(detection_threshold=args.detection_thresh, device=args.device)
    tracker = StreamObjectTracker(iou_threshold=0.35, max_disappeared=20, min_hits_to_confirm=2)
    db = get_db()

    print("✅ Live camera running! Point camera at waste objects (Bottles, Books, Cans, Boxes).")
    print("Press SPACE to flag snapshot for human review | Press Q to exit.")

    saved_timer = 0.0
    logged_track_ids = set()
    last_frame_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue

        now = time.perf_counter()
        dt = now - last_frame_time
        last_frame_time = now
        fps = 1.0 / dt if dt > 0 else 30.0

        # Run non-living object detection & waste classification
        result = detector.detect_and_classify(frame)

        # Update spatial tracker across frames
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        active_tracks = tracker.update(result.objects, full_image=pil_frame)

        # Draw clean, sleek, non-cluttered overlays
        annotated = detector.draw_tracked_overlays(frame, active_tracks if active_tracks else result.objects)
        bgr_display = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        h, w, _ = bgr_display.shape

        # Automatically log newly confirmed unique physical objects to SQLite database
        for trk in active_tracks:
            if trk.hit_streak >= 2 and trk.track_id not in logged_track_ids:
                logged_track_ids.add(trk.track_id)
                sample_id = f"live_obj_{trk.track_id}_{int(time.time())}"
                save_dir = Path("data/incoming")
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"{sample_id}.jpg"
                if trk.crop_image is not None:
                    trk.crop_image.save(save_path)
                else:
                    cv2.imwrite(str(save_path), frame)

                db.log_inference(
                    sample_id=sample_id,
                    image_path=str(save_path.resolve()),
                    predicted_class=trk.waste_class,
                    confidence=trk.waste_confidence,
                    entropy=trk.entropy,
                    latency_ms=result.processing_time_ms,
                    model_version="champion",
                    condition="camera_live_auto"
                )
                print(f"📡 Logged #{trk.track_id}: {trk.detector_label.upper()} -> {trk.waste_class.upper()} ({trk.waste_confidence*100:.0f}%) -> {trk.suggested_bin}")

        # Top Sleek HUD Bar
        overlay = bgr_display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 42), (15, 20, 25), -1)
        cv2.rectangle(overlay, (0, h - 30), (w, h), (15, 20, 25), -1)
        cv2.addWeighted(overlay, 0.75, bgr_display, 0.25, 0, bgr_display)

        # Left: App title and live tracking count
        obj_count = len(active_tracks) if active_tracks else result.total_objects
        cv2.putText(bgr_display, f"EDGELOOP LIVE AI | {obj_count} ITEM(S) IN VIEW", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (88, 166, 255), 2, cv2.LINE_AA)

        # Center: Current Dominant Class & Bin
        if result.dominant_class:
            bin_data = BIN_MAPPINGS.get(result.dominant_class, BIN_MAPPINGS["trash"])
            bin_tag = bin_data["bin_name"].split(" (")[0]
            bgr_col = bin_data["color_bgr"]
            cv2.putText(bgr_display, f">> {result.dominant_class.upper()} -> {bin_tag}", (w // 2 - 120, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, bgr_col, 2, cv2.LINE_AA)

        # Right: FPS and Drift status
        fps_str = f"{fps:.1f} FPS"
        cv2.putText(bgr_display, fps_str, (w - 180, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
        if result.frame_drift:
            cv2.putText(bgr_display, "DRIFT", (w - 80, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(bgr_display, "LIVE", (w - 70, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (63, 185, 80), 2, cv2.LINE_AA)

        # Bottom Bar: Instructions
        cv2.putText(bgr_display, "[SPACE] Queue Item for Review   |   [Q] Exit Stream", (14, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 160, 160), 1, cv2.LINE_AA)

        # User feedback alert
        if saved_timer > 0:
            saved_timer -= dt
            cv2.putText(bgr_display, "QUEUED FOR REVIEW!", (w // 2 - 90, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2, cv2.LINE_AA)

        cv2.imshow("EdgeLoop — Live AI Waste Sorter (Widescreen HD)", bgr_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord(" ") and (active_tracks or result.objects):
            target = active_tracks[0] if active_tracks else result.objects[0]
            sample_id = f"manual_flag_{int(time.time())}"
            save_dir = Path("data/incoming")
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{sample_id}.jpg"
            cv2.imwrite(str(save_path), frame)

            inf_id = db.log_inference(
                sample_id=sample_id,
                image_path=str(save_path.resolve()),
                predicted_class=target.waste_class,
                confidence=target.waste_confidence,
                entropy=target.entropy,
                latency_ms=result.processing_time_ms,
                model_version="champion",
                condition="manual_flagged"
            )
            db.add_to_selection_queue(
                inference_id=inf_id,
                sample_id=sample_id,
                image_path=str(save_path.resolve()),
                strategy="operator_manual",
                upload_reason="live_camera_flag",
                entropy=target.entropy
            )
            saved_timer = 2.0
            print(f"🚩 Queued sample {sample_id} for human verification!")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
