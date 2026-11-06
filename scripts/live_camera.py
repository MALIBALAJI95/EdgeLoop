#!/usr/bin/env python3
"""EdgeLoop Live Camera Multi-Object Waste Sorter & Real-Time Dashboard Sync.

Continuously analyzes video frames from webcam at 16:9 widescreen HD (1280x720),
detects and tracks non-living waste objects (bottles, cans, paper, books, etc.),
classifies materials using the MobileNetV2 champion, suggests color-coded recycling bins,
and automatically synchronizes live video frames, bounding boxes, and metadata with the
Streamlit EdgeLoop Dashboard at http://localhost:8501.
"""

import argparse
import json
import os
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
    parser = argparse.ArgumentParser(description="EdgeLoop Live Camera Waste Sorter & Dashboard Sync")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (cpu, mps, cuda, auto)")
    parser.add_argument("--detection-thresh", type=float, default=0.12, help="Object detection threshold (default: 0.12)")
    parser.add_argument("--width", type=int, default=1280, help="Camera width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Camera height (default: 720)")
    args = parser.parse_args()

    print("=================================================================")
    print("♻️  EdgeLoop Live AI Edge Camera (Widescreen HD 1280x720)")
    print("=================================================================")
    print(f"Connecting to Camera ID: {args.camera_id}...")

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened() and args.camera_id == 0:
        print("Camera 0 failed, trying Camera ID 1 (Continuity / External)...")
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print(f"❌ Error: Unable to access camera device.")
        print("Tip: Ensure Terminal has Camera permission in macOS System Settings -> Privacy & Security -> Camera.")
        sys.exit(1)

    # Set 16:9 widescreen HD dimensions
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Initialize sync directory
    sync_dir = Path("data/live_stream")
    sync_crops_dir = sync_dir / "crops"
    sync_crops_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Two-Stage Object Localizer (SSDLite320) + MobileNetV2 Champion...")
    detector = WasteObjectDetector(detection_threshold=args.detection_thresh, device=args.device)
    tracker = StreamObjectTracker(iou_threshold=0.35, max_disappeared=15, min_hits_to_confirm=1)
    db = get_db()

    print("✅ Live camera running! Point camera at waste objects (Bottles, Cans, Boxes, Cups).")
    print("📡 Real-time sync active: View feed in Dashboard at http://localhost:8501")
    print("Press SPACE to flag snapshot for human review | Press Q to exit.")

    saved_timer = 0.0
    logged_track_ids = set()
    last_frame_time = time.perf_counter()
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            frame_idx += 1
            now = time.perf_counter()
            dt = now - last_frame_time
            last_frame_time = now
            fps = 1.0 / dt if dt > 0 else 30.0

            # Run non-living object detection & waste classification
            result = detector.detect_and_classify(frame)

            # Update spatial tracker across frames
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            active_tracks = tracker.update(result.objects, full_image=pil_frame)

            # Draw sleek overlays
            annotated = detector.draw_tracked_overlays(frame, active_tracks if active_tracks else result.objects)
            bgr_display = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            h, w, _ = bgr_display.shape

            # Sync crops and database logging for confirmed objects
            obj_payloads = []
            current_objects = active_tracks if active_tracks else result.objects
            for trk in current_objects:
                tid = getattr(trk, "track_id", 1)
                is_trk = hasattr(trk, "hit_streak")

                # Save crop image for dashboard review
                if hasattr(trk, "crop_image") and trk.crop_image is not None:
                    crop_p = sync_crops_dir / f"crop_{tid}.jpg"
                    try:
                        trk.crop_image.save(crop_p)
                    except Exception:
                        pass
                elif hasattr(trk, "box"):
                    bx1, by1, bx2, by2 = trk.box
                    crop_p = sync_crops_dir / f"crop_{tid}.jpg"
                    try:
                        crop_img = pil_frame.crop((bx1, by1, bx2, by2))
                        crop_img.save(crop_p)
                    except Exception:
                        pass

                # Automatically log newly confirmed physical objects immediately
                if is_trk and trk.hit_streak >= 1 and tid not in logged_track_ids:
                    logged_track_ids.add(tid)
                    sample_id = f"live_obj_{tid}_{int(time.time())}"
                    save_dir = Path("data/incoming")
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_dir / f"{sample_id}.jpg"
                    if trk.crop_image is not None:
                        trk.crop_image.save(save_path)
                    else:
                        cv2.imwrite(str(save_path), frame)

                    disp_name = getattr(trk, "display_label", trk.waste_class.title())
                    inf_id = db.log_inference(
                        sample_id=sample_id,
                        image_path=str(save_path.resolve()),
                        predicted_class=trk.waste_class,
                        confidence=trk.waste_confidence,
                        entropy=trk.entropy,
                        latency_ms=result.processing_time_ms,
                        model_version="champion",
                        condition=f"live_cam|{disp_name}"
                    )
                    print(f"📡 Logged #{tid}: {disp_name} [{trk.waste_class.upper()}] ({trk.waste_confidence*100:.0f}%) -> {trk.suggested_bin}")

                    # Automatically send ONLY genuinely uncertain detections or novel dynamic classes to persistent Human Review queue
                    is_novel = trk.waste_class not in ["cardboard", "glass", "metal", "paper", "plastic", "trash", "electronic_waste"]
                    is_unc = getattr(trk, "is_uncertain", False) or trk.waste_confidence < 0.42 or trk.entropy > 1.65 or is_novel
                    if is_unc:
                        reason = "novel_class" if is_novel else ("model_uncertainty" if trk.waste_confidence < 0.42 or trk.entropy > 1.65 else "drift_detected")
                        db.add_to_selection_queue(
                            inference_id=inf_id,
                            sample_id=sample_id,
                            image_path=str(save_path.resolve()),
                            strategy="uncertainty",
                            upload_reason=f"{reason}|{disp_name}",
                            entropy=trk.entropy
                        )
                        print(f"⚠️  [QUEUE] Flagged #{tid} ({disp_name}) for Human Review -> Saved to Persistent Queue!")

                obj_payloads.append({
                    "track_id": tid,
                    "detector_label": trk.detector_label,
                    "detector_confidence": round(float(getattr(trk, "detector_confidence", trk.waste_confidence)), 3),
                    "waste_class": trk.waste_class,
                    "waste_confidence": round(float(trk.waste_confidence), 3),
                    "entropy": round(float(trk.entropy), 3),
                    "suggested_bin": trk.suggested_bin,
                    "bin_code": getattr(trk, "bin_code", "TRASH"),
                    "bin_guidance": getattr(trk, "bin_guidance", ""),
                    "box": [int(v) for v in trk.box],
                    "is_uncertain": getattr(trk, "is_uncertain", False),
                    "display_label": getattr(trk, "display_label", trk.waste_class.title()),
                    "crop_file": f"crop_{tid}.jpg"
                })

            # Broadcast live feed & status to Streamlit dashboard
            feed_tmp = sync_dir / "feed.tmp.jpg"
            feed_final = sync_dir / "feed.jpg"
            cv2.imwrite(str(feed_tmp), bgr_display)
            feed_tmp.replace(feed_final)

            status_payload = {
                "timestamp": time.time(),
                "active": True,
                "fps": round(fps, 1),
                "processing_time_ms": round(result.processing_time_ms, 1),
                "total_objects": len(current_objects),
                "dominant_class": result.dominant_class,
                "dominant_bin": result.dominant_bin,
                "objects": obj_payloads
            }
            status_tmp = sync_dir / "status.tmp.json"
            status_final = sync_dir / "status.json"
            with open(status_tmp, "w") as f:
                json.dump(status_payload, f)
            status_tmp.replace(status_final)

            # Top Sleek HUD Bar for OpenCV Window
            overlay = bgr_display.copy()
            cv2.rectangle(overlay, (0, 0), (w, 42), (15, 20, 25), -1)
            cv2.rectangle(overlay, (0, h - 30), (w, h), (15, 20, 25), -1)
            cv2.addWeighted(overlay, 0.75, bgr_display, 0.25, 0, bgr_display)

            # Left: App title and live tracking count
            obj_count = len(current_objects)
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
            cv2.putText(bgr_display, "[SPACE] Queue for Review  |  [Q] Exit  |  Dashboard: http://localhost:8501", (14, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 160, 160), 1, cv2.LINE_AA)

            # User feedback alert
            if saved_timer > 0:
                saved_timer -= dt
                cv2.putText(bgr_display, "QUEUED FOR REVIEW!", (w // 2 - 90, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2, cv2.LINE_AA)

            cv2.imshow("EdgeLoop — Live AI Waste Sorter (Widescreen HD)", bgr_display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord(" ") and current_objects:
                target = current_objects[0]
                sample_id = f"manual_flag_{int(time.time())}"
                save_dir = Path("data/incoming")
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"{sample_id}.jpg"
                if hasattr(target, "crop_image") and target.crop_image is not None:
                    target.crop_image.save(save_path)
                else:
                    cv2.imwrite(str(save_path), frame)

                disp_name = getattr(target, "display_label", target.waste_class.title())
                inf_id = db.log_inference(
                    sample_id=sample_id,
                    image_path=str(save_path.resolve()),
                    predicted_class=target.waste_class,
                    confidence=target.waste_confidence,
                    entropy=target.entropy,
                    latency_ms=result.processing_time_ms,
                    model_version="champion",
                    condition=f"manual_flag|{disp_name}"
                )
                db.add_to_selection_queue(
                    inference_id=inf_id,
                    sample_id=sample_id,
                    image_path=str(save_path.resolve()),
                    strategy="operator_manual",
                    upload_reason=f"manual_flag|{disp_name}",
                    entropy=target.entropy
                )
                saved_timer = 2.0
                print(f"🚩 [QUEUE] Manually queued #{target.track_id} ({disp_name}) for Human Review!")
                print(f"🚩 Queued sample {sample_id} for human verification!")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        # Mark inactive in sync payload
        try:
            status_final = sync_dir / "status.json"
            with open(status_final, "w") as f:
                json.dump({"active": False, "timestamp": time.time()}, f)
        except Exception:
            pass
        print("Camera released. EdgeLoop feed stopped.")


if __name__ == "__main__":
    main()
