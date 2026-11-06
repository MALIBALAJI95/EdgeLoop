"""Object tracking and deduplication for continuous video streams.

Implements spatial IoU matching and temporal smoothing to prevent duplicate
frame counts and eliminate bounding box flickering as objects move.
"""

import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from edgeloop.inference.detector import DetectedWasteObject, DetectionFrameResult


def compute_iou(boxA: List[int], boxB: List[int]) -> float:
    """Compute Intersection over Union (IoU) between two boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return float(iou)


class TrackedWasteObject:
    """Represents a unique tracked physical waste object across video frames."""

    def __init__(
        self,
        track_id: int,
        detection: Optional[DetectedWasteObject] = None,
        frame_crop: Optional[Image.Image] = None,
        **kwargs
    ):
        self.track_id = track_id
        if detection is not None:
            self.box = list(detection.box)
            self.detector_label = detection.detector_label
            self.detector_confidence = getattr(detection, "detector_confidence", 1.0)
            self.waste_class = detection.waste_class
            self.waste_confidence = detection.waste_confidence
            self.entropy = detection.entropy
            self.suggested_bin = detection.suggested_bin
            self.bin_code = detection.bin_code
            self.bin_guidance = detection.bin_guidance
            self.is_drift = detection.is_drift

            # Uncertainty threshold: genuinely ambiguous prediction (low confidence and elevated entropy)
            self.is_uncertain = getattr(detection, "is_uncertain", False) or (detection.waste_confidence < 0.65 and detection.entropy > 0.80)
            self.display_label = getattr(detection, "display_label", detection.waste_class)
        else:
            self.box = list(kwargs.get("box", [0, 0, 100, 100]))
            self.detector_label = kwargs.get("detector_label", "waste_item")
            self.detector_confidence = float(kwargs.get("detector_confidence", 1.0))
            self.waste_class = kwargs.get("waste_class", "trash")
            self.waste_confidence = float(kwargs.get("waste_confidence", 0.9))
            self.entropy = float(kwargs.get("entropy", 0.2))
            self.suggested_bin = kwargs.get("suggested_bin", "General Waste")
            self.bin_code = kwargs.get("bin_code", "TRASH")
            self.bin_guidance = kwargs.get("bin_guidance", "")
            self.is_drift = kwargs.get("is_drift", False)
            self.is_uncertain = kwargs.get("is_uncertain", False) or (self.waste_confidence < 0.65 and self.entropy > 0.80)
            self.display_label = kwargs.get("display_label", self.waste_class.title())

        self.first_seen = kwargs.get("first_seen_timestamp", time.time())
        self.last_seen = kwargs.get("last_seen_timestamp", time.time())
        self.hit_streak = kwargs.get("hit_streak", 1)
        self.disappeared_frames = kwargs.get("disappeared_frames", 0)
        self.crop_image = frame_crop if frame_crop is not None else kwargs.get("crop_image")

        self.is_reviewed = kwargs.get("is_reviewed", False)
        self.is_confirmed = kwargs.get("is_confirmed", False)
        self.user_label = kwargs.get("user_label", self.waste_class)
        self.is_rejected = kwargs.get("is_rejected", False)

    def update(
        self,
        detection: DetectedWasteObject,
        frame_crop: Optional[Image.Image] = None,
        alpha: float = 0.70
    ) -> None:
        """Smooth bounding box and update predictions with temporal filter."""
        # Exponential moving average for bounding box smoothing to eliminate jitter
        for i in range(4):
            self.box[i] = int(alpha * detection.box[i] + (1.0 - alpha) * self.box[i])

        self.detector_confidence = getattr(detection, "detector_confidence", getattr(self, "detector_confidence", 1.0))

        # Temporal stability filter: lock in the standardized label identity
        # and prevent flickering across consecutive video frames
        is_established = self.hit_streak >= 2 and self.waste_confidence >= 0.70
        is_higher_confidence = detection.waste_confidence > (self.waste_confidence + 0.12)
        if (not is_established) or is_higher_confidence:
            # Update or refine identity
            if not is_established or detection.waste_confidence >= 0.65:
                self.waste_class = detection.waste_class
                self.display_label = getattr(detection, "display_label", detection.waste_class.title())
                self.suggested_bin = detection.suggested_bin
                self.bin_code = detection.bin_code
                self.bin_guidance = detection.bin_guidance
            self.waste_confidence = max(self.waste_confidence, detection.waste_confidence)
            self.entropy = detection.entropy
            self.is_drift = detection.is_drift
            self.is_uncertain = getattr(detection, "is_uncertain", False)
            if frame_crop is not None:
                self.crop_image = frame_crop
        else:
            # Maintain stable confirmed label identity and gently smooth confidence
            self.waste_confidence = max(self.waste_confidence, detection.waste_confidence)
            if not getattr(self, "display_label", None):
                self.display_label = getattr(detection, "display_label", detection.waste_class.title())

        self.last_seen = time.time()
        self.hit_streak += 1
        self.disappeared_frames = 0


class StreamObjectTracker:
    """
    Tracks waste objects across video frames using IoU overlap.
    Deduplicates detections so the same object is not counted repeatedly.
    """

    def __init__(
        self,
        iou_threshold: float = 0.35,
        max_disappeared: int = 15,
        min_hits_to_confirm: int = 3
    ):
        self.iou_threshold = iou_threshold
        self.max_disappeared = max_disappeared
        self.min_hits_to_confirm = min_hits_to_confirm

        self.next_track_id = 1
        self.active_tracks: Dict[int, TrackedWasteObject] = {}
        self.session_all_tracks: Dict[int, TrackedWasteObject] = {}

    def update(
        self,
        detections: List[DetectedWasteObject],
        full_image: Optional[Image.Image] = None
    ) -> List[TrackedWasteObject]:
        """
        Update tracker with detections from current frame.
        Returns list of currently active tracked objects.
        """
        active_ids = list(self.active_tracks.keys())

        if not detections:
            # Mark all active tracks as disappeared
            for tid in active_ids:
                track = self.active_tracks[tid]
                track.disappeared_frames += 1
                if track.disappeared_frames > self.max_disappeared:
                    del self.active_tracks[tid]
            return []

        # Pairwise IoU matrix between active tracks and current detections
        matched_tracks = set()
        matched_detections = set()

        if active_ids:
            iou_matrix = np.zeros((len(active_ids), len(detections)), dtype=float)
            for r, tid in enumerate(active_ids):
                track_box = self.active_tracks[tid].box
                for c, det in enumerate(detections):
                    iou_matrix[r, c] = compute_iou(track_box, det.box)

            # Greedy bipartite matching by highest IoU
            while True:
                if iou_matrix.size == 0:
                    break
                max_val = np.max(iou_matrix)
                if max_val < self.iou_threshold:
                    break

                r, c = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                tid = active_ids[r]

                # Extract crop if full image provided
                crop = None
                if full_image is not None:
                    bx = detections[c].box
                    crop = full_image.crop((bx[0], bx[1], bx[2], bx[3]))

                self.active_tracks[tid].update(detections[c], frame_crop=crop)
                matched_tracks.add(tid)
                matched_detections.add(c)

                # Zero out row and column
                iou_matrix[r, :] = -1.0
                iou_matrix[:, c] = -1.0

        # Create new tracks for unmatched detections
        for c, det in enumerate(detections):
            if c not in matched_detections:
                crop = None
                if full_image is not None:
                    bx = det.box
                    crop = full_image.crop((bx[0], bx[1], bx[2], bx[3]))

                new_track = TrackedWasteObject(
                    track_id=self.next_track_id,
                    detection=det,
                    frame_crop=crop
                )
                self.active_tracks[self.next_track_id] = new_track
                self.session_all_tracks[self.next_track_id] = new_track
                self.next_track_id += 1

        # Increment disappeared for unmatched active tracks
        for tid in active_ids:
            if tid not in matched_tracks:
                self.active_tracks[tid].disappeared_frames += 1
                if self.active_tracks[tid].disappeared_frames > self.max_disappeared:
                    del self.active_tracks[tid]

        # Return strictly visible tracks matched in the current frame
        return [t for t in self.active_tracks.values() if t.disappeared_frames == 0]

    def get_session_unique_objects(self) -> List[TrackedWasteObject]:
        """Return all unique objects observed in the active session."""
        # Filter tracks that had at least min_hits_to_confirm to remove single-frame noise
        return [
            t for t in self.session_all_tracks.values()
            if t.hit_streak >= self.min_hits_to_confirm and not t.is_rejected
        ]

    def reset_session(self) -> None:
        """Clear session tracker state."""
        self.active_tracks.clear()
        self.session_all_tracks.clear()
        self.next_track_id = 1
