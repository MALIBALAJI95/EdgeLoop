"""Multi-object detection and waste classification pipeline.

Combines lightweight SSDLite320 object detection with the preserved
MobileNetV2 waste classification champion to support multi-object scenes
and bin recommendations.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
import torch
import torchvision.models.detection as detection
from torchvision import transforms

from edgeloop.common.config import get_config
from edgeloop.drift.detector import DriftDetector
from edgeloop.drift.features import DriftFeatureExtractor
from edgeloop.inference.engine import InferenceEngine

logger = logging.getLogger("edgeloop.detector")

# COCO category labels for SSDLite
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table",
    "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "N/A", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Bin recommendations for the 6 waste categories
BIN_MAPPINGS = {
    "paper": {
        "bin_name": "Blue Bin (Paper & Books)",
        "bin_code": "PAPER_BLUE",
        "color_rgb": (30, 136, 229),
        "color_bgr": (229, 136, 30),
        "guidance": "Clean paper, books, notebooks, envelopes, and newsprint."
    },
    "cardboard": {
        "bin_name": "Brown/Blue Bin (Cardboard)",
        "bin_code": "CARDBOARD_BROWN",
        "color_rgb": (141, 110, 99),
        "color_bgr": (99, 110, 141),
        "guidance": "Flattened corrugated boxes, carton packaging, and stiff book covers."
    },
    "glass": {
        "bin_name": "Green Bin (Glass)",
        "bin_code": "GLASS_GREEN",
        "color_rgb": (67, 160, 71),
        "color_bgr": (71, 160, 67),
        "guidance": "Glass bottles, jars, and beverage glassware (rinse first)."
    },
    "plastic": {
        "bin_name": "Yellow Bin (Plastics)",
        "bin_code": "PLASTIC_YELLOW",
        "color_rgb": (253, 216, 53),
        "color_bgr": (53, 216, 253),
        "guidance": "Plastic bottles, drink cups, jugs, and clean rigid plastic containers."
    },
    "metal": {
        "bin_name": "Grey/Yellow Bin (Metals)",
        "bin_code": "METAL_GREY",
        "color_rgb": (120, 144, 156),
        "color_bgr": (156, 144, 120),
        "guidance": "Aluminum beverage cans, tin food cans, and clean metal foil."
    },
    "trash": {
        "bin_name": "Black Bin (General Refuse)",
        "bin_code": "TRASH_BLACK",
        "color_rgb": (220, 53, 69),
        "color_bgr": (69, 53, 220),
        "guidance": "Composite packaging, soiled items, and non-recyclable solid waste."
    }
}


class DetectedWasteObject(BaseModel):
    """Detailed record of a single localized waste object."""
    box: List[int] = Field(..., description="[x1, y1, x2, y2] in pixels")
    detector_label: str
    detector_confidence: float
    waste_class: str
    waste_confidence: float
    entropy: float
    suggested_bin: str
    bin_code: str
    bin_guidance: str
    is_drift: bool = False
    triggered_features: List[str] = Field(default_factory=list)


class DetectionFrameResult(BaseModel):
    """Result of multi-object detection and waste classification on a video frame."""
    objects: List[DetectedWasteObject]
    total_objects: int
    dominant_class: Optional[str] = None
    dominant_bin: Optional[str] = None
    frame_drift: bool = False
    frame_drift_features: List[str] = Field(default_factory=list)
    processing_time_ms: float


class WasteObjectDetector:
    """
    Two-stage detector:
    Stage 1: Pretrained SSDLite320 proposes candidate object bounding boxes.
    Stage 2: Preserved MobileNetV2 champion classifies each crop into waste categories.
    """

    def __init__(
        self,
        classifier_engine: Optional[InferenceEngine] = None,
        detection_threshold: float = 0.25,
        device: str = "auto"
    ):
        self.config = get_config()
        self.detection_threshold = detection_threshold

        # Classifier engine
        self.classifier = classifier_engine or InferenceEngine(device=device, log_predictions=False)
        self.device = self.classifier.device

        # Drift detector
        self.drift_detector = DriftDetector()
        self.feature_extractor = DriftFeatureExtractor()

        # Load SSDLite object detector
        logger.info("Initializing SSDLite320 object detector on %s...", self.device)
        self.detector_model = detection.ssdlite320_mobilenet_v3_large(
            weights=detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        )
        self.detector_model.to(self.device)
        self.detector_model.eval()

        self.det_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def detect_and_classify(
        self,
        image_input: Union[Image.Image, np.ndarray],
        detection_threshold: Optional[float] = None
    ) -> DetectionFrameResult:
        """Execute multi-object localization and waste category classification."""
        import time
        start_time = time.perf_counter()
        thresh = detection_threshold if detection_threshold is not None else self.detection_threshold

        if isinstance(image_input, np.ndarray):
            # If BGR from OpenCV, convert to RGB PIL
            if image_input.ndim == 3 and image_input.shape[2] == 3:
                rgb_arr = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_arr)
            else:
                pil_img = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        w, h = pil_img.size

        # 1. Whole frame drift check
        frame_feats = self.feature_extractor.extract_from_image(pil_img)
        frame_drift, frame_trig = self.drift_detector.check_sample(frame_feats, log_to_db=False)

        # 2. Run SSDLite object detector
        det_tensor = self.det_transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            predictions = self.detector_model(det_tensor)[0]

        boxes = predictions["boxes"].detach().cpu().numpy()
        scores = predictions["scores"].detach().cpu().numpy()
        labels = predictions["labels"].detach().cpu().numpy()

        detected_objects: List[DetectedWasteObject] = []

        # Filter candidate object proposals
        for i in range(len(scores)):
            score = float(scores[i])
            if score < thresh:
                continue

            box = boxes[i].astype(int)
            x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
            box_w, box_h = x2 - x1, y2 - y1

            # Ignore tiny boxes
            if box_w < 25 or box_h < 25:
                continue

            coco_idx = int(labels[i])
            coco_name = COCO_CLASSES[coco_idx] if coco_idx < len(COCO_CLASSES) else "object"

            # Crop region
            crop_img = pil_img.crop((x1, y1, x2, y2))

            # Run MobileNetV2 waste classification on crop
            pred_res = self.classifier.predict_image(crop_img, sample_id=f"crop_{x1}_{y1}")
            bin_info = BIN_MAPPINGS.get(pred_res.predicted_class, BIN_MAPPINGS["trash"])

            # Crop-level drift check
            crop_feats = self.feature_extractor.extract_from_image(crop_img)
            crop_drift, crop_trig = self.drift_detector.check_sample(crop_feats, log_to_db=False)

            detected_objects.append(
                DetectedWasteObject(
                    box=[int(x1), int(y1), int(x2), int(y2)],
                    detector_label=coco_name,
                    detector_confidence=score,
                    waste_class=pred_res.predicted_class,
                    waste_confidence=pred_res.confidence,
                    entropy=pred_res.entropy,
                    suggested_bin=bin_info["bin_name"],
                    bin_code=bin_info["bin_code"],
                    bin_guidance=bin_info["guidance"],
                    is_drift=crop_drift,
                    triggered_features=crop_trig
                )
            )

        # 3. Fallback: If no candidate objects were found by COCO detector
        # (e.g. user holds up a book or object close-up filling the frame)
        if not detected_objects:
            pred_res = self.classifier.predict_image(pil_img, sample_id="full_frame")
            bin_info = BIN_MAPPINGS.get(pred_res.predicted_class, BIN_MAPPINGS["trash"])

            # Propose central 85% area box
            margin_x = int(w * 0.07)
            margin_y = int(h * 0.07)
            detected_objects.append(
                DetectedWasteObject(
                    box=[margin_x, margin_y, w - margin_x, h - margin_y],
                    detector_label="waste_item",
                    detector_confidence=1.0,
                    waste_class=pred_res.predicted_class,
                    waste_confidence=pred_res.confidence,
                    entropy=pred_res.entropy,
                    suggested_bin=bin_info["bin_name"],
                    bin_code=bin_info["bin_code"],
                    bin_guidance=bin_info["guidance"],
                    is_drift=frame_drift,
                    triggered_features=frame_trig
                )
            )

        # Sort detected objects by confidence descending
        detected_objects.sort(key=lambda o: -o.waste_confidence)

        dominant = detected_objects[0]
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return DetectionFrameResult(
            objects=detected_objects,
            total_objects=len(detected_objects),
            dominant_class=dominant.waste_class,
            dominant_bin=dominant.suggested_bin,
            frame_drift=frame_drift,
            frame_drift_features=frame_trig,
            processing_time_ms=elapsed_ms
        )

    def draw_overlays(
        self,
        image_input: Union[Image.Image, np.ndarray],
        detection_result: DetectionFrameResult
    ) -> np.ndarray:
        """
        Draw high-visibility bounding boxes, class pills, confidence badges,
        and bin recommendations onto the image for UI display.
        """
        if isinstance(image_input, Image.Image):
            # Convert PIL RGB to OpenCV BGR
            frame = cv2.cvtColor(np.array(image_input.convert("RGB")), cv2.COLOR_RGB2BGR)
        else:
            frame = image_input.copy()

        h, w, _ = frame.shape

        for obj in detection_result.objects:
            x1, y1, x2, y2 = obj.box
            bin_info = BIN_MAPPINGS.get(obj.waste_class, BIN_MAPPINGS["trash"])
            bgr_color = bin_info["color_bgr"]

            # Draw outer glow and rounded bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, 3, cv2.LINE_AA)

            # Build label text: e.g. "PLASTIC 94% -> Yellow Bin"
            label_text = f"{obj.waste_class.upper()} {obj.waste_confidence * 100:.0f}% -> {obj.suggested_bin.split(' ')[0]} Bin"
            sub_label = f"Obj: {obj.detector_label}"
            if obj.is_drift:
                sub_label += " | DRIFT"

            # Compute label size
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.55
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label_text, font, scale, thickness + 1)

            # Pill background coordinates
            pill_y1 = max(0, y1 - th - baseline - 14)
            pill_y2 = max(th + 14, y1)
            pill_x2 = min(w, x1 + tw + 18)

            # Draw pill background
            cv2.rectangle(frame, (x1, pill_y1), (pill_x2, pill_y2), bgr_color, -1)

            # Draw label text
            text_color = (0, 0, 0) if obj.waste_class in ["plastic"] else (255, 255, 255)
            cv2.putText(
                frame,
                label_text,
                (x1 + 8, pill_y1 + th + 6),
                font,
                scale,
                text_color,
                thickness + 1,
                cv2.LINE_AA
            )

        # Convert back to RGB for Streamlit display
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
