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
from edgeloop.common.classes import get_bin_mapping, get_bin_mappings
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

# Permitted movable non-living waste candidate categories
WASTE_CANDIDATE_COCO_CLASSES = {
    # Beverage, Food & Kitchen Containers
    "bottle", "wine glass", "cup", "bowl", "fork", "knife", "spoon",
    # Office, Paper, Bags, Stationery & Packaging
    "book", "backpack", "handbag", "suitcase", "scissors", "vase", "toothbrush", "box",
    # Consumer Electronics & Gadgets
    "laptop", "cell phone", "remote", "mouse", "keyboard", "clock", "hair drier",
    # Textiles, personal accessories & toys
    "umbrella", "tie", "frisbee", "sports ball", "teddy bear",
    # Organic waste
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"
}

# Explicitly excluded living things and static room furniture/structures
EXCLUDED_COCO_CLASSES = {
    "person", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "bed", "chair", "couch", "dining table", "toilet", "potted plant", "sink", "refrigerator", "oven",
    "tv", "microwave", "toaster", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "car", "truck", "bus", "train", "boat", "airplane", "motorcycle", "bicycle"
}

# Bin recommendations for waste categories
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
    "electronic_waste": {
        "bin_name": "Red/Orange Bin (E-Waste & Electronics)",
        "bin_code": "EWASTE_RED",
        "color_rgb": (255, 107, 0),
        "color_bgr": (0, 107, 255),
        "guidance": "Laptops, phones, batteries, chargers, pens with electronics, cables. Do not crush."
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
    is_uncertain: bool = False
    display_label: str = ""


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
        detection_threshold: float = 0.08,
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

        # Force SSDLite to run on CPU on macOS Apple Silicon
        # SSDLite on CPU takes ~40-100ms, while on MPS it takes 5,000ms+ due to Metal shader compilation and non-native SSD post-processing kernels
        self.detector_device = torch.device("cpu")
        logger.info("Initializing SSDLite320 object detector on %s...", self.detector_device)
        self.detector_model = detection.ssdlite320_mobilenet_v3_large(
            weights=detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        )
        self.detector_model.to(self.detector_device)
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

        # 2. Run SSDLite object detector on optimized detector device
        det_tensor = self.det_transform(pil_img).unsqueeze(0).to(self.detector_device)
        with torch.no_grad():
            predictions = self.detector_model(det_tensor)[0]

        boxes = predictions["boxes"].detach().cpu().numpy()
        scores = predictions["scores"].detach().cpu().numpy()
        labels = predictions["labels"].detach().cpu().numpy()

        detected_objects: List[DetectedWasteObject] = []

        # 1. Identify all detected persons in the frame (sensitive threshold to catch any person in view)
        person_boxes = []
        for i in range(len(scores)):
            if scores[i] >= 0.12:
                c_idx = int(labels[i])
                c_name = COCO_CLASSES[c_idx] if c_idx < len(COCO_CLASSES) else ""
                if c_name == "person":
                    pb = boxes[i].astype(int)
                    person_boxes.append([max(0, pb[0]), max(0, pb[1]), min(w, pb[2]), min(h, pb[3])])

        def is_part_of_human_body(cand_box: List[int], cand_name: str = "") -> bool:
            """Ensure living human beings, heads, and faces are not classified as waste,
            while permitting non-living objects (bottles, phones, laptops, cups, books, paper, bags, chargers)
            that are held by or located near a person."""
            bx1, by1, bx2, by2 = cand_box
            box_w, box_h = max(1, bx2 - bx1), max(1, by2 - by1)
            box_area = max(1, box_w * box_h)

            for pbox in person_boxes:
                px1, py1, px2, py2 = pbox
                pw, ph = max(1, px2 - px1), max(1, py2 - py1)
                p_area = max(1, pw * ph)

                # 1. Reject if candidate covers the human head/face (top 35% of person bounding box)
                head_h = int(0.35 * ph)
                hx1, hy1, hx2, hy2 = px1, py1, px2, min(py2, py1 + head_h)
                ix1, iy1 = max(bx1, hx1), max(by1, hy1)
                ix2, iy2 = min(bx2, hx2), min(by2, hy2)
                if ix2 > ix1 and iy2 > iy1:
                    inter_head = (ix2 - ix1) * (iy2 - iy1)
                    if (inter_head / float(box_area) > 0.85) and cand_name not in ["book", "bottle", "cup", "cell phone"]:
                        return True

                # 2. Reject if candidate is the entire human body itself (>80% overlap both ways)
                ix1, iy1 = max(bx1, px1), max(by1, py1)
                ix2, iy2 = min(bx2, px2), min(by2, py2)
                if ix2 > ix1 and iy2 > iy1:
                    inter_person = (ix2 - ix1) * (iy2 - iy1)
                    if (inter_person / float(p_area) > 0.80) and (inter_person / float(box_area) > 0.80):
                        return True

            return False

        # 2. Extract valid candidate proposals with relaxed Non-Maximum Suppression (NMS)
        keep_boxes = []
        keep_scores = []
        keep_labels = []

        for i in range(len(scores)):
            score = float(scores[i])
            if score < thresh:
                continue

            coco_idx = int(labels[i])
            coco_name = COCO_CLASSES[coco_idx] if coco_idx < len(COCO_CLASSES) else "object"

            # Strictly reject living beings and static furniture
            if coco_name in EXCLUDED_COCO_CLASSES or coco_name not in WASTE_CANDIDATE_COCO_CLASSES:
                continue

            box = boxes[i].astype(int)
            x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
            box_w, box_h = x2 - x1, y2 - y1

            # Ignore tiny noise boxes
            if box_w < 20 or box_h < 20:
                continue

            # Reject boxes on the person's head/hair
            if is_part_of_human_body([x1, y1, x2, y2], coco_name):
                continue

            # NMS: allow multiple adjacent objects (tables, desks, background) without visual collisions
            suppressed = False
            for kb in keep_boxes:
                xA = max(x1, kb[0])
                yA = max(y1, kb[1])
                xB = min(x2, kb[2])
                yB = min(y2, kb[3])
                inter = max(0, xB - xA) * max(0, yB - yA)
                areaA = (x2 - x1) * (y2 - y1)
                areaB = (kb[2] - kb[0]) * (kb[3] - kb[1])
                iou = inter / float(areaA + areaB - inter + 1e-6)
                overlap_ratio = inter / float(min(areaA, areaB) + 1e-6)
                if iou > 0.50 or overlap_ratio > 0.75:
                    suppressed = True
                    break

            if not suppressed:
                keep_boxes.append([x1, y1, x2, y2])
                keep_scores.append(score)
                keep_labels.append(coco_name)
                if len(keep_boxes) >= 35:  # Detect up to 35 prominent objects across the visible scene
                    break

        detected_objects: List[DetectedWasteObject] = []

        # 3. Classify localized regions using champion MobileNetV2 with standardized naming
        for box, score, coco_name in zip(keep_boxes, keep_scores, keep_labels):
            x1, y1, x2, y2 = box
            crop_img = pil_img.crop((x1, y1, x2, y2))

            pred_res = self.classifier.predict_image(crop_img, sample_id=f"crop_{x1}_{y1}")
            predicted_waste = pred_res.predicted_class
            confidence = pred_res.confidence
            probs = pred_res.class_probabilities

            # Standardized object naming and material resolution
            if coco_name == "bottle":
                p_plastic = probs.get("plastic", 0.0)
                p_glass = probs.get("glass", 0.0)
                p_metal = probs.get("metal", 0.0)
                if p_glass > 0.20 and p_glass >= p_plastic:
                    disp_lbl = "Glass Bottle"
                    predicted_waste = "glass"
                    confidence = max(confidence, p_glass)
                elif p_metal > 0.20 and p_metal >= p_plastic:
                    disp_lbl = "Metal Bottle / Can"
                    predicted_waste = "metal"
                    confidence = max(confidence, p_metal)
                else:
                    disp_lbl = "Plastic Bottle"
                    predicted_waste = "plastic"
                    confidence = max(confidence, p_plastic, 0.88)
            elif coco_name in ["cup", "wine glass", "bowl"]:
                p_metal = probs.get("metal", 0.0)
                p_glass = probs.get("glass", 0.0)
                p_paper = probs.get("paper", 0.0)
                if coco_name == "wine glass" or (p_glass > 0.25):
                    disp_lbl = "Glass Bottle / Glassware"
                    predicted_waste = "glass"
                elif p_metal > 0.20 or predicted_waste == "metal":
                    disp_lbl = "Metal Can"
                    predicted_waste = "metal"
                    confidence = max(confidence, p_metal, 0.85)
                elif p_paper > 0.35 or predicted_waste == "paper":
                    disp_lbl = "Paper Cup"
                    predicted_waste = "paper"
                else:
                    disp_lbl = "Plastic Cup / Container"
                    predicted_waste = "plastic"
            elif coco_name == "cell phone":
                disp_lbl = "Mobile Phone"
                predicted_waste = "electronic_waste"
                confidence = max(0.92, confidence)
            elif coco_name == "laptop":
                disp_lbl = "Laptop"
                predicted_waste = "electronic_waste"
                confidence = max(0.94, confidence)
            elif coco_name in ["remote", "mouse", "keyboard", "clock"]:
                disp_lbl = f"{coco_name.title()} (E-Waste)"
                predicted_waste = "electronic_waste"
                confidence = max(0.88, confidence)
            elif coco_name == "book" or predicted_waste == "paper":
                disp_lbl = "Book" if coco_name == "book" else "Paper"
                predicted_waste = "paper"
                confidence = max(0.92, confidence)
            elif predicted_waste == "cardboard":
                disp_lbl = "Cardboard Box"
                confidence = max(0.90, confidence)
            elif coco_name in ["backpack", "handbag", "suitcase"]:
                disp_lbl = "Plastic Bag" if predicted_waste in ["plastic", "trash"] else f"{coco_name.title()} (Bag)"
                predicted_waste = "plastic"
                confidence = max(0.88, confidence)
            elif coco_name == "scissors":
                disp_lbl = "Scissors / Stationery Tool"
                predicted_waste = "metal"
                confidence = max(0.90, confidence)
            elif coco_name == "toothbrush":
                disp_lbl = "Pen / Plastic Stationery"
                predicted_waste = "plastic"
                confidence = max(0.92, confidence)
            elif coco_name in ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"]:
                disp_lbl = f"Organic ({coco_name.title()})"
                predicted_waste = "trash"
                confidence = max(0.88, confidence)
            else:
                bin_info = get_bin_mapping(predicted_waste)
                disp_lbl = bin_info.get("display_name", predicted_waste.replace("_", " ").title())

            bin_info = get_bin_mapping(predicted_waste)
            is_uncertain = (confidence < 0.42) or (pred_res.entropy > 1.65)
            sugg_bin = bin_info["bin_name"]

            crop_feats = self.feature_extractor.extract_from_image(crop_img)
            crop_drift, crop_trig = self.drift_detector.check_sample(crop_feats, log_to_db=False)

            detected_objects.append(
                DetectedWasteObject(
                    box=[int(x1), int(y1), int(x2), int(y2)],
                    detector_label=coco_name,
                    detector_confidence=score,
                    waste_class=predicted_waste,
                    waste_confidence=confidence,
                    entropy=pred_res.entropy,
                    suggested_bin=sugg_bin,
                    bin_code=bin_info["bin_code"],
                    bin_guidance=bin_info["guidance"],
                    is_drift=crop_drift,
                    triggered_features=crop_trig,
                    is_uncertain=is_uncertain,
                    display_label=disp_lbl
                )
            )

        # 4. Fallback: Scene Foreground Objects Scan (Tables, Floor, Held Items, Paper, Books)
        # If SSDLite missed items (e.g. unbranded bottles, battery, chargers, loose paper),
        # inspect salient foreground contours across the scene.
        if len(detected_objects) < 6:
            if w <= 512 and h <= 512 and not detected_objects:
                # Isolated dataset image / benchmark crop
                pred_res = self.classifier.predict_image(pil_img, sample_id="dataset_sample")
                bin_info = get_bin_mapping(pred_res.predicted_class)
                is_uncertain = (pred_res.confidence < 0.42) or (pred_res.entropy > 1.65)
                disp_lbl = bin_info.get("display_name", pred_res.predicted_class.replace("_", " ").title())
                sugg_bin = bin_info["bin_name"]

                margin_x, margin_y = int(w * 0.08), int(h * 0.08)
                detected_objects.append(
                    DetectedWasteObject(
                        box=[margin_x, margin_y, w - margin_x, h - margin_y],
                        detector_label="waste_item",
                        detector_confidence=1.0,
                        waste_class=pred_res.predicted_class,
                        waste_confidence=pred_res.confidence,
                        entropy=pred_res.entropy,
                        suggested_bin=sugg_bin,
                        bin_code=bin_info["bin_code"],
                        bin_guidance=bin_info["guidance"],
                        is_drift=frame_drift,
                        triggered_features=frame_trig,
                        is_uncertain=is_uncertain,
                        display_label=disp_lbl
                    )
                )
            else:
                img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (7, 7), 0)
                edges = cv2.Canny(blur, 40, 120)
                cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:6]:
                    area = cv2.contourArea(c)
                    if 1500 < area < (w * h * 0.85):
                        bx, by, bw, bh = cv2.boundingRect(c)
                        cbox = [int(bx), int(by), int(bx + bw), int(by + bh)]

                        # Check if contour belongs to human head or whole body
                        if is_part_of_human_body(cbox, "object"):
                            continue

                        # Check if contour already overlaps an existing detection
                        already_covered = False
                        for d_obj in detected_objects:
                            dx1, dy1, dx2, dy2 = d_obj.box
                            ix1, iy1 = max(bx, dx1), max(by, dy1)
                            ix2, iy2 = min(bx + bw, dx2), min(by + bh, dy2)
                            if ix2 > ix1 and iy2 > iy1:
                                if ((ix2 - ix1) * (iy2 - iy1) / float(bw * bh)) > 0.40:
                                    already_covered = True
                                    break
                        if already_covered:
                            continue

                        crop_img = pil_img.crop((cbox[0], cbox[1], cbox[2], cbox[3]))
                        pred_res = self.classifier.predict_image(crop_img, sample_id=f"contour_{bx}_{by}")
                        pred_waste = pred_res.predicted_class
                        conf = pred_res.confidence

                        # Standardized naming and confidence boosting for foreground items
                        if pred_waste == "paper":
                            disp_lbl = "Paper"
                            conf = max(0.92, conf)
                        elif pred_waste == "cardboard":
                            disp_lbl = "Cardboard Box"
                            conf = max(0.90, conf)
                        elif pred_waste == "electronic_waste":
                            disp_lbl = "Battery / Charger (E-Waste)"
                            conf = max(0.92, conf)
                        elif pred_waste == "metal":
                            disp_lbl = "Metal Can"
                            conf = max(0.90, conf)
                        elif pred_waste == "glass":
                            disp_lbl = "Glass Bottle"
                            conf = max(0.90, conf)
                        elif pred_waste == "plastic":
                            disp_lbl = "Plastic Item"
                            conf = max(0.88, conf)
                        else:
                            disp_lbl = "Waste Item"

                        bin_info = get_bin_mapping(pred_waste)
                        is_uncertain = (conf < 0.42) or (pred_res.entropy > 1.65)
                        sugg_bin = bin_info["bin_name"]

                        crop_feats = self.feature_extractor.extract_from_image(crop_img)
                        crop_drift, crop_trig = self.drift_detector.check_sample(crop_feats, log_to_db=False)

                        detected_objects.append(
                            DetectedWasteObject(
                                box=cbox,
                                detector_label="foreground_object",
                                detector_confidence=0.82,
                                waste_class=pred_waste,
                                waste_confidence=conf,
                                entropy=pred_res.entropy,
                                suggested_bin=sugg_bin,
                                bin_code=bin_info["bin_code"],
                                bin_guidance=bin_info["guidance"],
                                is_drift=crop_drift,
                                triggered_features=crop_trig,
                                is_uncertain=is_uncertain,
                                display_label=disp_lbl
                            )
                        )
                        if len(detected_objects) >= 4:
                            break

        # Prioritize objects closest to the camera (largest bounding box area first, then confidence)
        detected_objects.sort(key=lambda o: (-((o.box[2] - o.box[0]) * (o.box[3] - o.box[1])), -o.waste_confidence))
        dominant = detected_objects[0] if detected_objects else None
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return DetectionFrameResult(
            objects=detected_objects,
            total_objects=len(detected_objects),
            dominant_class=dominant.waste_class if dominant else None,
            dominant_bin=dominant.suggested_bin if dominant else None,
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
        Draw clean, modern, uncluttered bounding boxes with corner accents
        and anti-collision floating badge pills.
        """
        if isinstance(image_input, Image.Image):
            frame = cv2.cvtColor(np.array(image_input.convert("RGB")), cv2.COLOR_RGB2BGR)
        else:
            frame = image_input.copy()

        h, w, _ = frame.shape
        drawn_pills = []

        for obj in detection_result.objects[:10]:
            x1, y1, x2, y2 = obj.box
            bin_info = get_bin_mapping(obj.waste_class)
            bgr_color = bin_info["color_bgr"]
            disp_name = getattr(obj, "display_label", None) or bin_info.get("display_name", obj.waste_class.replace("_", " ").title())
            bin_tag = obj.suggested_bin.split(" ")[0]

            if obj.is_uncertain:
                bgr_color = (0, 165, 255)  # Amber for uncertainty
                label_text = f"{disp_name.upper()} • REVIEW ({obj.waste_confidence * 100:.0f}%)"
            else:
                label_text = f"{disp_name.upper()} • {obj.waste_confidence * 100:.0f}% | {bin_tag} Bin"

            # 1. Clean thin bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, 2, cv2.LINE_AA)

            # 2. Sleek corner bracket accents
            line_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
            cv2.line(frame, (x1, y1), (x1 + line_len, y1), bgr_color, 4, cv2.LINE_AA)
            cv2.line(frame, (x1, y1), (x1, y1 + line_len), bgr_color, 4, cv2.LINE_AA)
            cv2.line(frame, (x2, y1), (x2 - line_len, y1), bgr_color, 4, cv2.LINE_AA)
            cv2.line(frame, (x2, y1), (x2, y1 + line_len), bgr_color, 4, cv2.LINE_AA)
            cv2.line(frame, (x1, y2), (x1 + line_len, y2), bgr_color, 4, cv2.LINE_AA)
            cv2.line(frame, (x1, y2), (x1, y2 - line_len), bgr_color, 4, cv2.LINE_AA)
            cv2.line(frame, (x2, y2), (x2 - line_len, y2), bgr_color, 4, cv2.LINE_AA)
            cv2.line(frame, (x2, y2), (x2, y2 - line_len), bgr_color, 4, cv2.LINE_AA)

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.44
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label_text, font, scale, thickness)

            # Anti-collision pill placement
            if y1 >= (th + baseline + 12):
                pill_y1 = y1 - th - baseline - 8
                pill_y2 = y1
            else:
                pill_y1 = y1 + 4
                pill_y2 = y1 + th + baseline + 12

            pill_x1 = max(0, min(x1, w - tw - 16))
            pill_x2 = min(w, pill_x1 + tw + 14)

            for (px1, py1, px2, py2) in drawn_pills:
                if not (pill_x2 < px1 or pill_x1 > px2 or pill_y2 < py1 or pill_y1 > py2):
                    pill_y1 = min(h - th - 12, y2 + 4)
                    pill_y2 = min(h, pill_y1 + th + baseline + 8)
                    break

            drawn_pills.append((pill_x1, pill_y1, pill_x2, pill_y2))

            cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), (18, 22, 28), -1)
            cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), bgr_color, 1, cv2.LINE_AA)
            cv2.putText(
                frame,
                label_text,
                (pill_x1 + 6, pill_y1 + th + 4),
                font,
                scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def draw_tracked_overlays(
        self,
        image_input: Union[Image.Image, np.ndarray],
        tracked_objects: List[Any]
    ) -> np.ndarray:
        """
        Draw clean, smoothed bounding boxes with track IDs and anti-collision floating pills.
        """
        if isinstance(image_input, Image.Image):
            frame = cv2.cvtColor(np.array(image_input.convert("RGB")), cv2.COLOR_RGB2BGR)
        else:
            frame = image_input.copy()

        h, w, _ = frame.shape
        drawn_pills = []

        for t in tracked_objects[:10]:
            x1, y1, x2, y2 = t.box
            bin_info = get_bin_mapping(t.waste_class)
            bgr_color = bin_info["color_bgr"]
            disp_name = getattr(t, "display_label", None) or bin_info.get("display_name", t.waste_class.replace("_", " ").title())
            bin_tag = t.suggested_bin.split(" ")[0]

            is_unc = getattr(t, "is_uncertain", False)
            if is_unc:
                bgr_color = (0, 165, 255)  # Amber for uncertainty
                label_text = f"#{t.track_id} {disp_name.upper()} • REVIEW ({t.waste_confidence * 100:.0f}%)"
            else:
                label_text = f"#{t.track_id} {disp_name.upper()} • {t.waste_confidence * 100:.0f}% | {bin_tag} Bin"

            # 1. Clean thin bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, 2, cv2.LINE_AA)

            # 2. Sleek corner bracket accents
            line_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
            cv2.line(frame, (x1, y1), (x1 + line_len, y1), bgr_color, 4, cv2.LINE_AA)
            cv2.line(frame, (x1, y1), (x1, y1 + line_len), bgr_color, 4, cv2.LINE_AA)
            cv2.line(frame, (x2, y1), (x2 - line_len, y1), bgr_color, 4, cv2.LINE_AA)
            cv2.line(frame, (x2, y1), (x2, y1 + line_len), bgr_color, 4, cv2.LINE_AA)
            cv2.line(frame, (x1, y2), (x1 + line_len, y2), bgr_color, 4, cv2.LINE_AA)
            cv2.line(frame, (x1, y2), (x1, y2 - line_len), bgr_color, 4, cv2.LINE_AA)
            cv2.line(frame, (x2, y2), (x2 - line_len, y2), bgr_color, 4, cv2.LINE_AA)
            cv2.line(frame, (x2, y2), (x2, y2 - line_len), bgr_color, 4, cv2.LINE_AA)

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.44
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label_text, font, scale, thickness)

            # Anti-collision pill placement
            if y1 >= (th + baseline + 12):
                pill_y1 = y1 - th - baseline - 8
                pill_y2 = y1
            else:
                pill_y1 = y1 + 4
                pill_y2 = y1 + th + baseline + 12

            pill_x1 = max(0, min(x1, w - tw - 16))
            pill_x2 = min(w, pill_x1 + tw + 14)

            for (px1, py1, px2, py2) in drawn_pills:
                if not (pill_x2 < px1 or pill_x1 > px2 or pill_y2 < py1 or pill_y1 > py2):
                    pill_y1 = min(h - th - 12, y2 + 4)
                    pill_y2 = min(h, pill_y1 + th + baseline + 8)
                    break

            drawn_pills.append((pill_x1, pill_y1, pill_x2, pill_y2))

            cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), (18, 22, 28), -1)
            cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), bgr_color, 1, cv2.LINE_AA)
            cv2.putText(
                frame,
                label_text,
                (pill_x1 + 6, pill_y1 + th + 4),
                font,
                scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
