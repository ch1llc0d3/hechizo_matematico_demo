#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Sociological Profiling Demo — Image & Metadata "Vibe" Analysis
# -----------------------------------------------------------------
# Warning: This demonstrates the end of digital privacy.
# For educational / live-demo use only. Do not deploy for actual
# profiling or surveillance without legal and ethical review.
# -----------------------------------------------------------------

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# -----------------------------------------------------------------------------
# Lifestyle indicator mapping: raw object class -> provocative sociological label
# (COCO/YOLO class names; unmapped classes get a generic "Lifestyle Signal" label)
# -----------------------------------------------------------------------------

OBJECT_TO_VIBE: Dict[str, str] = {
    "person": "Presence / Body",
    "bicycle": "Eco-Conscious / Urban Mobility",
    "car": "Suburban / Commuter",
    "motorcycle": "Risk-Taker / Speed",
    "airplane": "Frequent Flyer / High Mobility",
    "bus": "Mass Transit / Budget-Conscious",
    "train": "Commuter / City Dweller",
    "truck": "Blue-Collar / Industrial",
    "boat": "Leisure / Coastal Affluence",
    "traffic light": "Urban Context",
    "fire hydrant": "Urban Context",
    "stop sign": "Rule-Follower",
    "parking meter": "Downtown / Paid Parking",
    "bench": "Public Space / Pause",
    "bird": "Nature Proximity",
    "cat": "Domestic Introvert",
    "dog": "Stable Domesticity",
    "horse": "Equestrian / Country",
    "sheep": "Rural / Pastoral",
    "cow": "Rural / Agricultural",
    "elephant": "Travel / Safari",
    "bear": "Outdoor / Wilderness",
    "zebra": "Exotic / Travel",
    "giraffe": "Exotic / Travel",
    "backpack": "Adventurer / Nomad",
    "umbrella": "Prepared / Urban",
    "handbag": "Fashion-Conscious",
    "tie": "Corporate / Formal",
    "suitcase": "Traveler / Transient",
    "frisbee": "Outdoor / Casual",
    "skis": "Winter Sports / Affluence",
    "snowboard": "Adrenaline / Youth",
    "sports ball": "Athletic / Team",
    "kite": "Leisure / Family",
    "baseball bat": "Sports / American",
    "baseball glove": "Sports / American",
    "skateboard": "Street Culture / Youth",
    "surfboard": "Coastal / Surf Culture",
    "tennis racket": "Country Club / Affluence",
    "bottle": "Social / Beverage",
    "wine glass": "Fine Living / Social",
    "cup": "Daily Ritual / Cafe",
    "fork": "Dining / Domestic",
    "knife": "Dining / Kitchen",
    "spoon": "Dining / Domestic",
    "bowl": "Home Cook / Domestic",
    "banana": "Health-Conscious",
    "apple": "Health-Conscious",
    "sandwich": "On-the-Go / Casual",
    "orange": "Health-Conscious",
    "broccoli": "Health-Conscious",
    "carrot": "Health-Conscious",
    "hot dog": "Street Food / Casual",
    "pizza": "Casual / Social",
    "donut": "Comfort / Indulgence",
    "cake": "Celebration / Social",
    "chair": "Workspace / Home",
    "couch": "Home / Relaxation",
    "potted plant": "Domestic / Curated",
    "bed": "Private Space",
    "dining table": "Gathering / Domestic",
    "toilet": "Private Space",
    "tv": "Entertainment / Home",
    "laptop": "High-Value Professional",
    "mouse": "Knowledge Worker",
    "remote": "Home Entertainment",
    "keyboard": "Knowledge Worker",
    "cell phone": "Connected / On-Call",
    "microwave": "Convenience / Domestic",
    "oven": "Home Cook",
    "toaster": "Domestic Ritual",
    "sink": "Domestic",
    "refrigerator": "Domestic / Kitchen",
    "book": "Reader / Intellectual",
    "clock": "Time-Aware",
    "vase": "Aesthetic / Curated",
    "scissors": "Craft / Office",
    "teddy bear": "Nostalgia / Comfort",
    "hair drier": "Grooming / Self-Care",
    "toothbrush": "Routine / Self-Care",
    "watch": "Status / Time-Keeper",
}

# High-income heuristic: known city centers / upscale areas (lat, lon approx)
# Demo: treat certain lat/lon bands as "high-income" for the +20 Watch rule.
HIGH_INCOME_LAT_RANGES: List[Tuple[float, float]] = [
    (40.7, 40.8),   # NYC Manhattan-ish
    (34.0, 34.1),   # LA Beverly Hills-ish
    (51.5, 51.52),  # London central
    (48.85, 48.86), # Paris central
]
HIGH_INCOME_LON_RANGES: List[Tuple[float, float]] = [
    (-74.0, -73.95),
    (-118.4, -118.3),
    (-0.15, -0.1),
    (2.32, 2.35),
]

# HUD colors (BGR for OpenCV)
COLOR_TECH = (255, 255, 0)      # Cyan
COLOR_LIFESTYLE = (255, 0, 255) # Purple
COLOR_SCAN = (0, 255, 255)      # Yellow
COLOR_WHITE = (255, 255, 255)


@dataclass
class ExifMetadata:
    """Extracted EXIF metadata from an image file."""
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    camera_model: Optional[str] = None
    timestamp: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_high_income_region(self) -> bool:
        if self.gps_lat is None or self.gps_lon is None:
            return False
        for (lo, hi) in HIGH_INCOME_LAT_RANGES:
            if lo <= self.gps_lat <= hi:
                for (lo2, hi2) in HIGH_INCOME_LON_RANGES:
                    if lo2 <= self.gps_lon <= hi2:
                        return True
        return False

    @property
    def is_premium_device(self) -> bool:
        if not self.camera_model:
            return False
        m = self.camera_model.lower()
        return "iphone" in m and ("pro" in m or "max" in m) or "pixel" in m or "galaxy s" in m


@dataclass
class VibeDetection:
    """Single detected object with sociological vibe label and score."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    raw_class: str
    vibe_label: str
    confidence: float
    vibe_score_delta: int  # points added to global Vibe_Score for this detection


def _exif_gps_to_decimal(coord: Tuple[Any, Any, Any], ref: str) -> float:
    """Convert EXIF rational GPS (deg, min, sec) to decimal degrees."""
    try:
        d = float(coord[0]) if not hasattr(coord[0], "num") else coord[0].num / max(coord[0].den, 1)
        m = float(coord[1]) if not hasattr(coord[1], "num") else coord[1].num / max(coord[1].den, 1)
        s = float(coord[2]) if not hasattr(coord[2], "num") else coord[2].num / max(coord[2].den, 1)
        dec = d + m / 60.0 + s / 3600.0
        if ref in ("S", "W"):
            dec = -dec
        return dec
    except (TypeError, ZeroDivisionError, IndexError):
        return 0.0


def extract_exif(image_path: str | Path) -> ExifMetadata:
    """
    EXIF Metadata Extractor. Exposes GPS, camera model, and timestamp
    to demonstrate how much the image alone reveals.
    """
    meta = ExifMetadata()
    path = Path(image_path)
    if not path.is_file():
        return meta
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
    except ImportError:
        return meta
    try:
        img = Image.open(path)
        exif = img.getexif()
        if not exif:
            return meta
        for tag_id, value in exif.items():
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8", errors="replace")
                except Exception:
                    value = "<binary>"
            tag_name = TAGS.get(tag_id, tag_id)
            meta.raw[tag_name] = value
            if tag_name == "Model":
                meta.camera_model = str(value) if value else None
            if tag_name == "DateTimeOriginal":
                meta.timestamp = str(value) if value else None
        # GPS (IFD 34853)
        gps_ifd = exif.get_ifd(0x8825)
        if gps_ifd:
            lat = gps_ifd.get(2)   # GPSLatitude
            lat_ref = gps_ifd.get(1, "N")
            lon = gps_ifd.get(4)   # GPSLongitude
            lon_ref = gps_ifd.get(3, "E")
            if lat and lon:
                meta.gps_lat = _exif_gps_to_decimal(tuple(lat), str(lat_ref))
                meta.gps_lon = _exif_gps_to_decimal(tuple(lon), str(lon_ref))
    except Exception:
        pass
    return meta


def _run_yolo(frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    """Run YOLO and return list of (bbox_xyxy, class_name, confidence)."""
    try:
        from ultralytics import YOLO
    except ImportError:
        return []
    model = getattr(_run_yolo, "_model", None)
    if model is None:
        _run_yolo._model = YOLO("yolov8n.pt")
        model = _run_yolo._model
    results = model(frame, verbose=False)[0]
    out: List[Tuple[Tuple[int, int, int, int], str, float]] = []
    if results.boxes is None:
        return out
    names = results.names or {}
    for box in results.boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        name = names.get(cls_id, "object")
        if isinstance(name, int):
            name = str(name)
        out.append(((x1, y1, x2, y2), name, conf))
    return out


def _vibe_score_delta(raw_class: str, exif: Optional[ExifMetadata]) -> int:
    """
    Cybersecurity/SRE edge: If GPS is high-income AND object is Watch, +20.
    Other objects get small deltas for demo flair.
    """
    delta = 5
    if raw_class == "watch":
        delta = 15
        if exif and exif.is_high_income_region:
            delta += 20
    if raw_class == "laptop":
        delta = 12
    if raw_class in ("dog", "cat"):
        delta = 8
    if raw_class == "backpack":
        delta = 10
    if raw_class == "cell phone":
        delta = 7
    return delta


def analyze_vibe(
    frame: np.ndarray,
    exif: Optional[ExifMetadata] = None,
    min_confidence: float = 0.25,
) -> Tuple[List[VibeDetection], int]:
    """
    Detect objects and map them to Lifestyle Indicators (vibes).
    Returns (list of VibeDetection, total Vibe_Score).
    """
    raw_detections = _run_yolo(frame)
    detections: List[VibeDetection] = []
    vibe_score = 0
    for (x1, y1, x2, y2), raw_class, conf in raw_detections:
        if conf < min_confidence:
            continue
        vibe_label = OBJECT_TO_VIBE.get(raw_class, "Lifestyle Signal")
        delta = _vibe_score_delta(raw_class, exif)
        vibe_score += delta
        detections.append(VibeDetection(
            bbox=(x1, y1, x2, y2),
            raw_class=raw_class,
            vibe_label=vibe_label,
            confidence=conf,
            vibe_score_delta=delta,
        ))
    return detections, vibe_score


def draw_hud(
    frame: np.ndarray,
    detections: List[VibeDetection],
    vibe_score: int,
    scan_angle: float,
    exif: Optional[ExifMetadata] = None,
) -> np.ndarray:
    """
    Overlay HUD: scanning animation, vibe bounding boxes, confidence bars,
    neon cyan (tech) / purple (lifestyle) styling.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Scanning line (radar-style)
    cx, cy = w // 2, h // 2
    r = min(w, h) // 2
    rad = math.radians(scan_angle)
    x2 = int(cx + r * math.cos(rad))
    y2 = int(cy - r * math.sin(rad))
    cv2.line(overlay, (cx, cy), (x2, y2), COLOR_SCAN, 2)
    cv2.circle(overlay, (cx, cy), r, COLOR_SCAN, 1)

    # "SCANNING..." text
    cv2.putText(
        overlay, "SCANNING...", (w // 2 - 80, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TECH, 2, cv2.LINE_AA
    )
    cv2.putText(
        overlay, f"VIBE_SCORE: {vibe_score}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_LIFESTYLE, 2, cv2.LINE_AA
    )

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = COLOR_TECH if "Professional" in det.vibe_label or "Tech" in det.vibe_label else COLOR_LIFESTYLE
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{det.vibe_label} | +{det.vibe_score_delta}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
        cv2.putText(overlay, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        # Confidence bar (horizontal, below box)
        bar_y = y2 + 4
        bar_w = max(60, int(120 * det.confidence))
        cv2.rectangle(overlay, (x1, bar_y), (x1 + 120, bar_y + 6), (60, 60, 60), -1)
        cv2.rectangle(overlay, (x1, bar_y), (x1 + bar_w, bar_y + 6), color, -1)
        cv2.putText(overlay, f"{det.confidence:.0%}", (x1 + 122, bar_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_WHITE, 1)

    if exif and (exif.gps_lat is not None or exif.camera_model or exif.timestamp):
        y_meta = h - 70
        cv2.putText(overlay, "METADATA:", (10, y_meta), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TECH, 1, cv2.LINE_AA)
        if exif.gps_lat is not None:
            cv2.putText(overlay, f"  GPS: {exif.gps_lat:.4f}, {exif.gps_lon:.4f}", (10, y_meta + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1)
        if exif.camera_model:
            cv2.putText(overlay, f"  Device: {exif.camera_model}", (10, y_meta + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1)
        if exif.timestamp:
            cv2.putText(overlay, f"  Time: {exif.timestamp}", (10, y_meta + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1)

    alpha = 0.85
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def run_webcam(
    camera_id: int = 0,
    inference_every_n: int = 15,
) -> None:
    """Run live webcam feed with scanning HUD and vibe analysis."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Could not open webcam.")
        return
    frame_count = 0
    scan_angle = 0.0
    last_detections: List[VibeDetection] = []
    last_score = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            scan_angle = (scan_angle + 4.0) % 360.0
            if frame_count % inference_every_n == 0:
                last_detections, last_score = analyze_vibe(frame, exif=None, min_confidence=0.35)
            out = draw_hud(frame, last_detections, last_score, scan_angle, exif=None)
            cv2.imshow("Sociological Profiler — Live", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_image(image_path: str | Path) -> None:
    """Run analysis on a single image (with EXIF) and display result."""
    path = Path(image_path)
    if not path.is_file():
        print(f"File not found: {path}")
        return
    exif = extract_exif(path)
    frame = cv2.imread(str(path))
    if frame is None:
        print("Could not load image.")
        return
    detections, vibe_score = analyze_vibe(frame, exif=exif, min_confidence=0.25)
    scan_angle = (time.time() * 100) % 360.0
    out = draw_hud(frame, detections, vibe_score, scan_angle, exif=exif)
    cv2.imshow("Sociological Profiler — Image", out)
    print(f"Vibe_Score: {vibe_score} | Detections: {len(detections)}")
    if exif.camera_model:
        print(f"Device: {exif.camera_model}")
    if exif.gps_lat is not None:
        print(f"GPS: {exif.gps_lat:.4f}, {exif.gps_lon:.4f}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Sociological Profiling Demo — Vibe & EXIF analysis")
    parser.add_argument("--image", type=str, help="Path to image file (enables EXIF extraction)")
    parser.add_argument("--webcam", action="store_true", help="Run on webcam")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device id")
    args = parser.parse_args()
    if args.image:
        run_image(args.image)
    else:
        run_webcam(camera_id=args.camera)


if __name__ == "__main__":
    main()
