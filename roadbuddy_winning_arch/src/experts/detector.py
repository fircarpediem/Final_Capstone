"""
Traffic Object Detector
YOLOv10 based detector fine-tuned for Vietnamese traffic signs
"""

import torch
import numpy as np
import cv2
from typing import List, Dict, Optional
from ultralytics import YOLO
from loguru import logger


class TrafficDetector:
    """
    Object detection for traffic signs, signals, and vehicles
    Uses YOLOv10 fine-tuned on Vietnamese traffic dataset
    """
    
    def __init__(self, config):
        self.config = config
        self.model_path = config.model.experts.detector.weights
        self.conf_threshold = config.model.experts.detector.conf_threshold
        self.iou_threshold = config.model.experts.detector.iou_threshold
        self.class_names = config.model.experts.detector.classes
        
        # Load model
        logger.info(f"Loading detector: {self.model_path}")
        self.model = YOLO(self.model_path)
        
        logger.info(f"Detector initialized with {len(self.class_names)} classes")
    
    def detect(self, frames: List[np.ndarray]) -> List[Dict]:
        """
        Run detection on multiple frames
        
        Args:
            frames: List of frames (H, W, C) in RGB
            
        Returns:
            List of detections with tracking across frames
        """
        all_detections = []
        
        for frame_idx, frame in enumerate(frames):
            # Run YOLO inference
            results = self.model(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            # Parse results
            for det in results[0].boxes:
                x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
                
                detection = {
                    "frame_idx": frame_idx,
                    "class_id": int(det.cls),
                    "class_name": self.class_names[int(det.cls)],
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(det.conf),
                    "crop": frame[int(y1):int(y2), int(x1):int(x2)].copy()
                }
                
                all_detections.append(detection)
        
        # Track objects across frames
        tracked_objects = self._track_across_frames(all_detections)
        
        logger.debug(f"Detected {len(all_detections)} raw detections, "
                    f"{len(tracked_objects)} unique objects after tracking")
        
        return tracked_objects
    
    def _track_across_frames(self, detections: List[Dict]) -> List[Dict]:
        """
        Track same object across multiple frames using IoU matching
        Merges duplicate detections of same object
        """
        if not detections:
            return []
        
        tracked = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
            
            # Find all detections of same object in other frames
            track = [det1]
            used.add(i)
            
            for j, det2 in enumerate(detections[i+1:], start=i+1):
                if j in used:
                    continue
                
                # Same class + high IoU → same object
                if (det1["class_id"] == det2["class_id"] and
                    self._compute_iou(det1["bbox"], det2["bbox"]) > 0.5):
                    track.append(det2)
                    used.add(j)
            
            # Aggregate track info (use best detection)
            best_det = max(track, key=lambda d: d["confidence"])
            
            tracked_obj = {
                "object_id": len(tracked),
                "class_id": best_det["class_id"],
                "class_name": best_det["class_name"],
                "frames": [d["frame_idx"] for d in track],
                "bbox": best_det["bbox"],  # Best bbox
                "confidence": best_det["confidence"],
                "crop": best_det["crop"],
                "num_appearances": len(track)
            }
            
            tracked.append(tracked_obj)
        
        return tracked
    
    def _compute_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """
        Compute Intersection over Union (IoU) between two bboxes
        
        Args:
            bbox1, bbox2: [x1, y1, x2, y2]
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Union
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        # IoU
        iou = inter_area / union_area if union_area > 0 else 0
        
        return iou
    
    def visualize_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes on frame for visualization
        
        Args:
            frame: Input frame (H, W, C)
            detections: List of detections
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            class_name = det["class_name"]
            conf = det["confidence"]
            
            # Draw bbox
            color = self._get_class_color(det["class_id"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(
                annotated, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        
        return annotated
    
    def _get_class_color(self, class_id: int) -> tuple:
        """Get color for each class (for visualization)"""
        colors = [
            (255, 0, 0),     # Red - prohibitory
            (255, 165, 0),   # Orange - warning
            (0, 0, 255),     # Blue - mandatory
            (0, 255, 0),     # Green - guide
            (128, 128, 128), # Gray - supplementary
            (255, 255, 0),   # Yellow - traffic light
            (0, 255, 255),   # Cyan - vehicles
        ]
        return colors[class_id % len(colors)]
