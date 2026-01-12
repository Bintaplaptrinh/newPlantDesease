# module detect lá bằng yolo cho pipeline nhận diện bệnh cây
# detect chạy ở độ phân giải thấp, nhưng crop trên ảnh gốc độ phân giải cao để giữ chất lượng
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class DetectionBox:
    # biểu diễn một bounding box đã detect
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    def scale_to_original(
        self, 
        original_size: Tuple[int, int], 
        resized_size: Tuple[int, int]
    ) -> "DetectionBox":
        # scale bbox từ tọa độ ảnh resize về tọa độ ảnh gốc
        # original_size: (width, height) của ảnh gốc
        # resized_size: (width, height) của ảnh dùng để detect
        # trả về DetectionBox mới với tọa độ đã scale
        orig_w, orig_h = original_size
        resized_w, resized_h = resized_size
        
        scale_x = orig_w / resized_w
        scale_y = orig_h / resized_h
        
        return DetectionBox(
            x1=int(self.x1 * scale_x),
            y1=int(self.y1 * scale_y),
            x2=int(self.x2 * scale_x),
            y2=int(self.y2 * scale_y),
            confidence=self.confidence,
            class_id=self.class_id,
            class_name=self.class_name
        )

    def clamp_to_image(self, width: int, height: int) -> "DetectionBox":
        # giới hạn tọa độ bbox nằm trong biên ảnh
        return DetectionBox(
            x1=max(0, min(self.x1, width - 1)),
            y1=max(0, min(self.y1, height - 1)),
            x2=max(0, min(self.x2, width)),
            y2=max(0, min(self.y2, height)),
            confidence=self.confidence,
            class_id=self.class_id,
            class_name=self.class_name
        )


class YOLODetector:
    # yolo detector dùng để phát hiện lá

    DEFAULT_DETECTION_SIZE = 640  # Standard YOLO input size
    DEFAULT_CONFIDENCE_THRESHOLD = 0.25
    DEFAULT_IOU_THRESHOLD = 0.45

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        detection_size: int = DEFAULT_DETECTION_SIZE,
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.detection_size = detection_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._model = None

    def _load_model(self):
        # load model yolo theo kiểu lazy (chỉ load khi cần)
        if self._model is None:
            try:
                from ultralytics import YOLO
                self._model = YOLO(str(self.model_path))
                # Move to device if GPU
                if self.device == "cuda":
                    self._model.to(self.device)
            except ImportError:
                raise ImportError(
                    "ultralytics package is required for YOLO detection. "
                    "Install it with: pip install ultralytics"
                )
        return self._model

    def detect(
        self,
        image: Image.Image,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> List[DetectionBox]:
        # detect đối tượng trong ảnh
        # detect chạy ở độ phân giải thấp (detection_size), nhưng tọa độ trả về theo ảnh gốc
        # conf_threshold/iou_threshold có thể override giá trị mặc định
        model = self._load_model()
        
        # Store original size
        original_size = image.size  # (width, height)
        
        # Resize for detection (YOLO handles this internally, but we track it)
        # YOLO will resize internally, so we need to pass imgsz
        conf = self.conf_threshold if conf_threshold is None else float(conf_threshold)
        iou = self.iou_threshold if iou_threshold is None else float(iou_threshold)
        
        # Run detection
        results = model.predict(
            image,
            imgsz=self.detection_size,
            conf=conf,
            iou=iou,
            verbose=False,
            device=self.device,
        )
        
        detections: List[DetectionBox] = []
        
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                # Get boxes in xyxy format (already scaled to original image size by ultralytics)
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    conf_val = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    
                    # Get class name if available
                    cls_name = result.names.get(cls_id, f"class_{cls_id}")
                    
                    box = DetectionBox(
                        x1=int(xyxy[0]),
                        y1=int(xyxy[1]),
                        x2=int(xyxy[2]),
                        y2=int(xyxy[3]),
                        confidence=conf_val,
                        class_id=cls_id,
                        class_name=cls_name
                    )
                    
                    # Clamp to image boundaries just in case
                    box = box.clamp_to_image(original_size[0], original_size[1])
                    detections.append(box)
        
        # Sort by area (largest first) or confidence
        detections.sort(key=lambda d: d.area, reverse=True)
        
        return detections

    def detect_and_crop(
        self,
        image: Image.Image,
        padding: int = 10,
        conf_threshold: Optional[float] = None,
    ) -> Tuple[Image.Image, Optional[DetectionBox]]:
        # detect đối tượng tốt/đại diện nhất rồi crop từ ảnh gốc
        # chiến lược: detect low-res, crop high-res để giữ chất lượng
        # nếu không detect được thì trả về (ảnh gốc, None)
        detections = self.detect(image, conf_threshold=conf_threshold)
        
        if not detections:
            return image, None
        
        # Use the largest detection (first after sorting)
        best_box = detections[0]
        
        # Add padding
        w, h = image.size
        x1 = max(0, best_box.x1 - padding)
        y1 = max(0, best_box.y1 - padding)
        x2 = min(w, best_box.x2 + padding)
        y2 = min(h, best_box.y2 + padding)
        
        # Crop from original high-res image
        cropped = image.crop((x1, y1, x2, y2))
        
        # Update box to reflect actual crop coordinates
        final_box = DetectionBox(
            x1=x1, y1=y1, x2=x2, y2=y2,
            confidence=best_box.confidence,
            class_id=best_box.class_id,
            class_name=best_box.class_name
        )
        
        return cropped, final_box


@lru_cache(maxsize=1)
def get_yolo_detector(model_path: str, device: str = "cpu") -> YOLODetector:
    # lấy instance yolo detector theo cache
    return YOLODetector(model_path=model_path, device=device)
