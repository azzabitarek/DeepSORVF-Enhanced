"""
modules/models/yolov8_wrapper.py — YOLOv8 / KOLOMVERSE detector wrapper.
Optional import — gracefully disables if ultralytics is not installed.
"""

import torch
import numpy as np
from pathlib import Path

try:
    from ultralytics import YOLO as UltralyticsYOLO
    ULTRALYTICS_OK = True
except ImportError:
    ULTRALYTICS_OK = False


class YOLOv8Detector:
    """
    YOLOv8 detector wrapper — same interface as YOLOXDetector.

    Parameters
    ----------
    weights : str
        Path to .pt weights
    conf : float
        Detection confidence threshold
    iou : float
        NMS IoU threshold
    imgsz : int
        Inference image size
    device : str or None
        'cuda', 'cpu', or None (auto-detect)
    """

    def __init__(self, weights=None, conf=0.35, iou=0.45,
                 imgsz=640, device=None):
        if not ULTRALYTICS_OK:
            raise RuntimeError("ultralytics not installed — run: pip install ultralytics")

        project_root = Path(__file__).resolve().parent.parent.parent
        self.weights = weights or str(project_root / "best.pt")
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model = UltralyticsYOLO(self.weights)
        print(f'[YOLOv8] Loaded {self.weights} on {self.device} conf={conf} iou={iou}')

    def detect_image(self, image_bgr):
        """
        Detect objects in a BGR numpy frame.

        Parameters
        ----------
        image_bgr : numpy.ndarray
            BGR frame from cv2

        Returns
        -------
        list of (x1, y1, x2, y2, class_name, conf_tensor)
        """
        results = self.model.predict(
            source=image_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        out = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            names = r.names

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                conf = float(confs[i])
                name = names.get(int(cls_ids[i]), 'vessel')
                out.append((
                    int(x1), int(y1), int(x2), int(y2),
                    name,
                    torch.tensor(conf)
                ))
        return out

    def set_conf(self, new_conf):
        """Update confidence threshold dynamically (for weather-adaptive mode)."""
        self.conf = new_conf
