"""
detection_yolov8/yolov8_detector.py
────────────────────────────────────
Wrapper around ultralytics YOLOv8 / best.pt that produces the exact same
output format as YOLOX detect_image():

    List of (x1, y1, x2, y2, class_name, conf_tensor)

Drop-in compatible with VIS_utils.py track() which reads:
    for x1, y1, x2, y2, _, conf in bboxes
"""

import torch
import numpy as np

try:
    from ultralytics import YOLO as UltralyticsYOLO
    ULTRALYTICS_OK = True
except ImportError:
    ULTRALYTICS_OK = False
    print('[YOLOv8] ultralytics not installed — run: pip install ultralytics')


class YOLOv8Detector:
    """
    Parameters
    ----------
    weights   : path to best.pt
    conf      : detection confidence threshold (default 0.35 — tuned for maritime)
    iou       : NMS IoU threshold
    imgsz     : inference image size
    device    : 'cuda' or 'cpu'
    """
    def __init__(self, weights='best.pt', conf=0.35, iou=0.45,
                 imgsz=640, device=None):
        if not ULTRALYTICS_OK:
            raise RuntimeError('ultralytics not installed')

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model  = UltralyticsYOLO(weights)
        self.conf   = conf
        self.iou    = iou
        self.imgsz  = imgsz
        self.device = device
        print(f'[YOLOv8] Loaded {weights} on {device}  conf={conf}  iou={iou}')

    def detect_image(self, image_bgr):
        """
        Parameters
        ----------
        image_bgr : numpy BGR frame (as used throughout VIS_utils)

        Returns
        -------
        List of (x1, y1, x2, y2, class_name, conf_tensor)
        Same format as YOLOX detect_image().
        """
        results = self.model.predict(
            source   = image_bgr,
            conf     = self.conf,
            iou      = self.iou,
            imgsz    = self.imgsz,
            device   = self.device,
            verbose  = False,
        )

        out = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            boxes  = r.boxes.xyxy.cpu().numpy()    # (N,4) x1y1x2y2
            confs  = r.boxes.conf.cpu().numpy()    # (N,)
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)  # (N,)
            names  = r.names                       # {id: name}

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
