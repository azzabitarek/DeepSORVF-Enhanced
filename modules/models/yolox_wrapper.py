"""
modules/models/yolox_wrapper.py — YOLOX detector wrapper with device flexibility.
Removes hardcoded CUDA dependency from original detection_yolox/yolo.py.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path


class YOLOXDetector:
    """
    YOLOX vessel detector — drop-in replacement for detection_yolox.yolo.YOLO
    with automatic device detection (CUDA/CPU).

    Parameters
    ----------
    model_path : str
        Path to YOLOX .pth weights
    classes_path : str
        Path to classes.txt
    confidence : float
        Detection confidence threshold
    nms_iou : float
        NMS IoU threshold
    input_shape : list
        Input resolution [H, W]
    device : str or None
        'cuda', 'cpu', or None (auto-detect)
    """

    def __init__(self, model_path=None, classes_path=None,
                 confidence=0.5, nms_iou=0.3, input_shape=None,
                 device=None, letterbox_image=True):
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.model_path = model_path or str(self.project_root / "detection_yolox/model_data/YOLOX-final.pth")
        self.classes_path = classes_path or str(self.project_root / "detection_yolox/model_data/ship_classes.txt")
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.input_shape = input_shape or [640, 640]
        self.letterbox_image = letterbox_image

        # Device selection
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.cuda = self.device.type == 'cuda'

        self.net = None
        self.class_names = None
        self.num_classes = 0
        self._load_model()

    def _load_classes(self):
        """Load class names from classes_path."""
        with open(self.classes_path, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f if line.strip()]
        return class_names, len(class_names)

    def _load_model(self):
        """Load YOLOX model weights."""
        # Add parent path for imports
        yolox_dir = str(self.project_root / "detection_yolox")
        if yolox_dir not in sys.path:
            sys.path.insert(0, yolox_dir)

        from nets.yolo import YoloBody
        from utils.utils import get_classes as _get_classes

        self.class_names, self.num_classes = _get_classes(self.classes_path)

        self.net = YoloBody(self.num_classes, phi='s')
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.net.load_state_dict(state_dict)
        self.net = self.net.eval()

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

        print(f'[YOLOX] Loaded {self.model_path} on {self.device}')

    def detect_image(self, image_pil):
        """
        Detect vessels in a PIL image.

        Parameters
        ----------
        image_pil : PIL.Image
            RGB image

        Returns
        -------
        list of (x1, y1, x2, y2, class_name, conf_tensor)
        """
        import cv2
        from utils.utils import cvtColor, preprocess_input, resize_image
        from utils.utils_bbox import decode_outputs, non_max_suppression

        image_shape = np.array(np.shape(image_pil)[0:2])
        image = cvtColor(image_pil)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            outputs = self.net(images)
            outputs = decode_outputs(outputs, self.input_shape)
            results = non_max_suppression(outputs, self.num_classes, self.input_shape,
                                          image_shape, self.letterbox_image,
                                          conf_thres=self.confidence, nms_thres=self.nms_iou)

            if results[0] is None:
                return []

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]

        out = []
        for i, c in enumerate(top_label):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]
            top, left, bottom, right = box

            y1 = max(0, np.floor(top).astype('int32'))
            x1 = max(0, np.floor(left).astype('int32'))
            y2 = min(image.size[1], np.floor(bottom).astype('int32'))
            x2 = min(image.size[0], np.floor(right).astype('int32'))

            if self.cuda:
                conf_tensor = torch.from_numpy(np.array(score)).to(self.device)
            else:
                conf_tensor = torch.tensor(score)
            out.append((x1, y1, x2, y2, predicted_class, conf_tensor))

        return out
