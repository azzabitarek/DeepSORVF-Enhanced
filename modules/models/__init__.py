# modules/models/__init__.py
"""Model loading, wrappers, and checkpoint management."""

from .checkpoint_manager import CheckpointManager
from .yolox_wrapper import YOLOXDetector
from .yolov8_wrapper import YOLOv8Detector
