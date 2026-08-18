# modules/dataset/__init__.py
"""Dataset preparation: frame extraction, annotation, conversion."""

from .extract_frames import FrameExtractor
from .cache_frames import FrameCache
from .build_voc import VOCBuilder
from .convert_coco import COCOToYOLO
