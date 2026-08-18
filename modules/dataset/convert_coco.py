"""
modules/dataset/convert_coco.py — COCO-to-YOLO format converter for SeaDronesSee dataset.
Converts COCO JSON annotations to YOLO txt labels.
"""

import json
import os
from pathlib import Path
from collections import defaultdict


class COCOToYOLO:
    """
    Converts COCO-format annotations to YOLO txt format.

    SeaDronesSee COCO structure:
        annotations/
            instances_*.json  (COCO format with categories, images, annotations)
        images/
            *.jpg

    YOLO output:
        labels/
            <image_id>.txt    (class_id cx cy w h normalized)
    """

    # SeaDronesSee class mapping (COCO category_id → class_name)
    SEADRONESEE_CLASSES = {
        0: "swimmer",
        1: "boat",
        2: "ship",
        3: "windsurfer",
        4: "jetski",
        5: "lifesaver",
        6: "buoy",
        7: "zodiac",
        8: "sailboat",
        9: "kayak",
    }

    def __init__(self, coco_json_path, images_dir, output_labels_dir, class_map=None):
        self.coco_json_path = Path(coco_json_path)
        self.images_dir = Path(images_dir)
        self.output_labels_dir = Path(output_labels_dir)
        self.output_labels_dir.mkdir(parents=True, exist_ok=True)

        self.class_map = class_map or self.SEADRONESEE_CLASSES
        self.coco_data = None
        self.cat_id_to_class_id = {}
        self.image_id_to_info = {}

    def load_coco(self):
        """Load and parse the COCO JSON file."""
        with open(self.coco_json_path, "r", encoding="utf-8") as f:
            self.coco_data = json.load(f)

        # Build category mapping: COCO category_id → our class_id
        for cat in self.coco_data.get("categories", []):
            coco_id = cat["id"]
            name = cat["name"].lower().strip()
            # Find matching class in our map
            for cls_id, cls_name in self.class_map.items():
                if cls_name.lower() == name:
                    self.cat_id_to_class_id[coco_id] = cls_id
                    break

        # Build image lookup
        for img in self.coco_data.get("images", []):
            self.image_id_to_info[img["id"]] = img

        print(f"[COCO] Loaded {len(self.image_id_to_info)} images, "
              f"{len(self.coco_data.get('annotations', []))} annotations")
        print(f"[COCO] Class map: {self.cat_id_to_class_id}")

    def convert(self):
        """
        Convert all COCO annotations to YOLO txt format.

        Returns
        -------
        dict with conversion statistics
        """
        if self.coco_data is None:
            self.load_coco()

        annotations = self.coco_data.get("annotations", [])

        # Group annotations by image_id
        ann_by_image = defaultdict(list)
        for ann in annotations:
            ann_by_image[ann["image_id"]].append(ann)

        total_labels = 0
        skipped = 0

        for image_id, anns in ann_by_image.items():
            img_info = self.image_id_to_info.get(image_id)
            if img_info is None:
                skipped += 1
                continue

            img_w = img_info["width"]
            img_h = img_info["height"]
            img_filename = Path(img_info["file_name"]).stem

            yolo_lines = []
            for ann in anns:
                cat_id = ann["category_id"]
                if cat_id not in self.cat_id_to_class_id:
                    skipped += 1
                    continue

                class_id = self.cat_id_to_class_id[cat_id]
                bbox = ann["bbox"]  # [x_top_left, y_top_left, width, height]

                cx = (bbox[0] + bbox[2] / 2) / img_w
                cy = (bbox[1] + bbox[3] / 2) / img_h
                w = bbox[2] / img_w
                h = bbox[3] / img_h

                # Clamp to [0, 1]
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))

                yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                total_labels += 1

            if yolo_lines:
                txt_path = self.output_labels_dir / f"{img_filename}.txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(yolo_lines))

        stats = {
            "total_images": len(ann_by_image),
            "total_labels": total_labels,
            "skipped": skipped,
            "output_dir": str(self.output_labels_dir),
        }
        print(f"[COCO] Converted {total_labels} labels, skipped {skipped}")
        return stats

    def generate_data_yaml(self, output_yaml_path, train_images=None, val_images=None):
        """
        Generate a YOLOv8 data.yaml file for training.

        Parameters
        ----------
        output_yaml_path : str or Path
            Path to write the YAML file
        train_images : str
            Path to training images directory
        val_images : str
            Path to validation images directory
        """
        output_yaml_path = Path(output_yaml_path)
        output_yaml_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"# SeaDronesSee dataset config — generated {__import__('datetime').datetime.now().isoformat()}",
            f"train: {train_images or str(self.images_dir / 'train')}",
            f"val: {val_images or str(self.images_dir / 'val')}",
            f"nc: {len(self.class_map)}",
            f"names: {list(self.class_map.values())}",
        ]

        with open(output_yaml_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[COCO] data.yaml → {output_yaml_path}")
