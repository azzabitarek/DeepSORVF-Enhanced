"""
modules/dataset/build_voc.py — VOC XML ↔ YOLO TXT conversion and dataset split builder.
Handles the pipeline: extracted frames → annotations → YOLO training format.
"""

import os
import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime


class VOCBuilder:
    """
    Builds VOC-format dataset from annotated frames, then converts to YOLO.

    Directory structure created:
        voc_root/
            VOC2007/
                Annotations/   ← XML files
                JPEGImages/    ← JPEG images
                ImageSets/Main/
                    trainval.txt
                    train.txt
                    val.txt
    """

    def __init__(self, voc_root, classes_path, seed=42):
        self.voc_root = Path(voc_root)
        self.classes_path = Path(classes_path)
        self.seed = seed

        self.ann_dir = self.voc_root / "VOC2007" / "Annotations"
        self.img_dir = self.voc_root / "VOC2007" / "JPEGImages"
        self.sets_dir = self.voc_root / "VOC2007" / "ImageSets" / "Main"

        for d in [self.ann_dir, self.img_dir, self.sets_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.classes = self._load_classes()

    def _load_classes(self):
        """Load class names from classes.txt file."""
        with open(self.classes_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def add_frame(self, frame_name, image_path, annotations):
        """
        Add a frame with annotations to the VOC dataset.

        Parameters
        ----------
        frame_name : str
            Base name without extension (e.g. 'frame_0001')
        image_path : str or Path
            Path to the JPEG image
        annotations : list of dict
            Each dict: {class: str, xmin: int, ymin: int, xmax: int, ymax: int}
        """
        # Copy image
        src = Path(image_path)
        dst = self.img_dir / f"{frame_name}.jpg"
        if src.exists():
            shutil.copy2(str(src), str(dst))

        # Create XML
        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = "VOC2007"
        ET.SubElement(root, "filename").text = f"{frame_name}.jpg"

        size_el = ET.SubElement(root, "size")
        ET.SubElement(size_el, "width").text = "1920"
        ET.SubElement(size_el, "height").text = "1080"
        ET.SubElement(size_el, "depth").text = "3"

        for ann in annotations:
            cls_name = ann["class"]
            if cls_name not in self.classes:
                continue
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = cls_name
            ET.SubElement(obj, "difficult").text = "0"
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(ann["xmin"]))
            ET.SubElement(bndbox, "ymin").text = str(int(ann["ymin"]))
            ET.SubElement(bndbox, "xmax").text = str(int(ann["xmax"]))
            ET.SubElement(bndbox, "ymax").text = str(int(ann["ymax"]))

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(str(self.ann_dir / f"{frame_name}.xml"), encoding="utf-8")

    def build_splits(self, trainval_pct=0.95, train_pct=0.95):
        """
        Generate train/val/test splits from all annotations.

        Returns
        -------
        dict with split statistics
        """
        all_xmls = sorted([f.stem for f in self.ann_dir.glob("*.xml")])
        n = len(all_xmls)
        random.seed(self.seed)

        n_tv = int(n * trainval_pct)
        n_train = int(n_tv * train_pct)

        indices = list(range(n))
        tv_indices = set(random.sample(indices, n_tv))
        train_indices = set(random.sample(list(tv_indices), n_train))

        splits = {"trainval": [], "train": [], "val": [], "test": []}
        for i, name in enumerate(all_xmls):
            if i in train_indices:
                splits["train"].append(name)
                splits["trainval"].append(name)
            elif i in tv_indices:
                splits["val"].append(name)
                splits["trainval"].append(name)
            else:
                splits["test"].append(name)

        for split_name, names in splits.items():
            fpath = self.sets_dir / f"{split_name}.txt"
            with open(fpath, "w", encoding="utf-8") as f:
                f.write("\n".join(names) + "\n")

        stats = {
            "total": n,
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "trainval": len(splits["trainval"]),
            "test": len(splits["test"]),
            "created_at": datetime.now().isoformat()
        }

        # Save stats
        stats_path = self.voc_root / "split_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        print(f"[VOCBuilder] Splits: train={stats['train']}, val={stats['val']}, test={stats['test']}")
        return stats

    def convert_to_yolo(self, output_dir, trainval_pct=0.95, train_pct=0.95):
        """
        Convert VOC annotations to YOLO format and generate train/val txt files.

        Returns
        -------
        dict with conversion statistics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build splits first
        splits = self.build_splits(trainval_pct, train_pct)

        class_to_id = {name: i for i, name in enumerate(self.classes)}

        # Convert each XML
        for xml_file in self.ann_dir.glob("*.xml"):
            tree = ET.parse(str(xml_file))
            root = tree.getroot()

            img_w = int(root.find("size/width").text)
            img_h = int(root.find("size/height").text)

            yolo_lines = []
            for obj in root.iter("object"):
                cls_name = obj.find("name").text
                if cls_name not in class_to_id:
                    continue
                difficult = obj.find("difficult")
                if difficult is not None and int(difficult.text) == 1:
                    continue

                cls_id = class_to_id[cls_name]
                bndbox = obj.find("bndbox")
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)

                cx = (xmin + xmax) / 2 / img_w
                cy = (ymin + ymax) / 2 / img_h
                w = (xmax - xmin) / img_w
                h = (ymax - ymin) / img_h

                yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            txt_path = output_dir / f"{xml_file.stem}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))

        # Generate train/val txt with absolute paths
        for split_name in ["train", "val"]:
            split_file = self.sets_dir / f"{split_name}.txt"
            if not split_file.exists():
                continue
            with open(split_file, "r", encoding="utf-8") as f:
                names = [line.strip() for line in f if line.strip()]

            yolo_list_path = output_dir.parent / f"{split_name}.txt"
            with open(yolo_list_path, "w", encoding="utf-8") as f:
                for name in names:
                    img_path = str(self.img_dir / f"{name}.jpg").replace("\\", "/")
                    label_path = str(output_dir / f"{name}.txt").replace("\\", "/")
                    f.write(f"{img_path} {label_path}\n")

        print(f"[VOCBuilder] YOLO labels → {output_dir}")
        print(f"[VOCBuilder] YOLO train/val lists → {output_dir.parent}")
        return splits
