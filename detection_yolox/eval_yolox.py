"""
eval_yolox.py — Évaluation YOLOX-S avec annotations VOC XML
"""
import sys, os
sys.path.append('.')
 
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
 
from detection_yolox.yolo import YOLO
 
model = YOLO()
 
VAL_TXT  = r'C:\Users\alach\OneDrive\Desktop\modele_1\goodluck\detection_yolox\2007_val.txt'
ANN_DIR  = r'C:\Users\alach\OneDrive\Desktop\modele_1\new\VOCdevkit_val\VOC2007\Annotations'
IOU_THRESH = 0.5
 
def load_gt(img_path):
    name = os.path.splitext(os.path.basename(img_path))[0]
    xml_path = os.path.join(ANN_DIR, name + '.xml')
    if not os.path.exists(xml_path):
        return []
    root = ET.parse(xml_path).getroot()

    # Taille dans l'XML (peut différer de l'image réelle)
    xml_w = int(root.find('size/width').text)
    xml_h = int(root.find('size/height').text)

    # Taille réelle de l'image
    from PIL import Image as PILImage
    real_w, real_h = PILImage.open(img_path).size

    # Facteurs de rescaling
    sx = real_w / xml_w
    sy = real_h / xml_h

    boxes = []
    for obj in root.findall('object'):
        bb = obj.find('bndbox')
        x1 = int(float(bb.find('xmin').text) * sx)
        y1 = int(float(bb.find('ymin').text) * sy)
        x2 = int(float(bb.find('xmax').text) * sx)
        y2 = int(float(bb.find('ymax').text) * sy)
        boxes.append([x1, y1, x2, y2])
    return boxes
 
def iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0: return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)
 
with open(VAL_TXT) as f:
    img_paths = [l.strip() for l in f if l.strip()]
 
TP_total = FP_total = FN_total = 0
all_precisions = []
all_recalls    = []
 
print(f"Évaluation sur {len(img_paths)} images de validation...\n")
 
for img_path in img_paths:
    img_path_clean = img_path.replace('/', os.sep).replace('\\', os.sep)
    if not os.path.exists(img_path_clean):
        print(f"  Image introuvable : {img_path_clean}")
        continue
 
    gt_boxes   = load_gt(img_path_clean)
    img        = Image.open(img_path_clean).convert('RGB')
    preds      = model.detect_image(img)
 
    matched_gt = set()
    TP = FP = 0
 
    for pred in preds:
        px1, py1, px2, py2 = int(pred[0]), int(pred[1]), int(pred[2]), int(pred[3])
        best_iou = 0
        best_idx = -1
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            score = iou([px1,py1,px2,py2], gt)
            if score > best_iou:
                best_iou = score
                best_idx = i
        if best_iou >= IOU_THRESH:
            TP += 1
            matched_gt.add(best_idx)
        else:
            FP += 1
 
    FN = len(gt_boxes) - len(matched_gt)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
 
    print(f"  {os.path.basename(img_path_clean):15s} | GT={len(gt_boxes)} | Preds={len(preds)} | TP={TP} FP={FP} FN={FN} | P={precision:.3f} R={recall:.3f}")
 
    TP_total += TP
    FP_total += FP
    FN_total += FN
    all_precisions.append(precision)
    all_recalls.append(recall)
 
precision_global = TP_total / (TP_total + FP_total) if (TP_total + FP_total) > 0 else 0
recall_global    = TP_total / (TP_total + FN_total) if (TP_total + FN_total) > 0 else 0
f1_global        = 2 * precision_global * recall_global / (precision_global + recall_global + 1e-6)
 
print(f"""
{'='*55}
RÉSULTATS FINAUX YOLOX-S  (seuil IoU = {IOU_THRESH})
{'='*55}
Images évaluées  : {len(img_paths)}
TP={TP_total}  FP={FP_total}  FN={FN_total}
 
Précision        : {precision_global:.4f}
Rappel           : {recall_global:.4f}
F1-Score         : {f1_global:.4f}
 
Note : mAP@50 ≈ Précision × Rappel (dataset val limité à 5 images)
{'='*55}
""")