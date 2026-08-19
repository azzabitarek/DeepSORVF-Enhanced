"""Step-by-step debug of 1 frame through VIS pipeline."""
import sys, os, time
os.chdir('F:/MyWork/Article02/Pfe/projet')
sys.path.insert(0, '.')

import cv2
print("cv2 OK")
import pandas as pd
print("pd OK")

cap = cv2.VideoCapture('./clip-01/2022_06_04_12_05_12_12_07_02_b.mp4')
ret, im = cap.read()
cap.release()
print(f"Frame: {im.shape}")

from utils.VIS_utils import VISPRO, preprocess_frame
print("VIS_utils imported")

# Step 1: preprocess
print("Preprocessing...")
t0 = time.time()
image, brightness, sharpness, condition = preprocess_frame(im)
print(f"Preprocess: {time.time()-t0:.2f}s, brightness={brightness:.1f}")

# Step 2: YOLOX detection (via the global yolo object)
from utils.VIS_utils import yolo
print("Running YOLOX...")
t1 = time.time()
from PIL import Image
bboxes = yolo.detect_image(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
print(f"YOLOX: {time.time()-t1:.2f}s, {len(bboxes)} detections")
for b in bboxes:
    print(f"  {b[4]} {b[5]:.2f} [{b[0]},{b[1]},{b[2]},{b[3]}]")

# Step 3: YOLOv8
from utils.VIS_utils import _yolov8
if _yolov8 is not None:
    print("Running YOLOv8...")
    t2 = time.time()
    maritime = _yolov8.detect_image(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    print(f"YOLOv8: {time.time()-t2:.2f}s, {len(maritime)} detections")
else:
    print("YOLOv8 not loaded")

print("DONE")
