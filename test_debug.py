"""Debug: single detection frame step-by-step."""
import sys, os, time
os.chdir('F:/MyWork/Article02/Pfe/projet')
sys.path.insert(0, '.')

import cv2, pandas as pd, numpy as np

from utils.file_read import read_all, ais_initial, update_time, time2stamp
from utils.VIS_utils import VISPRO, preprocess_frame, yolo, deepsort

clip_path = './clip-01/'
result_dir = './data/results/test'
video_path, ais_path, _, _, initial_time, camera_para = read_all(clip_path, result_dir + '/')

cap = cv2.VideoCapture(video_path)
im_shape = [cap.get(3), cap.get(4)]
fps = int(cap.get(5))
t = int(1000 / fps)

ais_file = ais_initial(ais_path, initial_time)[0]
from utils.AIS_utils import AISPRO
from utils.FUS_utils import FUSPRO
AIS = AISPRO(ais_path, ais_file, im_shape, t)
FUS = FUSPRO(min(im_shape) // 2, im_shape, t)

Time = initial_time.copy()
frame_idx = 0

while frame_idx < 30:
    ret, im = cap.read()
    if im is None:
        break
    Time, timestamp, Time_name = update_time(Time, t)

    AIS_vis, AIS_cur = AIS.process(camera_para, timestamp, Time_name)

    if timestamp % 1000 < t:
        print(f'[{Time_name}] Detection frame - processing...', flush=True)

        image, brightness, sharpness, condition = preprocess_frame(im)
        print(f'  Preprocessed: brightness={brightness:.1f}', flush=True)

        bboxes = yolo.detect_image(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))) if 'Image' in dir() else []
        from PIL import Image
        bboxes = yolo.detect_image(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        print(f'  YOLOX: {len(bboxes)} detections', flush=True)

        if len(bboxes) > 0:
            import torch
            bbox_xywh = []
            confs = []
            for x1, y1, x2, y2, cls, conf in bboxes:
                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w/2, y1 + h/2
                bbox_xywh.append([cx, cy, w, h])
                confs.append(float(conf))
            
            bbox_xywh = np.array(bbox_xywh)
            confs = np.array(confs)
            
            print(f'  DeepSORT update: {len(bbox_xywh)} boxes...', flush=True)
            outputs = deepsort.update(bbox_xywh, confs, image, [], [], [], timestamp)
            print(f'  DeepSORT: {len(outputs)} tracked', flush=True)

    frame_idx += 1

cap.release()
print(f'Done: {frame_idx} frames', flush=True)
