"""Test: run 100 frames from clip-01, print detection results."""
import sys, os, time
os.chdir('F:/MyWork/Article02/Pfe/projet')
sys.path.insert(0, '.')

import cv2, pandas as pd
from utils.file_read import read_all, ais_initial, update_time, time2stamp
from utils.VIS_utils import VISPRO
from utils.AIS_utils import AISPRO
from utils.FUS_utils import FUSPRO

clip_path = './clip-01/'
result_dir = './data/results/test'
video_path, ais_path, result_video, result_metric, initial_time, camera_para = read_all(clip_path, result_dir + '/')

cap = cv2.VideoCapture(video_path)
im_shape = [cap.get(3), cap.get(4)]
max_dis = min(im_shape) // 2
fps = int(cap.get(5))
t = int(1000 / fps)

ais_file = ais_initial(ais_path, initial_time)[0]
AIS = AISPRO(ais_path, ais_file, im_shape, t)
VIS = VISPRO(1, 0, t)
FUS = FUSPRO(max_dis, im_shape, t)

Time = initial_time.copy()
frame_idx = 0
start = time.time()

while frame_idx < 100:
    ret, im = cap.read()
    if im is None:
        break
    Time, timestamp, Time_name = update_time(Time, t)
    t0 = time.time()
    AIS_vis, AIS_cur = AIS.process(camera_para, timestamp, Time_name)
    Vis_tra, Vis_cur = VIS.feedCap(im, timestamp, AIS_vis, pd.DataFrame(columns=['ID', 'mmsi', 'timestamp', 'match']))
    Fus_tra, _ = FUS.fusion(AIS_vis, AIS_cur, Vis_tra, Vis_cur, timestamp)
    elapsed = time.time() - t0
    if timestamp % 1000 < t:
        n_det = len(Vis_cur) if Vis_cur is not None else 0
        print(f'[{Time_name}] detections={n_det} vis={len(Vis_tra) if Vis_tra is not None else 0} ({elapsed:.2f}s)')
    frame_idx += 1

cap.release()
print(f'\nDone: {frame_idx} frames in {time.time()-start:.1f}s')
