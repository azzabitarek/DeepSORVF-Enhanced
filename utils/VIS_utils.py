import csv
import math
import numpy as np
import cv2
import torch
from PIL import Image
import pandas as pd
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from warnings import simplefilter
import cv2
from PIL import Image
import pandas as pd
from IPython import embed
import os
simplefilter(action='ignore', category=FutureWarning)

# ── YOLOX — vessel detector (feeds DeepSORT + AIS fusion) ────────────────────
from detection_yolox.yolo import YOLO
yolo = YOLO()
print('[YOLOX] Vessel detector loaded ✓')

# ── YOLOv8 KOLOMVERSE — maritime objects detector (display only) ─────────────
from detection_yolov8.yolov8_detector import YOLOv8Detector, ULTRALYTICS_OK
_yolov8 = None
# Always resolve best.pt relative to this file's directory (project root)
_YOLOV8_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'weights', 'best.pt')

def _load_yolov8():
    global _yolov8
    if _yolov8 is None and ULTRALYTICS_OK:
        try:
            if os.path.exists(_YOLOV8_WEIGHTS):
                _yolov8 = YOLOv8Detector(weights=_YOLOV8_WEIGHTS, conf=0.30)
                print(f'[YOLOv8 KOLOMVERSE] {_YOLOV8_WEIGHTS} loaded ✓')
            else:
                print(f'[YOLOv8] best.pt not found at {_YOLOV8_WEIGHTS} — maritime detection disabled')
        except Exception as e:
            print(f'[YOLOv8] Load failed: {e} — maritime detection disabled')

_load_yolov8()

# Dynamic YOLOX confidence thresholds — overridden by launcher UI
_YOLOX_CONF_CLEAR = 0.50
_YOLOX_CONF_RAIN  = 0.35
_YOLOX_CONF_FOG   = 0.30

# 初始化跟踪模型
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=torch.cuda.is_available(), use_reid=True)

def _ensemble_nms(boxes_ship, boxes_maritime, iou_thresh=0.5):
    """
    Merge YOLOX ship detections with YOLOv8 maritime object detections.
    
    - Different classes (ship vs buoy/swimmer/etc.) are NEVER suppressed 
      against each other — they detect completely different things.
    - Same class duplicates (e.g. both models detect the same ship) are 
      resolved by keeping the highest confidence box.
    
    Both lists: [(x1,y1,x2,y2,cls_name,conf_tensor), ...]
    """
    if len(boxes_ship) == 0 and len(boxes_maritime) == 0:
        return []

    def _nms_single_class(boxes, iou_thresh):
        """NMS within one class group."""
        if len(boxes) == 0:
            return []
        boxes = sorted(boxes, key=lambda b: float(b[5]), reverse=True)
        kept = []
        suppressed = [False] * len(boxes)
        for i in range(len(boxes)):
            if suppressed[i]:
                continue
            kept.append(boxes[i])
            x1i, y1i, x2i, y2i = boxes[i][:4]
            area_i = max(0, x2i-x1i) * max(0, y2i-y1i)
            for j in range(i+1, len(boxes)):
                if suppressed[j]: continue
                x1j, y1j, x2j, y2j = boxes[j][:4]
                ix1 = max(x1i,x1j); iy1 = max(y1i,y1j)
                ix2 = min(x2i,x2j); iy2 = min(y2i,y2j)
                inter = max(0,ix2-ix1) * max(0,iy2-iy1)
                if inter == 0: continue
                area_j = max(0,x2j-x1j) * max(0,y2j-y1j)
                iou = inter / (area_i + area_j - inter + 1e-6)
                if iou > iou_thresh:
                    suppressed[j] = True
        return kept

    # Group by class name, apply NMS per class, then concatenate all
    from collections import defaultdict
    groups = defaultdict(list)
    for b in boxes_ship + boxes_maritime:
        groups[b[4]].append(b)

    result = []
    for cls_name, cls_boxes in groups.items():
        result.extend(_nms_single_class(cls_boxes, iou_thresh))
    return result

# ──────────────────────────────────────────────────────────────────────────────
# Image quality analysis + adaptive preprocessing
# ──────────────────────────────────────────────────────────────────────────────
_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

def _image_quality(frame):
    """
    Returns (mean_brightness, sharpness).
    mean_brightness: 0-255  (low = dark / night)
    sharpness: Laplacian variance (low = foggy / blurry)
    """
    gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    sharpness  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return brightness, sharpness

def _detect_shoreline(frame):
    """Détecte la ligne de rive avec marge de sécurité plus grande."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Analyser sur une plus grande zone
    search_start = h // 6  # Commencer plus haut
    
    # Lissage pour éviter le bruit
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    best_y, best_grad = search_start, 0.0
    for y in range(search_start, h - 1):
        diff = np.abs(
            blurred[y + 1, :].astype(np.float32) - blurred[y, :].astype(np.float32)
        )
        mean_grad = float(np.percentile(diff, 85))
        if mean_grad > best_grad:
            best_grad = mean_grad
            best_y = y
    
    # Marge de sécurité agrandie
    return max(0, best_y - 60)  # ← 60px au lieu de 30


def _is_static_structure(mb, shore_y, BUILDING_MARGIN=70):
    """
    Filter out static structures (buildings, cranes, wind farms) from
    KOLOMVERSE maritime detections. Returns True if the detection should
    be suppressed (i.e. it IS a static structure).
    """
    cy = (mb[1] + mb[3]) / 2
    cls = mb[4]
    conf = mb[5]

    # Wind farm specific filter
    if cls == 'wind farm':
        if conf < 0.55:
            return True
        if cy < shore_y - 20:
            return True
        width = mb[2] - mb[0]
        height = mb[3] - mb[1]
        if height > 0 and (width / height) > 1.5:
            return True

    # Reject any object above the shoreline (on land)
    if cy < shore_y - BUILDING_MARGIN:
        return True

    return False


def preprocess_frame(frame):
    """
    1. CLAHE on L channel when brightness < 80 (night / dusk)
    2. Weather-adaptive YOLO confidence threshold:
       - Normal:      conf = 0.50
       - Fog/haze:    conf = 0.30  (low sharpness, medium brightness)
       - Rain/low:    conf = 0.35  (low sharpness, low brightness)
    Returns processed frame and adjusts YOLOv8 confidence threshold.
    """
    brightness, sharpness = _image_quality(frame)

    # — Nighttime CLAHE ────────────────────────────────────────────
    if brightness < 80:
        lab     = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l       = _clahe.apply(l)
        frame   = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # — Weather-adaptive YOLOX confidence threshold (values set by launcher UI)
    if sharpness < 80:        # blurry = foggy or rainy
        if brightness > 100:  # foggy (bright but low contrast)
            yolo.confidence = _YOLOX_CONF_FOG
            condition = 'fog'
        else:                 # dark + blurry = rain / night
            yolo.confidence = _YOLOX_CONF_RAIN
            condition = 'rain'
    else:
        yolo.confidence = _YOLOX_CONF_CLEAR
        condition = 'clear'

    return frame, brightness, sharpness, condition


# ──────────────────────────────────────────────────────────────────────────────
# Wake detector — sparse optical flow in the region behind each ship
# ──────────────────────────────────────────────────────────────────────────────
class WakeDetector:
    """
    Estimates wake heading from optical flow in the area behind each tracked ship.
    'Behind' is defined as opposite to the ship's direction of motion.

    Output: {track_id: wake_angle_degrees}  (compass degrees, 0=N, CW)
    Wake angle ≈ the direction the water is being pushed → opposite = ship heading.
    """
    FEATURE_PARAMS = dict(maxCorners=20, qualityLevel=0.2,
                          minDistance=5, blockSize=7)
    LK_PARAMS      = dict(winSize=(15, 15), maxLevel=2,
                          criteria=(cv2.TERM_CRITERIA_EPS |
                                    cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def __init__(self, shoot_hdir=0.0):
        self.prev_gray   = None
        self.shoot_hdir  = shoot_hdir   # camera compass heading (for pixel→compass)
        self.wake_angles = {}           # {track_id: degrees}

    def _roi_behind(self, x1, y1, x2, y2, dx, dy, frame_shape):
        """Return ROI (rx1,ry1,rx2,ry2) in the region behind the ship."""
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        w  = x2 - x1
        h  = y2 - y1
        # Normalise direction vector
        mag = math.sqrt(dx*dx + dy*dy) + 1e-6
        ndx, ndy = dx / mag, dy / mag
        # Move backward by 1 ship height
        offset = int(max(h, 30))
        rx_c = int(cx - ndx * offset)
        ry_c = int(cy - ndy * offset)
        hw   = max(w // 2, 20)
        hh   = max(h // 2, 20)
        H, W = frame_shape[:2]
        rx1  = max(0,   rx_c - hw)
        ry1  = max(0,   ry_c - hh)
        rx2  = min(W-1, rx_c + hw)
        ry2  = min(H-1, ry_c + hh)
        if rx2 <= rx1 or ry2 <= ry1:
            return None
        return rx1, ry1, rx2, ry2

    def update(self, frame, Vis_cur, Vis_tra):
        """
        frame    : current BGR frame
        Vis_cur  : current visual tracks dataframe
        Vis_tra  : full visual trajectory history
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        wake = {}

        if self.prev_gray is not None and Vis_cur is not None and len(Vis_cur) > 0:
            for _, row in Vis_cur.iterrows():
                try:
                    tid = int(row['ID'])
                    x1  = int(float(row['x1']))
                    y1  = int(float(row['y1']))
                    x2  = int(float(row['x2']))
                    y2  = int(float(row['y2']))
                except:
                    continue

                # Estimate motion direction from trajectory history
                traj = Vis_tra[Vis_tra['ID'] == tid].sort_values('timestamp')
                if len(traj) >= 3:
                    dx = float(traj['x'].iloc[-1]) - float(traj['x'].iloc[-3])
                    dy = float(traj['y'].iloc[-1]) - float(traj['y'].iloc[-3])
                elif len(traj) >= 2:
                    dx = float(traj['x'].iloc[-1]) - float(traj['x'].iloc[-2])
                    dy = float(traj['y'].iloc[-1]) - float(traj['y'].iloc[-2])
                else:
                    continue

                if abs(dx) < 0.5 and abs(dy) < 0.5:
                    continue   # ship not moving in frame

                roi = self._roi_behind(x1, y1, x2, y2, dx, dy, frame.shape)
                if roi is None:
                    continue
                rx1, ry1, rx2, ry2 = roi

                # Detect feature points in ROI of previous frame
                roi_prev = self.prev_gray[ry1:ry2, rx1:rx2]
                pts = cv2.goodFeaturesToTrack(roi_prev, **self.FEATURE_PARAMS)
                if pts is None or len(pts) < 3:
                    continue

                # Shift to full-frame coordinates
                pts[:, 0, 0] += rx1
                pts[:, 0, 1] += ry1

                # Track points to current frame
                pts_new, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, pts, None, **self.LK_PARAMS)

                good_old = pts[status.ravel() == 1]
                good_new = pts_new[status.ravel() == 1]
                if len(good_new) < 2:
                    continue

                # Mean flow vector in pixel space
                flow = good_new - good_old
                mean_fx = float(np.mean(flow[:, 0, 0]))
                mean_fy = float(np.mean(flow[:, 0, 1]))

                # Convert pixel flow direction to compass bearing
                # pixel +x = right, screen +y = down
                # camera points at shoot_hdir degrees
                # bearing ≈ shoot_hdir + atan2(flow_x, -flow_y)
                wake_px_angle  = math.degrees(math.atan2(mean_fx, -mean_fy))
                wake_compass   = (self.shoot_hdir + wake_px_angle) % 360
                wake[tid]      = round(wake_compass, 1)

        self.prev_gray   = gray.copy()
        self.wake_angles = wake
        return wake



def box_whether_in_area(bounding_box, Area):
    x_center = (bounding_box[0] + bounding_box[2]) / 2
    y_center = (bounding_box[1] + bounding_box[3]) / 2
    Area = [1] + Area # 添加一个虚拟id，为了使用whether函数
    # 中心点是否落在Area内
    return whether_in_area((x_center, y_center), Area)

def speed_extract(last_traj, now_traj):
    """
    :param last_traj: 若干秒前的轨迹数据
    :param now_traj: 当前时刻轨迹数据
    :return: 【水平速度， 垂直速度】
    """
    last_x = int(last_traj.loc['x'])
    last_y = int(last_traj.loc['y'])
    cur_x = int(now_traj.loc['x'])
    cur_y = int(now_traj.loc['y'])
    x_speed = (cur_x - last_x) / max(1, int(now_traj.loc['timestamp']) - int(last_traj.loc['timestamp']))
    y_speed = (cur_y - last_y) / max(1, int(now_traj.loc['timestamp']) - int(last_traj.loc['timestamp']))
    return [x_speed, y_speed]

def whether_in_area(point, bbox):
    """
    :param point: [x, y]
    :param bbox: [id,x1,y1,x2,y2]
    """
    if point[0] <= bbox[3] and point[0] >= bbox[1] and point[1] <= bbox[4] and point[1] >= bbox[2]:
        return 1
    else:
        return 0

def overlap(box1, box2, val):
    # 判断两个矩形是否相交
    # 思路来源于:https://www.cnblogs.com/avril/archive/2013/04/01/2993875.html
    # 然后把思路写成了代码
    minx1, miny1, maxx1, maxy1 = box1
    minx2, miny2, maxx2, maxy2 = box2
    minx = max(minx1, minx2)
    miny = max(miny1, miny2)
    maxx = min(maxx1, maxx2)
    maxy = min(maxy1, maxy2)
    if minx > maxx or miny > maxy:
        return 0
    else:
        max_x1 = max(minx1, minx2) # x1的最大值
        min_x2 = min(maxx1, maxx2) # x2的最小值
        max_y1 = max(miny1, miny2) # y1的最大值
        min_y2 = min(maxy1, maxy2)  # y2的最小值
        Cross_area = (min_x2 - max_x1) * (min_y2 - max_y1)
        box1_area = (maxx1 - minx1) * (maxy1 - miny1)
        box2_area = (maxx2 - minx2) * (maxy2 - miny2)
        if box1_area > 0 and box2_area > 0 and (Cross_area / box1_area > val or Cross_area / box2_area > val):
            return 1
        else:
            return 0
            
def whether_occlusion(bbox, cur_bbox_list, val):
    """
    :param bbox: [id,x1,y1,x2,y2]
    :param cur_bbox_list: [bbox1, bbox2,...]
    :param matched_id_list: [id1, id2,...]
    :return: flag, OAR
    """
    occlusion_bbox_list = []
    occlusion_id_list = []
    for i in range(len(cur_bbox_list)):
        # 判断这个bbox与剩下的bbox是否有遮挡
        flag = overlap(bbox[1:], cur_bbox_list[i][1:], val)
        if flag:
            if len(occlusion_id_list) == 0:
                occlusion_id_list.append(bbox[0])
                occlusion_bbox_list.append(bbox[1:])
            occlusion_bbox_list.append(cur_bbox_list[i][1:])
            occlusion_id_list.append(cur_bbox_list[i][0])
            break
    return occlusion_bbox_list, occlusion_id_list

def whether_in_OAR(point, OAR_list):
    flag = 0
    for oar in OAR_list:
        oar_id = [0, oar[0], oar[1], oar[2], oar[3]]
        if whether_in_area(point, oar_id):
            flag = whether_in_area(point, oar_id)
            break
    return flag


def OAR_extractor(his_traj_dataframe_list,val):
    # 1. 初始化遮挡区域和id列表
    OAR_list = []
    OAR_id_list = []
    # 2. 如果是第一帧则不做处理
    if len(his_traj_dataframe_list) == 0:
        return OAR_list, OAR_id_list
    # 3. 提取上一时刻的跟踪结果
    his_id_list = his_traj_dataframe_list[-1]['ID'].unique()
    his_bbox_list = []
    for i in range(len(his_id_list)):
        visual_traj = his_traj_dataframe_list[-1].iloc[i]
        his_bbox_list.append([visual_traj['ID'], visual_traj['x1'], visual_traj['y1'], visual_traj['x2'],
                              visual_traj['y2']])
    # 提取有历史纪录的、存在遮挡的船舶检测框及对应id，表示当前遮挡区域
    for i in range(len(his_bbox_list)):
        if i < len(his_bbox_list) - 1:
            occlusion_boxes, occlusion_ids = whether_occlusion(his_bbox_list[i], his_bbox_list[i + 1:], val)
            for index in range(len(occlusion_boxes)):
                if (occlusion_ids[index] not in OAR_id_list) and (occlusion_ids[index] in his_id_list):
                    OAR_list.append(occlusion_boxes[index])
                    OAR_id_list.append(occlusion_ids[index])
    return OAR_list, OAR_id_list

def motion_features_extraction(his_traj_dataframe_list, VIS_tra_cur):
    """
    :param his_traj_dataframe_list: 过去五秒内每秒的视觉轨迹数据
    :param VIS_tra_cur: 当前秒的视觉轨迹数据
    :return:
    """
    speed_list = []
    VIS_traj_cur_withfeature = VIS_tra_cur.copy()
    cur_id_list = VIS_tra_cur['ID'].unique()
    for i in range(len(cur_id_list)):
        speed_list.append('[0, 0]')
    VIS_traj_cur_withfeature['speed'] = speed_list
    for k in range(len(cur_id_list)):
        if len(his_traj_dataframe_list) == 0:
            #VIS_tra_cur_withfeature.iloc[k].loc['speed'] = [0, 0]
            continue
        id = cur_id_list[k]
        for i in his_traj_dataframe_list:
            his_id_list = list(i['ID'].unique())
            if id not in his_id_list:
                #VIS_tra_cur_withfeature.iloc[k].loc['speed'] = [0, 0]
                continue
            else:
                index = his_id_list.index(id)
                last_traj = i.iloc[index]
                VIS_traj_cur_withfeature.loc[k, 'speed'] = str(speed_extract(last_traj, VIS_traj_cur_withfeature.iloc[k]))
                break
    return VIS_traj_cur_withfeature

# 判断某一ID是否在过去五秒内的视觉轨迹内存在
def id_whether_stable(id, last_5_trajs):
    for traj in last_5_trajs:
        if id in list(traj['ID'].unique()):
            continue
        else:
            return False
    return True

# 目标检测跟踪
# Remplacez la classe VISPRO entière par ceci :

class VISPRO(object):
    def __init__(self, anti, val, t, ais_enabled=True):
        self.anti = anti
        self.last5_vis_tra_list = []
        self.Vis_tra_cur_3      = pd.DataFrame(columns=['ID','x1','y1','x2','y2','x','y','timestamp'])
        self.Vis_tra_cur        = pd.DataFrame(columns=['ID','x1','y1','x2','y2','x','y','timestamp'])
        self.Vis_tra            = pd.DataFrame(columns=['ID','x1','y1','x2','y2','x','y','timestamp'])
        self.VIS_tra_last = pd.DataFrame(columns=['ID','x1','y1','x2','y2','x','y', 'speed','timestamp'])
        self.OAR_list = []
        self.OAR_ids_list = []
        self.OAR_mmsi_list = []
        self.val = val
        self.t = t
        self.Anti_occlusion_traj = pd.DataFrame(columns=['ID','x1','y1','x2','y2','x','y','speed','timestamp'])
        # Track how many seconds each ID has been in OAR — clears ghost boxes
        self.OAR_age = {}        # {track_id: seconds_in_oar}
        self.max_occ_age = 4     # drop prediction after this many seconds
        # YOLOv8 maritime detections (display only, not in fusion)
        # Each entry: (x1,y1,x2,y2,cls,conf,expire_ms) — expires after 1 second
        self._maritime_bboxes_timed = []   # internal with expiry
        self.maritime_bboxes = []           # current valid boxes (no expiry field) for drawing/logging
        self.maritime_bboxes_suppressed = []  # kept for logger
        self.ais_enabled = ais_enabled  # AJOUT

    def detection(self, image):
        """
        YOLOX vessel detector — feeds DeepSORT tracking and AIS fusion.
        YOLOv8 maritime objects are handled separately in feedCap().
        """
        im0 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im0_pil = Image.fromarray(im0)
        return yolo.detect_image(im0_pil)

    @staticmethod
    def _iou(a, b):
        """IoU between two boxes (x1,y1,x2,y2)."""
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        if inter == 0: return 0.0
        area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
        area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    def track(self, image, bboxes, bboxes_anti_occ, id_list, timestamp):
        """
        bboxes: 检测结果（不含遮挡区域）
        bboxes_anti_occ: 遮挡区域抗遮挡推算
        id_list: 抗遮挡推算对应ID
        """
        # 用于目标跟踪
        bbox_xywh, confs = [], []
        bbox_xywh_anti_occ, confs_anti_occ = [], []
        if len(bboxes) or len(bboxes_anti_occ):
            # 检测结果整理
            for x1, y1, x2, y2, _, conf in bboxes:
                #获取框信息 [中心点x坐标，中心点y坐标，宽度，高度]
                obj = [int((x1+x2)/2), int((y1+y2)/2),x2-x1, y2-y1]
                bbox_xywh.append(obj)#框信息
                confs.append(conf)#置信度
            # 抗遮挡预测结果整理
            for x1, y1, x2, y2, _, conf in bboxes_anti_occ:
                #获取框信息 [中心点x坐标，中心点y坐标，宽度，高度]
                obj = [int((x1+x2)/2), int((y1+y2)/2),x2-x1, y2-y1]
                bbox_xywh_anti_occ.append(obj)#框信息
                confs_anti_occ.append(conf)#置信度

            # 检测结果、抗遮挡预测结果转tensor
            xywhs = torch.Tensor(bbox_xywh)
            confss = torch.Tensor(confs)
            xywhs_anti_occ = torch.Tensor(bbox_xywh_anti_occ)
            confss_anti_occ = torch.Tensor(confs_anti_occ)
            # 放入DeepSORT, 输出outputs = [x1,y1,x2,y2,[track],ID]
            outputs = deepsort.update(xywhs, confss, image, xywhs_anti_occ, confss_anti_occ, id_list, timestamp)
            for value in list(outputs):
                x1, y1, x2, y2, _, track_id = value
                if track_id in id_list:
                    x1, y1, x2, y2, _, _ = bboxes_anti_occ[id_list.index(track_id)] # 要把没有历史纪录的id从id——list中删掉
                # 存储至pd中[ID,x1,y1,x2,y2,trackx,tracky,time]
                self.Vis_tra_cur_3 = pd.concat([self.Vis_tra_cur_3, pd.DataFrame([{'ID':track_id,\
                    'x1':int(x1),'y1':int(y1),'x2':int(x2),'y2':int(y2),'x':int((x1 + x2) / 2),\
                        'y':int((y1 + y2) / 2), 'timestamp':timestamp//1000}])], ignore_index=True)

    def update_tra(self, Vis_tra, timestamp):
        # 用于轨迹更新
        self.Vis_tra_cur = pd.DataFrame(columns=['ID','x1','y1','x2','y2','x','y','timestamp'])
        id_list = self.Vis_tra_cur_3['ID'].unique()
        for k in range(len(id_list)):
            id_current = self.Vis_tra_cur_3[self.Vis_tra_cur_3['ID'] == id_list[k]].reset_index(drop=True)
            # 求取均值
            df = id_current.mean().astype(int)
            df['timestamp'] = timestamp // 1000
            self.Vis_tra_cur = pd.concat([self.Vis_tra_cur, df.to_frame().T], ignore_index=True)
        self.Vis_tra_cur_3 = pd.DataFrame(columns=['ID','x1','y1','x2','y2','x','y','timestamp'])

        Vis_tra_cur_withfeature = motion_features_extraction(self.last5_vis_tra_list, VIS_tra_cur= self.Vis_tra_cur)
        self.Vis_tra = pd.concat([self.Vis_tra, Vis_tra_cur_withfeature])
        if len(self.last5_vis_tra_list) > 4:
            self.last5_vis_tra_list.pop(0)
        self.last5_vis_tra_list.append(Vis_tra_cur_withfeature)
        # 删除时间过长的数据  时间以2分钟为限
        time_limited = 2
        self.Vis_tra = self.Vis_tra.drop(self.Vis_tra[self.Vis_tra['timestamp'] <\
                                                      (timestamp // 1000 - time_limited * 60)].index)
        return Vis_tra_cur_withfeature

    def traj_prediction_via_visual(self, last_traj, timestamp, speed):
        """
        :param last_traj: 若干秒的轨迹
        :return:
        """
        Vis_tra_prediction = last_traj.copy()
        x_move = int(timestamp - last_traj.loc['timestamp']) * float(speed[0])
        y_move = int(timestamp - last_traj.loc['timestamp']) * float(speed[1])
        Vis_tra_prediction.loc['x'] = Vis_tra_prediction.loc['x'] + x_move
        Vis_tra_prediction.loc['x1'] = Vis_tra_prediction.loc['x1'] + x_move
        Vis_tra_prediction.loc['x2'] = Vis_tra_prediction.loc['x2'] + x_move
        Vis_tra_prediction.loc['y'] = Vis_tra_prediction.loc['y'] + y_move
        Vis_tra_prediction.loc['y1'] = Vis_tra_prediction.loc['y1'] + y_move
        Vis_tra_prediction.loc['y2'] = Vis_tra_prediction.loc['y2'] + y_move
        Vis_tra_prediction.loc['timestamp'] = timestamp

        return Vis_tra_prediction

    def anti_occ(self, last5_vis_tra_list, bboxes, AIS_vis, bind_inf,timestamp):
        # 1.参数初始化
        bboxes_anti_occ = []
        if len(self.OAR_list):
            # 2. 删除处在OAR内的检测结果
            pop_index_list = []
            for index in range(len(bboxes)):
                for OAR in self.OAR_list:
                    if box_whether_in_area(bboxes[index][:4], OAR):
                        pop_index_list.append(index)
                        break
            for pop_index in range(len(pop_index_list)):
                bboxes.pop(pop_index_list[pop_index] - pop_index)
            
            # 所有遮挡id的mmsi提取
            bind_id_list = list(bind_inf['ID'].unique())
            self.OAR_mmsi_list = []
            OAR_ids_list_copy = self.OAR_ids_list.copy()
            for k in range(len(OAR_ids_list_copy)):
                if OAR_ids_list_copy[k] in bind_id_list:
                    mmsi = bind_inf.iloc[bind_id_list.index(OAR_ids_list_copy[k])].loc['mmsi']
                    self.OAR_mmsi_list.append([OAR_ids_list_copy[k], int(mmsi)])
                else:
                    self.OAR_mmsi_list.append([OAR_ids_list_copy[k], 0])

            # 预测bbox位置
            ais_vis_mmsi_list = list(AIS_vis['mmsi'])
            pop_index_list = []
            for k in range(len(self.OAR_mmsi_list)):
                final_find_flg = 0 # 是否找到最后时刻的位置
                second_final_find_flg = 0 # 是否找到前一时刻的位置
                final_pos = [] # 最新时刻的AIS投影位置
                second_final_pos = [] # 上一时刻的AIS投影位置
                # 若存在MMSI
                if not self.OAR_mmsi_list[k][1] == 0 and self.OAR_mmsi_list[k][1] in ais_vis_mmsi_list:
                    for i in range(len(ais_vis_mmsi_list)):
                        # 找到这条mmsi最后位置
                        # 对于有MMSI的船，用AIS预测位置
                        if int(AIS_vis.iloc[len(ais_vis_mmsi_list) - i - 1].loc['mmsi']) == self.OAR_mmsi_list[k][1] and \
                                int(AIS_vis.iloc[len(ais_vis_mmsi_list) - i - 1].loc['timestamp']) == timestamp - 1:
                            final_find_flg = 1
                            final_pos = [AIS_vis.iloc[len(ais_vis_mmsi_list) - i - 1].loc['x'],
                                         AIS_vis.iloc[len(ais_vis_mmsi_list) - i - 1].loc['y']]
                            continue
                        # 找到这条mmsi前一个位置
                        elif int(AIS_vis.iloc[len(ais_vis_mmsi_list) - i - 1].loc['mmsi']) == self.OAR_mmsi_list[k][
                            1] and int(AIS_vis.iloc[len(ais_vis_mmsi_list) - i - 1].loc['timestamp']) == timestamp - 2:
                            second_final_find_flg = 1
                            second_final_pos = [AIS_vis.iloc[len(ais_vis_mmsi_list) - i - 1].loc['x'],
                                                AIS_vis.iloc[len(ais_vis_mmsi_list) - i - 1].loc['y']]
                            continue
                        # 位置作差，得出位移向量，作用在视觉目标上
                        if final_find_flg and second_final_find_flg:
                            x_motion = final_pos[0] - second_final_pos[0]
                            y_motion = final_pos[1] - second_final_pos[1]
                            # 预测结果添加到抗遮挡检测框中
                            bboxes_anti_occ.append(
                                (self.Anti_occlusion_traj.iloc[k].loc['x1'] + x_motion,
                                 self.Anti_occlusion_traj.iloc[k].loc['y1'] + y_motion,
                                 self.Anti_occlusion_traj.iloc[k].loc['x2'] + x_motion,
                                 self.Anti_occlusion_traj.iloc[k].loc['y2'] + y_motion,
                                 'vessel', 1))  # 预测框生成
                            break
                # 若不存在MMSI
                else:
                    # 对于没有MMSI的船，用视觉预测位置
                    # 找到n秒前的轨迹点
                    # 如果过去五秒内轨迹不稳定，不预测
                    if not id_whether_stable(self.OAR_mmsi_list[k][0], last5_vis_tra_list):
                        pop_index_list.append(k)
                        continue
                    index = list(last5_vis_tra_list[0]['ID'].unique()).index(self.OAR_mmsi_list[k][0])
                    # 获取轨迹速度
                    speed_str = last5_vis_tra_list[0].iloc[index].loc['speed']
                    speed = [float(speed_str[1:-1].split(',')[0]), float(speed_str[1:-1].split(',')[1])]
                    # 轨迹速度作用在n秒前的轨迹点上
                    trajs = last5_vis_tra_list[0]
                    id_list = list(trajs['ID'].unique())
                    last_traj = trajs.iloc[id_list.index(self.OAR_mmsi_list[k][0])]
                    Vis_traj_now = self.traj_prediction_via_visual(last_traj, timestamp, speed)
                    # 添加到抗遮挡检测结果中
                    bboxes_anti_occ.append(
                        (Vis_traj_now.loc['x1'],
                         Vis_traj_now.loc['y1'],
                         Vis_traj_now.loc['x2'],
                         Vis_traj_now.loc['y2'],
                         'vessel', 1))

                # 删除既没有五秒历史数据，又没有AIS的目标
            for i in range(len(pop_index_list)):
                self.OAR_mmsi_list.pop(pop_index_list[i] - i)
                self.OAR_ids_list.pop(pop_index_list[i] - i)
                self.OAR_list.pop(pop_index_list[i] - i)
            if not len(self.OAR_ids_list) == len(bboxes_anti_occ):
                embed()
        return bboxes_anti_occ

        
    def feedCap(self, image, timestamp, AIS_vis, bind_inf, ais_enabled=True,
                use_ensemble=True, use_static_filter=True):
        # Mise à jour de l'état AIS enabled à chaque frame
        self.ais_enabled = ais_enabled
        
        # 情况1: 当前时刻需要进行检测
        if timestamp % 1000 < self.t:
            
            # 0. Preprocessing: CLAHE + weather-adaptive threshold
            image, brightness, sharpness, condition = preprocess_frame(image)
            
            # 1.1.目标检测框生成 — YOLOX vessels (→ tracking + fusion)
            bboxes = self.detection(image)

            # 1.1b. YOLOv8 KOLOMVERSE — maritime objects (display only)
            # Suppressed boxes (overlap with YOLOX) are discarded entirely.
            # Kept boxes expire after 1000 ms.
            self.maritime_bboxes_suppressed = []
            expire_at = timestamp + self.t  # one frame duration

            if use_ensemble and _yolov8 is not None:
                try:
                    raw_maritime = _yolov8.detect_image(image)

                    shore_y = _detect_shoreline(image)
                    BUILDING_MARGIN = 70

                    new_timed = []
                    for mb in raw_maritime:
                        if use_static_filter:
                            if _is_static_structure(mb, shore_y, BUILDING_MARGIN):
                                self.maritime_bboxes_suppressed.append(mb)
                                continue
                        
                        # NMS with YOLOX (inline IoU check)
                        if any(self._iou(mb[:4], vb[:4]) > 0.4 for vb in bboxes):
                            self.maritime_bboxes_suppressed.append(mb)
                            continue
                        
                        new_timed.append((*mb, expire_at))

                    self._maritime_bboxes_timed = [
                        b for b in self._maritime_bboxes_timed
                        if b[6] > timestamp
                    ] + new_timed
                except Exception:
                    pass  # YOLOv8 failure never blocks vessel tracking
            else:
                # use_ensemble=False or _yolov8 not loaded: no maritime detections
                self._maritime_bboxes_timed = [
                    b for b in self._maritime_bboxes_timed if b[6] > timestamp
                ]
            
            # Expire old boxes even if YOLOv8 didn't run this frame
            self._maritime_bboxes_timed = [
                b for b in self._maritime_bboxes_timed if b[6] > timestamp
            ]
            # Expose without expiry field for drawing/logging
            self.maritime_bboxes = [b[:6] for b in self._maritime_bboxes_timed]

            # 1.2.抗遮挡
            bboxes_anti_occ = self.anti_occ(self.last5_vis_tra_list, bboxes, AIS_vis, bind_inf, timestamp // 1000)

            # 1.3.DeepSORT跟踪
            # print(bboxes_anti_occ)
            self.track(image, bboxes, bboxes_anti_occ=bboxes_anti_occ,\
                    id_list=self.OAR_ids_list, timestamp=timestamp // 1000)

            # 轨迹数据更新
            Vis_tra_cur = self.Vis_tra_cur
            if timestamp % 1000 < self.t:
                Vis_tra_cur = self.update_tra(self.Vis_tra, timestamp)
                if self.anti:
                    # 根据上一时刻跟踪结果，提取出存在AIS的遮挡重叠船舶框以及对应ID
                    self.OAR_list, self.OAR_ids_list = OAR_extractor(self.last5_vis_tra_list, self.val)

                # ── Ghost box prevention ──────────────────────────────────
                # Update age counter for each OAR id
                current_oar_set = set(self.OAR_ids_list)
                for tid in list(self.OAR_age.keys()):
                    if tid not in current_oar_set:
                        del self.OAR_age[tid]   # ship left OAR — reset
                for tid in current_oar_set:
                    self.OAR_age[tid] = self.OAR_age.get(tid, 0) + 1

                # Drop any OAR entry that has been predicting too long
                expired = [tid for tid, age in self.OAR_age.items()
                           if age > self.max_occ_age]
                for tid in expired:
                    if tid in self.OAR_ids_list:
                        idx = self.OAR_ids_list.index(tid)
                        self.OAR_ids_list.pop(idx)
                        self.OAR_list.pop(idx)
                    del self.OAR_age[tid]
                # ─────────────────────────────────────────────────────────
                    # print("OAR_id_list", self.OAR_ids_list)
                self.VIS_tra_last = Vis_tra_cur

                # 更新被遮挡id对应的轨迹数据
                self.Anti_occlusion_traj = pd.DataFrame(columns=['ID', 'x1', 'y1', 'x2', 'y2', 'x', 'y', 'speed', 'timestamp'])
                id_list = list(self.VIS_tra_last['ID'].unique())
                for i in self.OAR_ids_list:
                    self.Anti_occlusion_traj = pd.concat([self.Anti_occlusion_traj, self.VIS_tra_last.iloc[id_list.index(i)].to_frame().T])
        
        return self.Vis_tra, self.Vis_tra_cur