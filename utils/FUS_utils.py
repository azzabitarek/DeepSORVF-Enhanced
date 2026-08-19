import time
from fastdtw import fastdtw
import pandas as pd
from scipy.spatial.distance import euclidean
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment as linear_assignment
from IPython import embed 

def __reduce_by_half(x):
    # Trajectory compression — average adjacent pairs
    return [(x[i] + x[1+i]) / 2 for i in range(0, len(x) - len(x) % 2, 2)]


def _greedy_assignment(cost_matrix):
    """
    Greedy assignment for ablation 'no Hungarian'.
    Sort all (i,j) pairs by cost ascending, assign greedily.
    """
    n_rows, n_cols = cost_matrix.shape
    pairs = [(cost_matrix[i, j], i, j) for i in range(n_rows) for j in range(n_cols)]
    pairs.sort(key=lambda x: x[0])
    used_rows, used_cols = set(), set()
    row_ind, col_ind = [], []
    for cost, i, j in pairs:
        if i not in used_rows and j not in used_cols:
            row_ind.append(i)
            col_ind.append(j)
            used_rows.add(i)
            used_cols.add(j)
    return np.array(row_ind), np.array(col_ind)


def heading_penalty(ais_heading_deg, vis_traj, ais_traj):
    """
    Mild penalty [1.0 .. 1.5] based only on trajectory direction agreement.
    DTW_fast already penalises angle via exp(theta) — we only add a small
    extra boost for ships whose trajectories clearly point in opposite directions,
    which is a strong sign of a mis-match.

    We do NOT compare compass heading to pixel-space direction (different
    coordinate frames) — that comparison is meaningless.
    """
    try:
        theta = angle([list(p) for p in vis_traj],
                      [list(p) for p in ais_traj])   # radians [0, pi]
        # Only penalise when trajectories are nearly opposite (>120 deg)
        if theta > math.pi * (2/3):
            return 1.5
        return 1.0
    except:
        return 1.0


def confidence_score(match_count):
    """
    Sigmoid-style confidence [0.0 .. 1.0] from consecutive match count.
    Reaches 0.5 at 5 matches, 0.9 at 20 matches, ~1.0 at 40+ matches.
    """
    return round(1.0 - math.exp(-match_count / 12.0), 3)

def angle(v1, v2):
    # Compute the included angle between two trajectory motion vectors
    if len(v1) >= 10:
        dx1 = v1[-1][0] - v1[-10][0]
        dy1 = v1[-1][1] - v1[-10][1]
    elif len(v1) < 10:
        dx1 = v1[-1][0] - v1[0][0]
        dy1 = v1[-1][1] - v1[0][1]
    if len(v2) >= 5:
        dx2 = v2[-1][0] - v2[0][0]
        dy2 = v2[-1][1] - v2[0][1]
    elif len(v2) < 5:
        dx2 = v2[-1][0] - v2[0][0]
        dy2 = v2[-1][1] - v2[0][1]

    angle1 = math.atan2(dy1, dx1)
    angle2 = math.atan2(dy2, dx2)

    if angle1 * angle2 >= 0:
        included_angle = abs(angle1 - angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > math.pi:
            included_angle = math.pi * 2 - included_angle
    return included_angle

def DTW_fast(traj0, traj1, use_angle_penalty=True):
    # 1. Compute the included angle between the two trajectories
    if len(traj0) > 1 and len(traj1) > 1:
        theta = angle(traj0, traj1) if use_angle_penalty else 0.0
        traj0 = __reduce_by_half(traj0)
        traj1 = __reduce_by_half(traj1)
    else:
        theta = 0

    # 2. Run fastDTW
    d, path = fastdtw(traj0, traj1, dist=euclidean)

    return d * math.exp(theta)


def traj_group(df_data, df_dataCur, kind):
    """
    Groups trajectory data by MMSI (AIS) or track ID (visual) and extracts
    the trajectory for each vessel or detection box.
    :param df_data:    full AIS or visual trajectory history
    :param df_dataCur: current-frame AIS or visual data
    :param kind:       'AIS' or 'VIS'
    :return: trajData_list, trajLabel_list, trajInf_list
    """
    # 1. Initialise output lists
    trajData_list  = []   # stores (x, y) position sequences
    trajLabel_list = []   # stores MMSI or track ID labels
    trajInf_list   = []   # stores full row data for each trajectory

    # 2. Group AIS data by MMSI
    if kind == 'AIS':
        grouped = df_data.groupby('mmsi')
        for value, group in grouped:
            # Only include vessels present at the current timestamp
            if value in df_dataCur['mmsi'].tolist():
                traj = group.values
                trajData_list.append(np.array(traj[:, 7:9]))
                trajLabel_list.append(int(traj[0, 0]))
                trajInf_list.append(traj)

    # 3. Group visual data by track ID
    elif kind == 'VIS':
        grouped = df_data.groupby('ID')
        for value, group in grouped:
            # Only include tracks present at the current timestamp
            if value in df_dataCur['ID'].tolist():
                traj = group.values
                trajData_list.append(np.array(traj[:, 5:7]))
                trajLabel_list.append(int(traj[0][0]))
                trajInf_list.append(traj)

    return trajData_list, trajLabel_list, trajInf_list

class FUSPRO(object):
    def __init__(self, max_dis, im_shape, t,
                 use_dtw=True, use_angle_penalty=True,
                 use_binding=True, use_hungarian=True):
        # Maximum matching distance (pixels)
        self.max_dis  = max_dis
        self.im_shape = im_shape
        # Number of consecutive matches required before a pair is locked (binding threshold)
        self.bin_num  = 1 if use_binding else float('inf')
        # Number of seconds a locked pair is kept alive without a new match (forgetting threshold)
        self.fog_num  = 15
        # Display duration per frame (ms)
        self.t        = t

        # --- Ablation flags ---
        self.use_dtw           = use_dtw
        self.use_angle_penalty = use_angle_penalty
        self.use_hungarian     = use_hungarian
        # Ablation log path (set by ablation runner to enable CSV logging)
        self._ablation_log_path = None

        # Data store 1: match records for the current timestamp
        self.mat_cur  = pd.DataFrame(pd.DataFrame(columns=['ID/mmsi', 'timestamp', 'match']))
        # Data store 2: fused output for the current timestamp
        self.mat_list = pd.DataFrame(columns=['ID', 'mmsi',
                                               'lon', 'lat', 'speed', 'course', 'heading', 'type',
                                               'timestamp', 'confidence'])
        # Data store 3: locked binding pairs for the current timestamp
        self.bin_cur  = pd.DataFrame(columns=['ID', 'mmsi', 'timestamp', 'match'])

    def initialization(self, AIS_list, VIS_list):
        # Reset per-frame working buffers
        mat_las  = self.mat_cur
        bin_las  = mat_las[mat_las['match'] > self.bin_num]
        mat_cur  = pd.DataFrame(pd.DataFrame(columns=['ID/mmsi', 'timestamp', 'match']))
        bin_cur  = pd.DataFrame(columns=['ID', 'mmsi', 'timestamp', 'match'])
        mat_list = pd.DataFrame(columns=['ID', 'mmsi',
                                          'lon', 'lat', 'speed', 'course', 'heading', 'type',
                                          'x1', 'y1', 'w', 'h', 'timestamp', 'confidence'])
        return mat_cur, bin_cur, mat_las, bin_las, mat_list

    def cal_similarity(self, AIS_list, AIS_MMSIlist, VIS_list, VIS_IDlist, bin_las, AInf_list=None):
        # 1. Initialise the similarity cost matrix
        matrix_S = np.zeros((len(VIS_list), len(AIS_list)))

        # 2. Extract currently locked (bound) ID/MMSI pairs from the previous frame
        binIDmmsi, bin_MMSI, bin_ID = [], [], []
        if len(bin_las) != 0:
            grouped = bin_las.groupby('ID/mmsi')
            for value, group in grouped:
                ID, MMSI = value.split('/')
                bin_ID.append(int(ID))
                bin_MMSI.append(int(MMSI))
                binIDmmsi.append(value)

        for i in range(len(VIS_list)):
            for j in range(len(AIS_list)):

                # 3. Get the current track ID and MMSI for this matrix cell
                cur_ID, cur_mmsi = VIS_IDlist[i], AIS_MMSIlist[j]
                cur_IDmmsi = str(int(cur_ID)) + '/' + str(int(cur_mmsi))

                # Case 1: No existing binding — compute FastDTW + heading penalty
                if int(cur_mmsi) not in bin_MMSI and int(cur_ID) not in bin_ID:
                    theta = angle(VIS_list[i], AIS_list[j]) if self.use_angle_penalty else 0.0
                    # Compute pixel-space distance between the latest positions
                    x_VIS = VIS_list[i][-1][0]
                    y_VIS = VIS_list[i][-1][1]
                    x_AIS = AIS_list[j][-1][0]
                    y_AIS = AIS_list[j][-1][1]
                    dis   = ((x_VIS - x_AIS) ** 2 + (y_VIS - y_AIS) ** 2) ** 0.5
                    # Only compute DTW if within distance and angle limits
                    if dis < self.max_dis and theta < math.pi * (5/6):
                        if self.use_dtw:
                            cost = DTW_fast(VIS_list[i], AIS_list[j],
                                             use_angle_penalty=self.use_angle_penalty)
                        else:
                            cost = dis
                        if self.use_angle_penalty:
                            try:
                                ais_heading = float(AInf_list[j][-1][5])
                            except:
                                ais_heading = 0.0
                            hp = heading_penalty(ais_heading, VIS_list[i], AIS_list[j])
                        else:
                            hp = 1.0
                        matrix_S[i][j] = cost * hp
                    else:
                        matrix_S[i][j] = 1000000000

                # Case 2: This exact pair is already locked — assign a strongly negative cost to force selection
                elif cur_IDmmsi in binIDmmsi:
                    matrix_S[i][j] = 0 - int(bin_las[bin_las['ID/mmsi'] == cur_IDmmsi]['match'].values) * 100

                # Case 3: One of the IDs is locked to a different partner — block this pair
                else:
                    matrix_S[i][j] = 1000000000
        return matrix_S

    def data_filter(self, row_ind, col_ind, VIS_list, AIS_list):
        # 1. Initialise match list
        matches = []

        # 2. Remove assignments that exceed distance or angle thresholds
        for row, col in zip(row_ind, col_ind):
            # Compute included angle between the two trajectories
            theta = angle(VIS_list[row], AIS_list[col])

            # Compute pixel-space distance between the latest positions
            x_VIS = VIS_list[row][-1][0]
            y_VIS = VIS_list[row][-1][1]
            x_AIS = AIS_list[col][-1][0]
            y_AIS = AIS_list[col][-1][1]
            dis   = ((x_VIS - x_AIS) ** 2 + (y_VIS - y_AIS) ** 2) ** 0.5

            # Keep the match only if within both limits
            if dis < self.max_dis and theta < math.pi * (5/6):
                matches.append((row, col))
        return matches

    def save_data(self, mat_cur, bin_cur, mat_las, bin_las, mat_list,
                  matches, AIS_MMSIlist, VIS_IDlist, AInf_list, VInf_list, timestamp):
        # 1. Store match records for the current timestamp
        for i in range(len(matches)):
            v_loc, a_loc = matches[i][0], matches[i][1]
            ID      = int(VIS_IDlist[v_loc])
            MMSI    = int(AIS_MMSIlist[a_loc])
            ID_MMSI = str(ID) + '/' + str(MMSI)

            lon     = AInf_list[a_loc][-1][1]
            lat     = AInf_list[a_loc][-1][2]
            speed   = AInf_list[a_loc][-1][3]
            course  = AInf_list[a_loc][-1][4]
            heading = AInf_list[a_loc][-1][5]
            types   = AInf_list[a_loc][-1][6]
            time    = AInf_list[a_loc][-1][9]

            x1 = max(VInf_list[v_loc][-1][1], 0)
            y1 = max(VInf_list[v_loc][-1][2], 0)
            x2 = min(VInf_list[v_loc][-1][3], self.im_shape[0])
            y2 = min(VInf_list[v_loc][-1][4], self.im_shape[1])
            w  = abs(x2 - x1)
            h  = abs(y2 - y1)

            # Compute confidence from consecutive match count
            if ID_MMSI in mat_las['ID/mmsi'].values:
                match = mat_las[mat_las['ID/mmsi'] == ID_MMSI]['match'].values[0] + 1
            else:
                match = 1
            conf = confidence_score(match)

            mat_list = pd.concat([mat_list, pd.DataFrame([{'ID': ID, 'mmsi': MMSI, 'lon': lon, 'lat': lat,
                                          'speed': speed, 'course': course, 'heading': heading,
                                          'type': types, 'x1': x1, 'y1': y1,
                                          'w': w, 'h': h, 'timestamp': time,
                                          'confidence': conf}])], ignore_index=True)

            # Case 1: Pair existed in the previous frame — increment match counter
            if ID_MMSI in mat_las['ID/mmsi'].values:
                mat_cur = pd.concat([mat_cur, pd.DataFrame([{'ID/mmsi': str(ID) + '/' + str(MMSI),
                                           'timestamp': time, 'match': match}])], ignore_index=True)
            # Case 2: New pair this frame — initialise match counter to 1
            else:
                mat_cur = pd.concat([mat_cur, pd.DataFrame([{'ID/mmsi': str(ID) + '/' + str(MMSI),
                                           'timestamp': time, 'match': 1}])], ignore_index=True)

        # Case 3: Pair was locked in a previous frame but absent this frame —
        #         carry forward if the MMSI is still in the scene and within the forgetting window
        for ind, inf in bin_las.iterrows():
            ID_MMSI  = inf['ID/mmsi']
            ID, MMSI = [int(x) for x in ID_MMSI.split('/')]
            time     = inf['timestamp']
            if MMSI in AIS_MMSIlist and ID_MMSI not in mat_cur['ID/mmsi'].values \
                    and timestamp // 1000 - time < self.fog_num:
                mat_cur = pd.concat([mat_cur, inf.to_frame().T], ignore_index=True)

        # 2. Promote matches that have exceeded the binding threshold to locked pairs
        for ind, inf in mat_cur.iterrows():
            ID, MMSI = [int(x) for x in inf['ID/mmsi'].split('/')]
            if inf['match'] > self.bin_num:
                bin_cur = pd.concat([bin_cur, pd.DataFrame([{'ID': ID, 'mmsi': MMSI,
                                           'timestamp': int(inf['timestamp']),
                                           'match': int(inf['match'])}])], ignore_index=True)

        # --- Ablation logging (optional) ---
        # Logs ALL matches from mat_cur (not just locked from bin_cur),
        # so entries are written even when FUS is freshly reinitialized.
        if self._ablation_log_path:
            import csv
            write_header = not os.path.exists(self._ablation_log_path)
            with open(self._ablation_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(['timestamp', 'ID', 'mmsi', 'match_count', 'is_new_lock'])
                for _, inf in mat_cur.iterrows():
                    ID_mmsi = inf['ID/mmsi']
                    ID_val, mmsi_val = [int(x) for x in ID_mmsi.split('/')]
                    writer.writerow([
                        int(inf['timestamp']),
                        ID_val,
                        mmsi_val,
                        int(inf['match']),
                        bool(inf['match'] == self.bin_num + 1)
                    ])

        return mat_list, mat_cur, bin_cur

    def traj_match(self, AIS_list, AIS_MMSIlist, VIS_list, VIS_IDlist, AInf_list, VInf_list, timestamp):
        # 1. Reset working buffers
        mat_cur, bin_cur, mat_las, bin_las, mat_list = self.initialization(AIS_list, VIS_list)

        # 2. Build similarity cost matrix
        matrix_S = self.cal_similarity(AIS_list, AIS_MMSIlist, VIS_list, VIS_IDlist, bin_las, AInf_list)

        # 3. Optimal assignment via Hungarian algorithm (or greedy for ablation)
        if self.use_hungarian:
            row_ind, col_ind = linear_assignment(matrix_S)
        else:
            row_ind, col_ind = _greedy_assignment(matrix_S)

        # 4. Filter out assignments that exceed distance / angle thresholds
        matches = self.data_filter(row_ind, col_ind, VIS_list, AIS_list)

        matric = pd.DataFrame(matrix_S, columns=AIS_MMSIlist, index=VIS_IDlist)

        # 5. Save match and binding data
        mat_list, mat_cur, bin_cur = self.save_data(mat_cur, bin_cur, mat_las, bin_las,
                                                     mat_list, matches, AIS_MMSIlist, VIS_IDlist,
                                                     AInf_list, VInf_list, timestamp)
        return mat_list, mat_cur, bin_cur

    def fusion(self, AIS_vis, AIS_cur, Vis_tra, Vis_cur, timestamp):
        if timestamp % 1000 < self.t:
            # 1. Extract and group trajectories
            AIS_list, AIS_MMSIlist, AInf_list = traj_group(AIS_vis, AIS_cur, 'AIS')
            VIS_list, VIS_IDlist, VInf_list   = traj_group(Vis_tra, Vis_cur, 'VIS')

            # 2. Run trajectory matching
            self.mat_list, self.mat_cur, self.bin_cur = self.traj_match(
                AIS_list, AIS_MMSIlist, VIS_list, VIS_IDlist, AInf_list, VInf_list, timestamp)

        return self.mat_list, self.bin_cur