"""
modules/pipeline/fusion_processor.py — AIS-Visual fusion processor.
Standalone module that can be used independently of the full pipeline.
"""

import math
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import linear_sum_assignment as linear_assignment

try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    FASTDTW_OK = True
except ImportError:
    FASTDTW_OK = False


class FusionProcessor:
    """
    Standalone AIS-Visual fusion processor.
    Extracted from utils/FUS_utils.py for modular use.

    Parameters
    ----------
    max_distance : float
        Maximum matching distance in pixels
    image_shape : list
        [width, height]
    frame_interval_ms : int
        Milliseconds between frames
    """

    def __init__(self, max_distance=None, image_shape=None, frame_interval_ms=33):
        self.max_dis = max_distance or 3000
        self.im_shape = image_shape or [1920, 1080]
        self.bin_num = 1       # binding threshold
        self.fog_num = 15      # forgetting threshold (seconds)
        self.t = frame_interval_ms

        self.mat_cur = pd.DataFrame(columns=['ID/mmsi', 'timestamp', 'match'])
        self.bin_cur = pd.DataFrame(columns=['ID', 'mmsi', 'timestamp', 'match'])

    @staticmethod
    def _angle(v1, v2):
        """Compute included angle between two trajectory motion vectors."""
        dx1 = v1[-1][0] - (v1[-10][0] if len(v1) >= 10 else v1[0][0])
        dy1 = v1[-1][1] - (v1[-10][1] if len(v1) >= 10 else v1[0][1])
        dx2 = v2[-1][0] - v2[0][0]
        dy2 = v2[-1][1] - v2[0][1]
        angle1 = math.atan2(dy1, dx1)
        angle2 = math.atan2(dy2, dx2)
        if angle1 * angle2 >= 0:
            return abs(angle1 - angle2)
        inc = abs(angle1) + abs(angle2)
        return math.pi * 2 - inc if inc > math.pi else inc

    @staticmethod
    def _dtw_distance(traj0, traj1):
        """DTW distance with trajectory compression and angle penalty."""
        if not FASTDTW_OK:
            return 999999.0
        theta = 0
        if len(traj0) > 1 and len(traj1) > 1:
            theta = FusionProcessor._angle(traj0, traj1)
            traj0 = [(traj0[i] + traj0[1+i]) / 2 for i in range(0, len(traj0) - len(traj0) % 2, 2)]
            traj1 = [(traj1[i] + traj1[1+i]) / 2 for i in range(0, len(traj1) - len(traj1) % 2, 2)]
        d, _ = fastdtw(traj0, traj1, dist=euclidean)
        return d * math.exp(theta)

    @staticmethod
    def confidence_score(match_count):
        """Sigmoid confidence [0..1] from consecutive match count."""
        return round(1.0 - math.exp(-match_count / 12.0), 3)

    def build_cost_matrix(self, ais_list, vis_list, bin_las, ais_inf_list=None):
        """Build similarity cost matrix for Hungarian assignment."""
        matrix = np.zeros((len(vis_list), len(ais_list)))

        bin_ids = {}
        if len(bin_las) > 0:
            for _, row in bin_las.iterrows():
                parts = row['ID/mmsi'].split('/')
                bin_ids[(int(parts[0]), int(parts[1]))] = int(row['match'])

        for i in range(len(vis_list)):
            for j in range(len(ais_list)):
                vis_id = int(ais_list[j][0][0]) if hasattr(ais_list[j][0], '__getitem__') else i
                ais_mmsi = int(ais_list[j][0][0]) if hasattr(ais_list[j][0], '__getitem__') else j

                # Check if pair is already locked
                if (vis_id, ais_mmsi) in bin_ids:
                    matrix[i][j] = -bin_ids[(vis_id, ais_mmsi)] * 100
                    continue

                # Compute DTW distance
                theta = self._angle(vis_list[i], ais_list[j])
                if theta > math.pi * (5/6):
                    matrix[i][j] = 1e9
                    continue

                matrix[i][j] = self._dtw_distance(vis_list[i], ais_list[j])

        return matrix

    def match(self, ais_trajectories, vis_trajectories, ais_inf_list=None, vis_inf_list=None):
        """
        Run trajectory matching and return fused results.

        Parameters
        ----------
        ais_trajectories : list of np.array
            AIS trajectory position sequences
        vis_trajectories : list of np.array
            Visual trajectory position sequences

        Returns
        -------
        list of dict with matched pairs and confidence
        """
        if not ais_trajectories or not vis_trajectories:
            return []

        matrix = self.build_cost_matrix(
            ais_trajectories, vis_trajectories, self.bin_cur, ais_inf_list
        )

        if matrix.size == 0:
            return []

        row_ind, col_ind = linear_assignment(matrix)

        matches = []
        for row, col in zip(row_ind, col_ind):
            cost = matrix[row][col]
            if cost < 1e8:
                conf = self.confidence_score(int(self.bin_cur.shape[0]) + 1)
                matches.append({
                    "vis_index": int(row),
                    "ais_index": int(col),
                    "cost": float(cost),
                    "confidence": conf
                })

        return matches

    def reset(self):
        """Reset internal state for a new video."""
        self.mat_cur = pd.DataFrame(columns=['ID/mmsi', 'timestamp', 'match'])
        self.bin_cur = pd.DataFrame(columns=['ID', 'mmsi', 'timestamp', 'match'])
