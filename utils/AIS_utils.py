import pandas as pd
from geopy.distance import geodesic
import pyproj
from math import radians, cos, sin, asin, sqrt, tan, atan2, degrees
import math
import numpy as np
import cv2
from IPython import embed
import os

# ──────────────────────────────────────────────────────────────────────────────
# Kalman filter — smooths AIS pixel positions between updates
# State: [x, y, vx, vy], constant-velocity model
# ──────────────────────────────────────────────────────────────────────────────
class KalmanAIS:
    def __init__(self):
        self.x = None
        self.P = np.eye(4) * 200.0
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        self.Q = np.diag([0.5, 0.5, 2.0, 2.0])
        self.R = np.diag([8.0, 8.0])

    def update(self, x_meas, y_meas):
        if self.x is None:
            self.x = np.array([float(x_meas), float(y_meas), 0., 0.])
            return float(x_meas), float(y_meas)
        F   = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=float)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        z   = np.array([float(x_meas), float(y_meas)])
        S   = self.H @ self.P @ self.H.T + self.R
        K   = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return float(self.x[0]), float(self.x[1])


def count_distance(point1, point2, Type='m'):
    '''
    Computes the distance between two GPS points using lat/lon.
    point1: (longitude, latitude) of point 1
    point2: (longitude, latitude) of point 2
    Type: 'nm' for nautical miles; 'm' for metres
    Returns: distance in metres (or nautical miles if Type='nm')
    '''
    # Compute distance in metres between the two lat/lon coordinates
    distance = geodesic(point1, point2).m
    if Type == 'nm':
        # Convert metres to nautical miles
        distance = distance * 0.00054
    return distance


def getDegree(latA, lonA, latB, lonB):
    '''
    Computes the compass bearing from point A to point B.
    latA: camera latitude
    lonA: camera longitude
    latB: vessel latitude
    lonB: vessel longitude
    Returns: bearing in degrees (0-360)
    '''
    radLatA = radians(latA)
    radLonA = radians(lonA)
    radLatB = radians(latB)
    radLonB = radians(lonB)
    dLon = radLonB - radLonA
    y = sin(dLon) * cos(radLatB)
    x = cos(radLatA) * sin(radLatB) - sin(radLatA) * cos(radLatB) * cos(dLon)
    brng = degrees(atan2(y, x))
    brng = (brng + 360) % 360
    return brng

def visual_transform(lon_v, lat_v, camera_para, shape):
    '''
    Projects a GPS position onto the image coordinate system.
    lon_cam:    camera longitude
    lat_cam:    camera latitude
    shoot_vdir: camera vertical tilt angle (degrees downward)
    shoot_hdir: camera horizontal heading
    height_cam: camera height above water surface
    width_pic:  image width
    height_pic: image height
    FOV_hor:    horizontal field of view (e.g. 55 degrees)
    FOV_ver:    vertical field of view
    Returns: (target_x, target_y) pixel coordinates
    '''
    # Initialise camera parameters
    lon_cam    = camera_para[0]
    lat_cam    = camera_para[1]
    shoot_hdir = camera_para[2]
    shoot_vdir = camera_para[3]
    height_cam = camera_para[4]
    FOV_hor    = camera_para[5]
    FOV_ver    = camera_para[6]
    width_pic  = shape[0]
    height_pic = shape[1]
    f_x        = camera_para[7]
    f_y        = camera_para[8]
    u0         = camera_para[9]
    v0         = camera_para[10]

    # 1. Compute distance from camera to vessel
    D_abs = count_distance((lat_cam, lon_cam), (lat_v, lon_v))

    # 2. Compute horizontal angle between camera heading and vessel bearing
    relative_angle = getDegree(lat_cam, lon_cam, lat_v, lon_v)
    Angle_hor = relative_angle - shoot_hdir
    if Angle_hor < -180:
        Angle_hor = Angle_hor + 360
    elif Angle_hor > 180:
        Angle_hor = Angle_hor - 360
    hor_rad = radians(Angle_hor)
    shv_rad = radians(-shoot_vdir)
    Z_w = D_abs * cos(hor_rad)
    X_w = D_abs * sin(hor_rad)
    Y_w = height_cam
    Z = Z_w / cos(shv_rad) + (Y_w - Z_w * tan(shv_rad)) * sin(shv_rad)
    X = X_w
    Y = (Y_w - Z_w * tan(shv_rad)) * cos(shv_rad)
    target_x = int(f_x * X / Z + u0)
    target_y = int(f_y * Y / Z + v0)
    return target_x, target_y

def data_filter(ais, camera_para):
    '''
    Determines whether an AIS record falls within the camera's field of view.
    :param ais:        vessel AIS data at the current timestamp
    :param camera_para: camera parameter array
    :return: 'transform'   — within FOV, proceed with coordinate projection
             'visTraj_del' — outside angular limit, drop visual trajectory
             'ais_del'     — outside wider angular limit, drop AIS record entirely
    '''
    # Initialise camera parameters
    lon_cam    = camera_para[0]
    lat_cam    = camera_para[1]
    shoot_hdir = camera_para[2]
    shoot_vdir = camera_para[3]
    height_cam = camera_para[4]
    FOV_hor    = camera_para[5]
    FOV_ver    = camera_para[6]

    lon, lat = ais['lon'], ais['lat']
    D_abs = count_distance((lat_cam, lon_cam), (lat, lon))
    angle = getDegree(lat_cam, lon_cam, lat, lon)
    in_angle = abs(shoot_hdir - angle) if abs(shoot_hdir - angle) < 180 \
               else 360 - abs(shoot_hdir - angle)

    # First check vertical visibility; if within vertical FOV, check horizontal FOV
    if height_cam == 0 or 90 + shoot_vdir - FOV_ver / 2 < math.degrees(math.atan(D_abs / max(1, height_cam))):
        # ─────────────────────────────────────────────────────────────────────
        # Coordinate transform & visual trajectory filter zone
        # ─────────────────────────────────────────────────────────────────────
        # Within the horizontal FOV (+ a small margin) → allow coordinate transform.
        # The margin (+8 deg) compensates for AIS position lag at frame edges,
        # improving fusion quality for vessels near the FOV boundary.
        if in_angle <= (FOV_hor / 2 + 8):
            return 'transform'
        # Outside the extended angular limit → drop the visual trajectory
        elif in_angle > (FOV_hor / 2 + 8):
            return 'visTraj_del'
        # ─────────────────────────────────────────────────────────────────────
        # AIS data filter zone
        # ─────────────────────────────────────────────────────────────────────
        # Outside the wider angular limit → drop the AIS record entirely
        if in_angle > (FOV_hor / 2 + 12):
            return 'ais_del'


def transform(AIS_current, AIS_vis, camera_para, shape):
    '''
    Projects AIS data into the image coordinate system.
    :param AIS_current: AIS records at the current timestamp
    :param AIS_vis:     AIS records that already have image coordinates
    :return: (AIS_vis, AIS_visCurrent) — updated history and current projected records
    '''
    # 1. Initialise output DataFrame
    AIS_visCurrent = pd.DataFrame(columns=['mmsi', 'lon', 'lat', 'speed',
                                           'course', 'heading', 'type', 'x', 'y', 'timestamp'])
    # 2. Iterate over all AIS records
    for index, ais in AIS_current.iterrows():
        # Check whether the record can be projected
        flag = data_filter(ais, camera_para)
        # Case 1: project to image coordinates
        if flag == 'transform':
            x, y = visual_transform(ais['lon'], ais['lat'], camera_para, shape)
            ais['x'], ais['y'] = x, y
            AIS_visCurrent = pd.concat([AIS_visCurrent, ais.to_frame().T], ignore_index=True)
        # Case 2: remove record from visual trajectory history
        elif flag == 'visTraj_del' or flag == 'ais_del':
            AIS_vis = AIS_vis.drop(AIS_vis[AIS_vis['mmsi'] == ais['mmsi']].index)
    return AIS_vis, AIS_visCurrent


def data_pre(ais, timestamp):
    '''
    Dead-reckons a vessel's AIS position forward to the current timestamp.
    :param ais:       AIS record from the previous second that is absent at the current time
    :param timestamp: current timestamp (seconds)
    :return: updated AIS record with reckoned position
    '''
    # If the vessel is stationary, only update the timestamp
    if ais['speed'] == 0:
        ais['timestamp'] = timestamp
    # Otherwise dead-reckon the position forward in time
    else:
        geo_d = pyproj.Geod(ellps="WGS84")

        # Compute distance travelled since the last fix
        distance = ais['speed'] * ((timestamp - ais['timestamp']) / 3600) * 1852
        ais['timestamp'] = timestamp

        # Compute new lon/lat using the vessel's course and distance
        ais['lon'], ais['lat'], c = geo_d.fwd(
            ais['lon'], ais['lat'], ais['course'], distance)
    return ais

def data_pred(AIS_cur, AIS_read, AIS_las, timestamp):

    for index, ais in AIS_read.iterrows():
        ais['timestamp'] = round(ais['timestamp'] / 1000)
        # 1. Record exists at the current timestamp — use as-is
        if ais['timestamp'] == int(timestamp // 1000):
            AIS_cur = pd.concat([AIS_cur, ais.to_frame().T], ignore_index=True)
        # 2. Record is from a previous timestamp — dead-reckon it forward
        else:
            AIS_cur = pd.concat([AIS_cur, data_pre(ais, timestamp // 1000).to_frame().T], ignore_index=True)

    for index, ais in AIS_las.iterrows():
        if ais['mmsi'] not in AIS_cur['mmsi'].values:
            AIS_cur = pd.concat([AIS_cur, data_pre(ais, timestamp // 1000).to_frame().T], ignore_index=True)
    return AIS_cur

def data_coarse_process(AIS_current, AIS_last, camera_para, max_dis):
    '''
    Coarse data pre-processing: cleaning and spatial filtering.
    :param AIS_current: vessel AIS records at the current timestamp
    :param AIS_last:    vessel AIS records from the previous timestamp
    :param camera_para: camera parameter array
    :param max_dis:     maximum detection range of the camera (metres)
    :return: cleaned AIS records for the current timestamp
    '''
    camera_loc = (camera_para[1], camera_para[0])

    for index, ais in AIS_current.iterrows():
        # 1. Remove records with invalid field values
        if ais['mmsi'] / 100000000 < 1 or ais['mmsi'] / 100000000 >= 10 or \
                ais['lon'] == -1 or ais['lat'] == -1 or ais['speed'] == -1 or \
                ais['course'] == -1 or ais['course'] == 360 or ais['heading'] == -1 or \
                ais['lon'] > 180 or ais['lon'] < 0 or ais['lat'] > 90 or \
                ais['lat'] < 0 or ais['speed'] < 0:
            AIS_current = AIS_current.drop(index=index)
            continue

        # 2. Remove records where position or speed changed implausibly since the last frame
        #    (only checked if the vessel also appeared in the previous frame)
        if ais['mmsi'] in AIS_last['mmsi'].values:
            temp = AIS_last[AIS_last.mmsi == ais['mmsi']]
            if abs(ais['lon'] - temp['lon'].values[-1]) >= 1 \
                    or abs(ais['lat'] - temp['lat'].values[-1]) >= 1 \
                    or abs(ais['speed'] - temp['speed'].values[-1]) >= 7:
                AIS_current = AIS_current.drop(index=index)
                continue

        # 3. Remove records that are too far away or outside the angular filter zone
        ship_loc = (ais['lat'], ais['lon'])
        dis = count_distance(camera_loc, ship_loc, Type='m')
        if dis > max_dis or data_filter(ais, camera_para) == 'ais_del':
            AIS_current = AIS_current.drop(index=index)
    return AIS_current


class AISPRO(object):
    def __init__(self, ais_path, ais_file, im_shape, t):
        # Path to the AIS CSV directory
        self.ais_path = ais_path
        # List of AIS filenames
        self.ais_file = ais_file
        # Image dimensions
        self.im_shape = im_shape
        # Maximum range for retaining AIS records (metres)
        self.max_dis  = 4 * 1852
        # Display duration per frame (ms)
        self.t        = t
        # Maximum retention time for AIS history (minutes)
        self.time_lim = 2
        # Data store 1: AIS records at the current timestamp
        self.AIS_cur  = pd.DataFrame(columns=['mmsi', 'lon', 'lat', 'speed',
                                               'course', 'heading', 'type', 'timestamp'])
        # Data store 2: AIS records projected into image coordinates
        self.AIS_vis  = pd.DataFrame(columns=['mmsi', 'lon', 'lat', 'speed',
                                               'course', 'heading', 'type', 'x', 'y', 'timestamp'])
        # One Kalman filter per vessel, keyed by MMSI
        self.kalman_filters = {}

    def initialization(self):
        # Reset per-frame working buffers
        AIS_las = self.AIS_cur
        AIS_vis = self.AIS_vis
        AIS_cur = pd.DataFrame(columns=['mmsi', 'lon', 'lat', 'speed',
                                        'course', 'heading', 'type', 'timestamp'])
        return AIS_cur, AIS_las, AIS_vis

    def read_ais(self, Time_name):
        try:
            # Load AIS CSV for this timestamp
            path = self.ais_path + '/' + Time_name + '.csv'
            ais_data = pd.read_csv(path, usecols=[1, 2, 3, 4, 5, 6, 7, 8], header=0)
        except:
            ais_data = pd.DataFrame(columns=['mmsi', 'lon', 'lat', 'speed',
                                             'course', 'heading', 'type', 'timestamp'])
        return ais_data

    def data_tran(self, AIS_cur, AIS_vis, camera_para, timestamp):
        # 1. Project AIS records into image coordinates
        AIS_vis, AIS_vis_cur = transform(AIS_cur, AIS_vis, camera_para, self.im_shape)

        # 2. Kalman smoothing on pixel positions
        # .copy() is critical — ensures .loc writes work on a real DataFrame, not a view
        AIS_vis_cur = AIS_vis_cur.copy()
        for idx in AIS_vis_cur.index:
            try:
                mmsi   = int(AIS_vis_cur.loc[idx, 'mmsi'])
                x_meas = float(AIS_vis_cur.loc[idx, 'x'])
                y_meas = float(AIS_vis_cur.loc[idx, 'y'])
                if mmsi not in self.kalman_filters:
                    self.kalman_filters[mmsi] = KalmanAIS()
                x_s, y_s = self.kalman_filters[mmsi].update(x_meas, y_meas)
                AIS_vis_cur.loc[idx, 'x'] = x_s
                AIS_vis_cur.loc[idx, 'y'] = y_s
            except:
                pass

        # 3. Append projected records to the running AIS history
        AIS_vis = pd.concat([AIS_vis, AIS_vis_cur], ignore_index=True)

        # 4. Drop records older than time_lim minutes from the history
        AIS_vis = AIS_vis.drop(AIS_vis[AIS_vis['timestamp'] < (
                timestamp // 1000 - self.time_lim * 60)].index)
        return AIS_vis

    def ais_pro(self, AIS_cur, AIS_las, AIS_vis, camera_para, timestamp, Time_name):
        # 1. Load AIS CSV for the current timestamp
        AIS_read = self.read_ais(Time_name)

        # 2. Coarse data cleaning and spatial filtering
        AIS_read = data_coarse_process(AIS_read, AIS_las, camera_para, self.max_dis)

        # 3. Dead-reckon any vessels absent from the current CSV
        AIS_cur = data_pred(AIS_cur, AIS_read, AIS_las, timestamp)

        # 4. Project AIS records into image coordinates
        AIS_vis = self.data_tran(AIS_cur, AIS_vis, camera_para, timestamp)
        return AIS_vis, AIS_cur

    def process(self, camera_para, timestamp, Time_name):
        # Only update when a new AIS second boundary is crossed
        if timestamp % 1000 < self.t:
            Time_name = Time_name[:-4]
            # 1. Reset working buffers
            AIS_cur, AIS_las, AIS_vis = self.initialization()

            # 2. Run full AIS pipeline for this timestamp
            self.AIS_vis, self.AIS_cur = self.ais_pro(AIS_cur, AIS_las, AIS_vis,
                                                       camera_para, timestamp, Time_name)

        return self.AIS_vis, self.AIS_cur