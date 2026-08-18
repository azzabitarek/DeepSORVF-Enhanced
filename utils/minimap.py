"""
minimap.py — Half-circle radar overlay for DeepSORVF
═════════════════════════════════════════════════════
Three ship categories:
  1. FUSED        — AIS + visual match       → colored dot + trail + 60s arrow + confidence ring
  2. AIS_ONLY     — AIS signal, no detection  → cyan diamond (ship exists but YOLO missed it)
  3. VISUAL_ONLY  — Detected visually, no AIS → orange ring (unknown ship)

Range = AIS max_dis (4 × 1852 m by default) so radar and AIS filter are always in sync.
"""

import cv2
import numpy as np
import math
import os
from geopy.distance import geodesic
import pyproj

# ── Ship type helpers ─────────────────────────────────────────────────────────
AIS_TYPE_MAP = {
    0:'Unknown', 3:'Fishing', 18:'Class-B', 22:'Towing',
    31:'SAR', 32:'Tug', 50:'Pilot', 51:'SAR', 52:'Tug',
    60:'Passenger', 70:'Cargo', 71:'Cargo', 72:'Cargo',
    80:'Tanker', 81:'Tanker', 90:'Other',
}
TYPE_COLOR = {
    'Cargo':     ( 51, 204,  51),
    'Tanker':    ( 51,  51, 255),
    'Passenger': (255, 204,  51),
    'Fishing':   ( 51, 255, 255),
    'Tug':       (204,  51, 204),
    'SAR':       ( 51, 255,  51),
    'Unknown':   (180, 180, 180),
}
def _tlabel(v):
    try:    return AIS_TYPE_MAP.get(int(float(v)), 'Unknown')
    except: return 'Unknown'
def _tcolor(l):
    return TYPE_COLOR.get(l, TYPE_COLOR['Unknown'])

# ── Geo helpers ───────────────────────────────────────────────────────────────
_geo = pyproj.Geod(ellps='WGS84')
KT_TO_MPS = 1852 / 3600.0

def _norm(a):
    while a >  180: a -= 360
    while a < -180: a += 360
    return a

def _bearing_dist(lon_cam, lat_cam, lon_s, lat_s):
    fwd, _, d = _geo.inv(lon_cam, lat_cam, lon_s, lat_s)
    return fwd % 360, d

def gps_to_hc(lon_s, lat_s, lon_cam, lat_cam, hdir, scale, cpx, cpy):
    """GPS → half-circle canvas pixel. Returns (None,None) if outside ±90°."""
    try:
        bearing, dist = _bearing_dist(lon_cam, lat_cam, lon_s, lat_s)
        rel = _norm(bearing - hdir)
        if abs(rel) > 90:
            return None, None
        r  = dist / scale
        px = int(cpx + r * math.sin(math.radians(rel)))
        py = int(cpy - r * math.cos(math.radians(rel)))
        return px, py
    except:
        return None, None

def project_forward(lon, lat, course_deg, dist_m):
    lon2, lat2, _ = _geo.fwd(lon, lat, course_deg, dist_m)
    return lon2, lat2

def pixel_to_gps_pinhole(u, v, camera_para):
    """Fallback pinhole pixel → GPS."""
    try:
        lon_cam, lat_cam = camera_para[0], camera_para[1]
        hdir, vdir, h    = camera_para[2], camera_para[3], camera_para[4]
        fx, fy, u0, v0   = camera_para[7], camera_para[8], camera_para[9], camera_para[10]
        bearing    = (hdir + math.degrees(math.atan((u - u0) / fx))) % 360
        depression = vdir + math.degrees(math.atan((v - v0) / fy))
        if depression < 0.5: depression = 0.5
        dist = h / math.tan(math.radians(depression))
        if not 0 < dist < 15000: return None, None
        lon2, lat2, _ = _geo.fwd(lon_cam, lat_cam, bearing, dist)
        return lon2, lat2
    except:
        return None, None

# ── CPA ───────────────────────────────────────────────────────────────────────
def compute_cpa(lon1, lat1, spd1, cog1, lon2, lat2, spd2, cog2):
    try:
        fwd, _, d = _geo.inv(lon1, lat1, lon2, lat2)
        rb  = math.radians(fwd)
        px  = d * math.sin(rb);  py = d * math.cos(rb)
        vx1 = spd1*KT_TO_MPS*math.sin(math.radians(cog1))
        vy1 = spd1*KT_TO_MPS*math.cos(math.radians(cog1))
        vx2 = spd2*KT_TO_MPS*math.sin(math.radians(cog2))
        vy2 = spd2*KT_TO_MPS*math.cos(math.radians(cog2))
        dvx = vx2-vx1;  dvy = vy2-vy1
        spd2 = dvx*dvx + dvy*dvy
        if spd2 < 1e-6: return None, None
        tcpa = -(px*dvx + py*dvy) / spd2
        if tcpa < 0: return None, None
        dcpa = math.sqrt((px+dvx*tcpa)**2 + (py+dvy*tcpa)**2)
        return tcpa, dcpa
    except:
        return None, None

# ── Homography calibrator ─────────────────────────────────────────────────────
class HomographyCalibrator:
    MIN_PTS = 4
    def __init__(self, lon_cam, lat_cam):
        zone = int((lon_cam + 180) / 6) + 1
        hem  = 'north' if lat_cam >= 0 else 'south'
        self.proj = pyproj.Proj(proj='utm', zone=zone,
                                ellps='WGS84', hemisphere=hem)
        self.cam_e, self.cam_n = self.proj(lon_cam, lat_cam)
        self.pix = [];  self.gps = [];  self.H = None

    def add(self, u, v, lon, lat):
        e, n = self.proj(lon, lat)
        self.pix.append([float(u), float(v)])
        self.gps.append([e - self.cam_e, n - self.cam_n])
        if len(self.pix) > 300: self.pix.pop(0); self.gps.pop(0)
        self._fit()

    def _fit(self):
        if len(self.pix) < self.MIN_PTS: self.H = None; return
        H, _ = cv2.findHomography(np.array(self.pix, np.float32),
                                   np.array(self.gps, np.float32),
                                   cv2.RANSAC, 5.0)
        self.H = H

    def to_gps(self, u, v):
        if self.H is None: return None, None
        pt  = np.array([[[float(u), float(v)]]], np.float32)
        dst = cv2.perspectiveTransform(pt, self.H)
        dx, dy = float(dst[0,0,0]), float(dst[0,0,1])
        lon, lat = self.proj(self.cam_e+dx, self.cam_n+dy, inverse=True)
        return lon, lat

    @property
    def ok(self): return self.H is not None
    @property
    def n(self):  return len(self.pix)

# ── Drawing helpers ───────────────────────────────────────────────────────────
def _ok(px, py, W, H, pad=6):
    return px is not None and pad < px < W-pad and pad < py < H-pad

def _conf_color(conf):
    r = int(255*(1-conf)); g = int(255*conf)
    return (0, g, r)

def _label(img, lx, ly, lines, color, W, H, lh=14, fs=0.29):
    rw = 110; rh = lh*len(lines)+6
    rx1 = max(0, lx-2);       ry1 = max(0, ly-lh)
    rx2 = min(W-1, rx1+rw);   ry2 = min(H-1, ry1+rh)
    sub = img[ry1:ry2, rx1:rx2]
    if sub.size > 0:
        img[ry1:ry2, rx1:rx2] = cv2.addWeighted(
            sub, 0.1, np.zeros_like(sub), 0.9, 0)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (lx, ly+i*lh),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1, cv2.LINE_AA)

def _diamond(img, cx, cy, size, color, thick=1):
    pts = np.array([[cx, cy-size],[cx+size, cy],
                    [cx, cy+size],[cx-size, cy]], np.int32).reshape((-1, 1, 2))
    if thick < 0:
        cv2.fillPoly(img, [pts], color)
    else:
        cv2.polylines(img, [pts], True, color, thick, cv2.LINE_AA)

def _make_bg(W, H, cpx, cpy, R, rings, scale, hdir):
    bg = np.zeros((H, W, 3), np.uint8)
    bg[:] = (18, 20, 26)
    mask = np.zeros((H, W), np.uint8)
    cv2.ellipse(mask, (cpx, cpy), (R, R), 0, -180, 0, 255, -1)
    bg[mask > 0] = (22, 26, 34)
    for r_m in rings:
        r_px = int(r_m / scale)
        cv2.ellipse(bg, (cpx, cpy), (r_px, r_px), 0, -180, 0,
                    (45,60,45), 1, cv2.LINE_AA)
        lx = cpx + r_px + 3
        if lx < W-5:
            cv2.putText(bg, f'{int(r_m/1852):.1f}nm' if r_m >= 1852 else f'{int(r_m)}m',
                        (lx, cpy-4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.27, (60,90,60), 1, cv2.LINE_AA)
    for rel in range(-90, 91, 30):
        rad = math.radians(rel)
        ex  = int(cpx + R*math.sin(rad));  ey = int(cpy - R*math.cos(rad))
        col = (65,80,65) if rel == 0 else (38,50,38)
        cv2.line(bg, (cpx, cpy), (ex, ey), col, 1, cv2.LINE_AA)
        if rel != 0:
            ab = int((hdir + rel) % 360)
            lx = max(2, min(W-30, int(cpx+(R+16)*math.sin(rad))-12))
            ly = max(12, min(H-4,  int(cpy-(R+16)*math.cos(rad))+5))
            cv2.putText(bg, f'{ab}°', (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (65,80,65), 1, cv2.LINE_AA)
    cv2.line(bg, (cpx-R, cpy), (cpx+R, cpy), (70,95,70), 1, cv2.LINE_AA)
    cv2.ellipse(bg, (cpx, cpy), (R, R), 0, -180, 0, (80,115,80), 1, cv2.LINE_AA)
    cv2.putText(bg, f'N {int(hdir%360)}°', (cpx-22, cpy-R-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140,200,140), 1, cv2.LINE_AA)
    cv2.circle(bg, (cpx, cpy), 7, (255,255,255), -1, cv2.LINE_AA)
    cv2.putText(bg, 'CAM', (cpx+10, cpy+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255,255,255), 1, cv2.LINE_AA)
    return bg

# ══════════════════════════════════════════════════════════════════════════════
class MINIMAP:
    """
    Parameters
    ----------
    camera_para : from read_all()
    im_shape    : [video_W, video_H]
    range_m     : radar radius in metres — set equal to AIS max_dis (4*1852)
    map_size    : canvas width in pixels
    result_path : folder for radar video output
    clip_name   : filename prefix
    fps         : video fps
    cpa_warn_m  : CPA distance alert threshold (metres)
    cpa_warn_s  : CPA time alert threshold (seconds)
    """
    def __init__(self, camera_para, im_shape,
                 range_m=4*1852, map_size=700,
                 result_path='./result/video/', clip_name='clip-01',
                 fps=1, cpa_warn_m=400, cpa_warn_s=120):

        self.cp    = camera_para
        self.lonc  = camera_para[0]
        self.latc  = camera_para[1]
        self.hdir  = camera_para[2]
        self.range_m = range_m

        self.W   = map_size
        self.mb  = 60    # bottom legend margin
        self.mt  = 48    # top margin
        self.H   = map_size // 2 + self.mb + self.mt
        self.R   = map_size // 2 - 14
        self.sc  = range_m / self.R      # metres per pixel
        self.cpx = self.W // 2
        self.cpy = self.H - self.mb

        rings   = [range_m//3, 2*range_m//3, range_m]
        self.bg = _make_bg(self.W, self.H, self.cpx, self.cpy,
                           self.R, rings, self.sc, self.hdir)

        self.trails     = {}
        self.tlen       = 50
        self.calib      = HomographyCalibrator(self.lonc, self.latc)
        self.cpa_m      = cpa_warn_m
        self.cpa_s      = cpa_warn_s
        self._cpa_blink = 0

        os.makedirs(result_path, exist_ok=True)
        vpath  = os.path.join(result_path, clip_name + '_radar.mp4')
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        self.writer = cv2.VideoWriter(vpath, fourcc, fps, (self.W, self.H))
        print(f'[MINIMAP] range={range_m/1852:.1f}nm  radar→{vpath}')

    # ── helpers ───────────────────────────────────────────────────────────────
    def _p(self, lon, lat):
        return gps_to_hc(lon, lat, self.lonc, self.latc,
                          self.hdir, self.sc, self.cpx, self.cpy)

    def _trail(self, key, px, py, color, mm):
        if key not in self.trails: self.trails[key] = []
        self.trails[key].append((px, py))
        if len(self.trails[key]) > self.tlen: self.trails[key].pop(0)
        t = self.trails[key]
        for k in range(1, len(t)):
            a  = int(220 * k / len(t))
            tc = tuple(int(c * a // 220) for c in color)
            cv2.line(mm, t[k-1], t[k], tc, 1, cv2.LINE_AA)

    def _feed_calib(self, fusion_list, cur_sec):
        if fusion_list is None or len(fusion_list) == 0: return
        cur = fusion_list[fusion_list['timestamp'] == cur_sec]
        for _, row in cur.iterrows():
            try:
                bx = int(float(row['x1']) + float(row['w'])/2)
                by = int(float(row['y1']) + float(row['h']))
                self.calib.add(bx, by, float(row['lon']), float(row['lat']))
            except: pass

    def _draw_60s(self, mm, lon, lat, speed, course, color):
        dist = float(speed) * KT_TO_MPS * 60
        if dist < 5: return
        lon2, lat2 = project_forward(lon, lat, course, dist)
        px1, py1   = self._p(lon, lat)
        px2, py2   = self._p(lon2, lat2)
        if px1 is None or px2 is None: return
        if not _ok(px2, py2, self.W, self.H, pad=2): return
        cv2.arrowedLine(mm, (px1,py1), (px2,py2), color, 1,
                        tipLength=0.25, line_type=cv2.LINE_AA)

    def _check_cpa(self, cur_fused):
        warnings = []
        if cur_fused is None or len(cur_fused) < 2: return warnings
        rows = cur_fused.to_dict('records')
        for i in range(len(rows)):
            for j in range(i+1, len(rows)):
                a, b = rows[i], rows[j]
                try:
                    tcpa, dcpa = compute_cpa(
                        float(a['lon']),float(a['lat']),
                        float(a['speed']),float(a['course']),
                        float(b['lon']),float(b['lat']),
                        float(b['speed']),float(b['course']))
                    if tcpa and dcpa < self.cpa_m and tcpa < self.cpa_s:
                        warnings.append((int(a['mmsi']),int(b['mmsi']),
                                         round(tcpa),round(dcpa)))
                except: pass
        return warnings

    def _draw_cpa(self, mm, warnings):
        if not warnings: return
        self._cpa_blink = (self._cpa_blink+1) % 6
        if self._cpa_blink < 3: return
        y = 28
        for m1, m2, t, d in warnings:
            cv2.rectangle(mm, (4,y-14),(self.W-4,y+4),(0,0,180),-1)
            cv2.putText(mm, f'! CPA {m1}&{m2}: {d}m in {t}s',
                        (8,y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32,(255,255,255),1,cv2.LINE_AA)
            y += 20

    # ── main draw ─────────────────────────────────────────────────────────────
    def draw(self, frame, AIS_vis, AIS_cur, fusion_list, Vis_tra, Vis_cur,
             timestamp, wake_angles=None):
        """
        Parameters
        ----------
        frame        : BGR video frame
        AIS_vis      : full AIS history dataframe (with x,y pixel columns)
        AIS_cur      : current-second AIS dataframe (lon,lat,mmsi,speed,course...)
        fusion_list  : Fus_tra from FUSPRO (has confidence column)
        Vis_tra      : visual trajectory history
        Vis_cur      : current-second visual tracks
        timestamp    : ms timestamp
        wake_angles  : {track_id: compass_degrees} from WakeDetector (optional)
        """
        cur_sec = timestamp // 1000
        mm = self.bg.copy()

        # Feed homography calibrator from fused ships
        self._feed_calib(fusion_list, cur_sec)

        # ── Build fused sets for current second ───────────────────────────────
        cur_fused      = None
        fused_mmsi_set = set()
        fused_id_set   = set()
        if fusion_list is not None and len(fusion_list) > 0:
            cf = fusion_list[fusion_list['timestamp'] == cur_sec]
            if len(cf) > 0:
                cur_fused      = cf
                fused_mmsi_set = set(cf['mmsi'].astype(int).unique())
                fused_id_set   = set(cf['ID'].astype(int).unique())

        # ══ LAYER 1 — AIS signal, no visual detection (cyan diamond) ══════════
        # These are ships in AIS_cur that were NOT matched to any visual track.
        # They have a real AIS position but YOLO didn't detect them this second.
        COLOR_AIS_NODET = (220, 220, 0)   # cyan-yellow
        if AIS_cur is not None and len(AIS_cur) > 0:
            for _, row in AIS_cur.iterrows():
                try:
                    mmsi = int(row['mmsi'])
                    if mmsi in fused_mmsi_set:
                        continue   # already shown as fused
                    lon = float(row['lon'])
                    lat = float(row['lat'])
                    spd = float(row['speed'])
                    cog = float(row['course'])
                    tlbl = _tlabel(row.get('type', 0))
                    dist_m = geodesic((self.latc,self.lonc),(lat,lon)).m
                except:
                    continue

                px, py = self._p(lon, lat)
                if not _ok(px, py, self.W, self.H, pad=8):
                    continue

                self._trail(f'a{mmsi}', px, py, COLOR_AIS_NODET, mm)
                _diamond(mm, px, py, 5, COLOR_AIS_NODET, thick=-1)
                self._draw_60s(mm, lon, lat, spd, cog, COLOR_AIS_NODET)

                lx = px + 11
                if lx + 112 > self.W: lx = px - 112
                _label(mm, lx, py-2, [
                    f'MMSI:{mmsi}',
                    f'{tlbl}',
                    f'SPD:{spd:.1f}kn',
                    f'COG:{cog:.0f}',
                    f'{dist_m:.0f}m',
                    'NO DET',
                ], COLOR_AIS_NODET, self.W, self.H)

        # ══ LAYER 2 — Fused ships (AIS + visual match) ════════════════════════
        COLOR_FUSED = None   # per ship type
        if cur_fused is not None:
            for _, row in cur_fused.iterrows():
                try:
                    mmsi   = int(row['mmsi'])
                    lon    = float(row['lon'])
                    lat    = float(row['lat'])
                    speed  = float(row['speed'])
                    course = float(row['course'])
                    tlbl   = _tlabel(row.get('type', 0))
                    color  = _tcolor(tlbl)
                    conf   = float(row['confidence']) if 'confidence' in row.index else 0.5
                    dist_m = geodesic((self.latc,self.lonc),(lat,lon)).m
                    cc     = _conf_color(conf)
                except:
                    continue

                px, py = self._p(lon, lat)
                if not _ok(px, py, self.W, self.H, pad=8):
                    continue

                self._trail(f'm{mmsi}', px, py, color, mm)
                self._draw_60s(mm, lon, lat, speed, course, color)

                cv2.circle(mm, (px,py), 7, color, -1, cv2.LINE_AA)
                cv2.circle(mm, (px,py), 8, (255,255,255), 1, cv2.LINE_AA)
                cv2.circle(mm, (px,py), 10, cc, 1, cv2.LINE_AA)

                lx = px + 12
                if lx + 112 > self.W: lx = px - 112
                _label(mm, lx, py-4, [
                    f'MMSI:{mmsi}',
                    f'{tlbl}',
                    f'SPD:{speed:.1f}kn',
                    f'COG:{course:.0f}',
                    f'{dist_m:.0f}m',
                    f'CNF:{int(conf*100)}%',
                ], color, self.W, self.H)

        # ══ LAYER 3 — Visual-only ships (detected, no AIS) ════════════════════
        COLOR_VIS = (0, 120, 255)   # orange
        if Vis_cur is not None and len(Vis_cur) > 0:
            vis_now = Vis_cur[Vis_cur['timestamp'] == cur_sec]
            if len(vis_now) == 0: vis_now = Vis_cur

            for _, vrow in vis_now.iterrows():
                try: tid = int(vrow['ID'])
                except: continue
                if tid in fused_id_set: continue

                try:
                    bx = int((float(vrow['x1'])+float(vrow['x2']))/2)
                    by = int(float(vrow['y2']))
                except:
                    try: bx, by = int(float(vrow['x'])), int(float(vrow['y']))
                    except: continue

                # GPS estimate
                if self.calib.ok:
                    lon_s, lat_s = self.calib.to_gps(bx, by)
                    method = f'H({self.calib.n})'
                else:
                    lon_s, lat_s = pixel_to_gps_pinhole(bx, by, self.cp)
                    method = 'approx'

                if lon_s is None: continue
                px, py = self._p(lon_s, lat_s)
                if not _ok(px, py, self.W, self.H, pad=5): continue

                dist_m = geodesic((self.latc,self.lonc),(lat_s,lon_s)).m
                self._trail(f'v{tid}', px, py, COLOR_VIS, mm)
                cv2.circle(mm, (px,py), 7, COLOR_VIS, 1, cv2.LINE_AA)
                cv2.circle(mm, (px,py), 3, COLOR_VIS, -1, cv2.LINE_AA)

                # Wake heading arrow
                if wake_angles and tid in wake_angles:
                    ship_head = (wake_angles[tid] + 180) % 360
                    rel = math.radians(_norm(ship_head - self.hdir))
                    ex  = int(px + 16*math.sin(rel))
                    ey  = int(py - 16*math.cos(rel))
                    cv2.arrowedLine(mm, (px,py), (ex,ey),
                                    (51,255,51), 1, tipLength=0.35,
                                    line_type=cv2.LINE_AA)

                lx = px + 10
                if lx + 112 > self.W: lx = px - 112
                lines = [f'ID:{tid}', 'NO AIS', method, f'~{dist_m:.0f}m']
                if wake_angles and tid in wake_angles:
                    lines.append(f'WK:{wake_angles[tid]:.0f}')
                _label(mm, lx, py-2, lines, COLOR_VIS, self.W, self.H, lh=14)

        # ══ CPA warnings ══════════════════════════════════════════════════════
        self._draw_cpa(mm, self._check_cpa(cur_fused))

        # ══ Status bar ════════════════════════════════════════════════════════
        cv2.putText(mm, f'RADAR  T={cur_sec}s  range={self.range_m/1852:.1f}nm',
                    (6,16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, (120,160,120), 1, cv2.LINE_AA)
        cal_txt = (f'Homo:{self.calib.n}pts OK' if self.calib.ok
                   else f'Homo:{self.calib.n}/{HomographyCalibrator.MIN_PTS} fallback')
        cv2.putText(mm, cal_txt, (6, self.H-self.mb+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                    (80,220,80) if self.calib.ok else (80,120,200),
                    1, cv2.LINE_AA)

        # ══ Legend ════════════════════════════════════════════════════════════
        ly = self.H - self.mb + 27
        for col, txt in [
            ((255,255,255),  'Fused (AIS+visual)'),
            (COLOR_AIS_NODET,'AIS / no detection'),
            (COLOR_VIS,      'Visual / no AIS'),
            ((51,255,51),    'Wake heading'),
        ]:
            cv2.circle(mm, (8,ly), 4, col, -1, cv2.LINE_AA)
            cv2.putText(mm, txt, (16,ly+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.27, col, 1, cv2.LINE_AA)
            ly += 13

        cv2.rectangle(mm, (0,0),(self.W-1,self.H-1),(70,100,70),1)

        # ══ Save radar video ══════════════════════════════════════════════════
        self.writer.write(mm)

        # ══ Overlay on main frame (bottom-right) ══════════════════════════════
        ih, iw = frame.shape[:2]
        ox = 10
        oy =  10
        if ox >= 0 and oy >= 0:
            roi = frame[oy:oy+self.H, ox:ox+self.W]
            if roi.shape == mm.shape:
                frame[oy:oy+self.H, ox:ox+self.W] = \
                    cv2.addWeighted(roi, 0.15, mm, 0.85, 0)
        return frame

    def release(self):
        if self.writer:
            self.writer.release()
            print('[MINIMAP] Radar video saved.')