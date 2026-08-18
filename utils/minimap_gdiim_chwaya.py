import cv2
import numpy as np
import math
from geopy.distance import geodesic
import pyproj

# ─────────────────────────────────────────────
#  Ship type lookup (AIS type field → label)
# ─────────────────────────────────────────────
AIS_TYPE_MAP = {
    0:  'Unknown',  1:  'Class-A',  3:  'Fishing',
    18: 'Class-B', 19: 'Class-B',  20: 'WIG',
    21: 'Fishing', 22: 'Towing',   23: 'Towing',
    24: 'Dredging',25: 'Diving',   26: 'Military',
    27: 'Sailing', 28: 'Pleasure', 29: 'HSC',
    30: 'Pilot',   31: 'SAR',      32: 'Tug',
    33: 'Port',    34: 'Anti-poll',35: 'Law Enf.',
    50: 'Pilot',   51: 'SAR',      52: 'Tug',
    53: 'Port',    54: 'Anti-poll',55: 'Law Enf.',
    60: 'Passenger',70: 'Cargo',   71: 'Cargo',
    72: 'Cargo',   73: 'Cargo',    74: 'Cargo',
    80: 'Tanker',  81: 'Tanker',   82: 'Tanker',
    83: 'Tanker',  84: 'Tanker',   90: 'Other',
}

TYPE_COLOR_MAP = {
    'Cargo':     (51,  204, 51),
    'Tanker':    (51,  51,  255),
    'Passenger': (255, 204, 51),
    'Fishing':   (51,  255, 255),
    'Class-A':   (255, 153, 51),
    'Class-B':   (255, 153, 51),
    'Tug':       (204, 51,  204),
    'SAR':       (51,  255, 51),
    'Unknown':   (180, 180, 180),
}

def get_type_label(type_val):
    try:
        t = int(float(type_val))
    except:
        t = 0
    return AIS_TYPE_MAP.get(t, 'Unknown')

def get_type_color(type_label):
    return TYPE_COLOR_MAP.get(type_label, TYPE_COLOR_MAP['Unknown'])

# ─────────────────────────────────────────────
#  GPS → minimap pixel
# ─────────────────────────────────────────────
def gps_to_minimap(lon_ship, lat_ship, lon_cam, lat_cam,
                   scale_m_per_px, mm_cx, mm_cy):
    dx = geodesic((lat_cam, lon_cam), (lat_cam, lon_ship)).m
    if lon_ship < lon_cam:
        dx = -dx
    dy = geodesic((lat_cam, lon_cam), (lat_ship, lon_cam)).m
    if lat_ship < lat_cam:
        dy = -dy
    px = int(mm_cx + dx / scale_m_per_px)
    py = int(mm_cy - dy / scale_m_per_px)
    return px, py

# ─────────────────────────────────────────────
#  Pixel → approximate GPS (ground-plane model)
#  Uses the bottom-centre of the bounding box
#  as the waterline contact point.
# ─────────────────────────────────────────────
def pixel_to_gps(u, v, camera_para):
    """
    Back-project image pixel (u, v) to approximate (lon, lat)
    using a flat-ground camera model.

    camera_para indices:
      0  lon_cam      4  height_cam
      1  lat_cam      5  FOV_hor
      2  shoot_hdir   6  FOV_ver
      3  shoot_vdir   7  f_x
                      8  f_y
                      9  u0
                      10 v0
    """
    lon_cam    = camera_para[0]
    lat_cam    = camera_para[1]
    shoot_hdir = camera_para[2]
    shoot_vdir = camera_para[3]
    height_cam = camera_para[4]
    f_x        = camera_para[7]
    f_y        = camera_para[8]
    u0         = camera_para[9]
    v0         = camera_para[10]

    # horizontal bearing from camera
    dx_norm    = (u - u0) / f_x
    hor_offset = math.degrees(math.atan(dx_norm))
    bearing    = (shoot_hdir + hor_offset) % 360

    # vertical depression angle
    dy_norm    = (v - v0) / f_y
    pix_vangle = math.degrees(math.atan(dy_norm))
    depression = shoot_vdir + pix_vangle

    if depression <= 0.5:
        depression = 0.5

    distance = height_cam / math.tan(math.radians(depression))

    if distance <= 0 or distance > 20000:
        return None, None

    geo_d = pyproj.Geod(ellps='WGS84')
    lon_ship, lat_ship, _ = geo_d.fwd(lon_cam, lat_cam, bearing, distance)
    return lon_ship, lat_ship

# ─────────────────────────────────────────────
#  Drawing helpers
# ─────────────────────────────────────────────
def draw_arrow(img, cx, cy, course_deg, length, color, thickness=1):
    rad = math.radians(course_deg)
    ex  = int(cx + length * math.sin(rad))
    ey  = int(cy - length * math.cos(rad))
    cv2.arrowedLine(img, (cx, cy), (ex, ey), color, thickness,
                    tipLength=0.35, line_type=cv2.LINE_AA)

def draw_compass(img, cx, cy, r):
    col = (180, 180, 180)
    for angle, label, ox, oy in [(0,'N',-4,-r-6),(90,'E',r+3,4),
                                   (180,'S',-4,r+12),(-90,'W',-r-14,4)]:
        rad = math.radians(angle)
        tx  = int(cx + r * math.sin(rad))
        ty  = int(cy - r * math.cos(rad))
        cv2.line(img, (cx, cy), (tx, ty), col, 1, cv2.LINE_AA)
        cv2.putText(img, label, (cx+ox, cy+oy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)

def label_block(img, lx, ly, lines, color, size, line_h=13, font_scale=0.27):
    rect_w = 86
    rect_h = line_h * len(lines) + 6
    rx1 = max(0,      lx - 2)
    ry1 = max(0,      ly - line_h)
    rx2 = min(size-1, rx1 + rect_w)
    ry2 = min(size-1, ry1 + rect_h)
    sub = img[ry1:ry2, rx1:rx2]
    if sub.size > 0:
        black = np.zeros_like(sub)
        img[ry1:ry2, rx1:rx2] = cv2.addWeighted(sub, 0.25, black, 0.75, 0)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (lx, ly + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    color, 1, cv2.LINE_AA)


# ═════════════════════════════════════════════
class MINIMAP:
    """
    Radar-style minimap overlay — bottom-right corner.

    Three layers:
      • Fused ships    : colored dot + trail + arrow + MMSI/type/speed/COG/distance
      • AIS-only       : small grey dot  (in AIS but not matched yet)
      • Visual-only    : orange-red ring (detected by YOLO, no AIS yet)
    """

    def __init__(self, camera_para, im_shape,
                 size    = 420,
                 range_m = 2000,
                 margin  = 15):
        self.camera_para = camera_para
        self.lon_cam  = camera_para[0]
        self.lat_cam  = camera_para[1]
        self.im_w     = int(im_shape[0])
        self.im_h     = int(im_shape[1])

        self.size     = size
        self.margin   = margin
        self.range_m  = range_m
        self.mm_cx    = size // 2
        self.mm_cy    = size // 2
        self.scale    = range_m / (size // 2)

        self.x0 = self.im_w - size - margin
        self.y0 = self.im_h - size - margin

        self.rings     = [range_m // 3, 2 * range_m // 3, range_m]
        self.trails    = {}
        self.trail_len = 40

    # ── background ─────────────────────────────────────────────────
    def _make_background(self):
        bg = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        bg[:] = (20, 22, 28)

        step = self.size // 8
        for i in range(0, self.size, step):
            cv2.line(bg, (i, 0), (i, self.size), (32, 34, 40), 1)
            cv2.line(bg, (0, i), (self.size, i),  (32, 34, 40), 1)

        for r_m in self.rings:
            r_px = int(r_m / self.scale)
            cv2.circle(bg, (self.mm_cx, self.mm_cy), r_px,
                       (50, 65, 50), 1, cv2.LINE_AA)
            cv2.putText(bg, f'{int(r_m)}m',
                        (self.mm_cx + r_px + 2, self.mm_cy - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.27,
                        (65, 90, 65), 1, cv2.LINE_AA)

        draw_compass(bg, self.mm_cx, self.mm_cy, self.size // 2 - 16)

        cv2.circle(bg, (self.mm_cx, self.mm_cy), 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(bg, 'CAM', (self.mm_cx + 8, self.mm_cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 255, 255), 1, cv2.LINE_AA)
        return bg

    def _fused_mmsi_set(self, fusion_list):
        if fusion_list is None or len(fusion_list) == 0:
            return set()
        return set(fusion_list['mmsi'].unique().astype(int))

    def _fused_id_set(self, fusion_list):
        if fusion_list is None or len(fusion_list) == 0:
            return set()
        return set(fusion_list['ID'].unique().astype(int))

    def _get_fused_info(self, fusion_list, mmsi):
        if fusion_list is None or len(fusion_list) == 0:
            return None
        rows = fusion_list[fusion_list['mmsi'] == mmsi]
        return rows.iloc[-1] if len(rows) > 0 else None

    def _in_canvas(self, px, py, pad=5):
        return pad < px < self.size - pad and pad < py < self.size - pad

    # ── main ───────────────────────────────────────────────────────
    def draw(self, frame, AIS_vis, fusion_list, Vis_tra, Vis_cur, timestamp):
        mm = self._make_background()

        fused_mmsi = self._fused_mmsi_set(fusion_list)
        fused_ids  = self._fused_id_set(fusion_list)

        # ══ LAYER 1 — AIS-only ghost dots ══════════════════════════
        if AIS_vis is not None and len(AIS_vis) > 0:
            latest = (AIS_vis.sort_values('timestamp')
                             .groupby('mmsi').last().reset_index())
            for _, row in latest.iterrows():
                try:
                    mmsi = int(row['mmsi'])
                    lon  = float(row['lon'])
                    lat  = float(row['lat'])
                except:
                    continue
                if mmsi in fused_mmsi:
                    continue
                px, py = gps_to_minimap(lon, lat, self.lon_cam, self.lat_cam,
                                        self.scale, self.mm_cx, self.mm_cy)
                if not self._in_canvas(px, py):
                    continue
                cv2.circle(mm, (px, py), 3, (75, 75, 75), -1, cv2.LINE_AA)

        # ══ LAYER 2 — Fused ships ═══════════════════════════════════
        if AIS_vis is not None and len(AIS_vis) > 0:
            latest = (AIS_vis.sort_values('timestamp')
                             .groupby('mmsi').last().reset_index())
            for _, row in latest.iterrows():
                try:
                    mmsi = int(row['mmsi'])
                    lon  = float(row['lon'])
                    lat  = float(row['lat'])
                except:
                    continue
                if mmsi not in fused_mmsi:
                    continue
                fused = self._get_fused_info(fusion_list, mmsi)
                if fused is None:
                    continue
                try:
                    speed    = float(fused['speed'])
                    course   = float(fused['course'])
                    type_lbl = get_type_label(fused['type'] if 'type' in fused.index else 0)
                    color    = get_type_color(type_lbl)
                    dist_m   = geodesic((self.lat_cam, self.lon_cam), (lat, lon)).m
                except:
                    continue

                px, py = gps_to_minimap(lon, lat, self.lon_cam, self.lat_cam,
                                        self.scale, self.mm_cx, self.mm_cy)
                if not self._in_canvas(px, py, pad=6):
                    continue

                # trail
                if mmsi not in self.trails:
                    self.trails[mmsi] = []
                self.trails[mmsi].append((px, py))
                if len(self.trails[mmsi]) > self.trail_len:
                    self.trails[mmsi].pop(0)
                trail = self.trails[mmsi]
                for k in range(1, len(trail)):
                    alpha = int(200 * k / len(trail))
                    tc    = tuple(int(c * alpha / 200) for c in color)
                    cv2.line(mm, trail[k-1], trail[k], tc, 1, cv2.LINE_AA)

                arrow_len = max(14, int(speed * 3))
                draw_arrow(mm, px, py, course, arrow_len, color, thickness=1)

                cv2.circle(mm, (px, py), 6, color,           -1, cv2.LINE_AA)
                cv2.circle(mm, (px, py), 7, (255, 255, 255),  1, cv2.LINE_AA)

                lx = px + 10
                if lx + 88 > self.size:
                    lx = px - 90
                label_block(mm, lx, py - 4, [
                    f'MMSI:{mmsi}',
                    f'{type_lbl}',
                    f'SPD:{speed:.1f}kn',
                    f'COG:{course:.0f}deg',
                    f'DST:{dist_m:.0f}m',
                ], color, self.size)

        # ══ LAYER 3 — Visual-only ships (no AIS) ═══════════════════
        NO_AIS_COLOR = (0, 110, 255)   # orange-red

        if Vis_cur is not None and len(Vis_cur) > 0:
            for _, vrow in Vis_cur.iterrows():
                try:
                    track_id = int(vrow['ID'])
                except:
                    continue
                if track_id in fused_ids:
                    continue

                # bottom-centre of bbox = waterline contact
                try:
                    bx = int((int(vrow['x1']) + int(vrow['x2'])) // 2)
                    by = int(vrow['y2'])
                except:
                    try:
                        bx = int(vrow['x'])
                        by = int(vrow['y'])
                    except:
                        continue

                lon_s, lat_s = pixel_to_gps(bx, by, self.camera_para)
                if lon_s is None:
                    continue

                px, py = gps_to_minimap(lon_s, lat_s,
                                        self.lon_cam, self.lat_cam,
                                        self.scale, self.mm_cx, self.mm_cy)
                if not self._in_canvas(px, py, pad=4):
                    continue

                cv2.circle(mm, (px, py), 6, NO_AIS_COLOR, 1, cv2.LINE_AA)
                cv2.circle(mm, (px, py), 3, NO_AIS_COLOR, -1, cv2.LINE_AA)

                lx = px + 9
                if lx + 72 > self.size:
                    lx = px - 74
                label_block(mm, lx, py - 2,
                            [f'ID:{track_id}', 'NO AIS'],
                            NO_AIS_COLOR, self.size, line_h=12)

        # ══ UI chrome ═══════════════════════════════════════════════
        cv2.putText(mm, f'RADAR   T={timestamp//1000}s',
                    (5, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.34, (130, 160, 130), 1, cv2.LINE_AA)

        # legend
        ly = self.size - 42
        for col, txt in [
            ((255, 255, 255), 'Fused'),
            ((75,  75,   75), 'AIS only'),
            (NO_AIS_COLOR,    'Visual only (no AIS)'),
        ]:
            cv2.circle(mm, (8, ly), 4, col, -1, cv2.LINE_AA)
            cv2.putText(mm, txt, (16, ly + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.27, col, 1, cv2.LINE_AA)
            ly += 14

        cv2.rectangle(mm, (0, 0), (self.size-1, self.size-1), (70, 100, 70), 1)

        # ══ blend onto frame ════════════════════════════════════════
        x0, y0 = self.x0, self.y0
        roi = frame[y0:y0 + self.size, x0:x0 + self.size]
        if roi.shape[:2] == mm.shape[:2]:
            frame[y0:y0 + self.size, x0:x0 + self.size] = \
                cv2.addWeighted(roi, 0.20, mm, 0.80, 0)

        return frame
