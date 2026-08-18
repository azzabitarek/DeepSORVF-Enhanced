#!/usr/bin/env python3
"""
Fix clip-02 — Corrige les données AIS et le camera_para.txt
Basé sur la formule exacte de visual_transform() du modèle DeepSORVF.
Lance ce script UNE SEULE FOIS pour clip-02.
"""

import os, math
from math import radians, cos, sin, atan2, degrees, tan
import pyproj
import pandas as pd
from datetime import datetime, timedelta

# ══════════════════════════════════════════════════════════════
CLIP_DIR   = r"C:\Users\alach\OneDrive\Desktop\modele_1\goodluck\clip-02"
START_TIME = "2022-06-04 14:30:00"
DURATION_S = 120   # durée de la vidéo en secondes
# ══════════════════════════════════════════════════════════════

# Nouveaux paramètres caméra (shoot_vdir=-8 au lieu de -4)
CAM = [114.32722222222222, 30.60027777777778,
       352,     # heading
       -8,      # shoot_vdir CORRIGÉ pour scène maritime
       20,      # height_cam
       55,      # FOV_hor
       30.94,   # FOV_ver
       2391.26, 2446.89,   # fx, fy
       1305.04, 855.214]   # u0, v0

# Navires observés dans la vidéo — positions pixel dans la fenêtre 1920x1080
# (issues de la mesure Paint sur le résultat du modèle)
VESSELS = [
    {"mmsi":380000000,"u":210,"v":375,"speed":6.0,"type":18,"desc":"Grand cargo"},
    {"mmsi":290000000,"u":380,"v":375,"speed":3.0,"type":18,"desc":"Navire moyen"},
    {"mmsi":340000000,"u":530,"v":360,"speed":0.5,"type":18,"desc":"Petit navire"},
    {"mmsi":220000000,"u":820,"v":370,"speed":0.5,"type":18,"desc":"Grande barge"},
    {"mmsi":210000000,"u":940,"v":370,"speed":4.0,"type":1, "desc":"Remorqueur"},
]

KN_TO_MS = 0.514444


def inverse_project(u, v, cam):
    """Pixel (1920x1080) → GPS via formule exacte de visual_transform()."""
    sx = 1920/2610;  sy = 1080/1710
    shv_rad = radians(-cam[3])
    s = sin(shv_rad);  c = cos(shv_rad)
    px = (u - cam[9]*sx) / (cam[7]*sx)
    py = (v - cam[10]*sy) / (cam[8]*sy)
    denom = s + py*c
    if denom <= 0:
        raise ValueError(f"Vessel hors champ vertical (denom={denom:.4f})")
    Zw = cam[4] * (c - py*s) / denom
    hr = atan2(px, c - py*s)
    D  = Zw / cos(hr)
    brg = (cam[2] + degrees(hr) + 360) % 360
    g = pyproj.Geod(ellps="WGS84")
    lon, lat, _ = g.fwd(cam[0], cam[1], brg, D)
    return lon, lat, D


def move_vessel(lon, lat, speed_ms, course_deg, dt_s):
    """Déplace un navire sur dt_s secondes."""
    d = speed_ms * dt_s
    c = radians(course_deg)
    R = 6371000
    lr = radians(lat)
    dlon = d * sin(c) / (cos(lr) * R * math.pi/180)
    dlat = d * cos(c) / (R * math.pi/180)
    return lon + dlon, lat + dlat


def run():
    print(f"\n{'='*60}")
    print(f"  Fix clip-02")
    print(f"{'='*60}")

    # 1. Écrire le nouveau camera_para.txt
    cam_path = os.path.join(CLIP_DIR, "camera_para.txt")
    with open(cam_path, "w") as f:
        f.write(str(CAM))
    print(f"\n[1/3] camera_para.txt mis à jour (shoot_vdir={CAM[3]})")

    # 2. Calculer les positions GPS initiales depuis les pixels
    print(f"\n[2/3] Calcul des positions GPS initiales...")
    start_dt = datetime.strptime(START_TIME, "%Y-%m-%d %H:%M:%S")

    for v in VESSELS:
        lon, lat, dist = inverse_project(v['u'], v['v'], CAM)
        v['init_lon'] = lon
        v['init_lat'] = lat
        # Cap : traverse l'image → perpendiculaire au heading caméra
        v['course'] = (CAM[2] + 270) % 360  # traverse vers la gauche
        print(f"  {v['desc']:<15}: GPS=({lon:.5f},{lat:.5f}) "
              f"dist={dist:.0f}m cap={v['course']:.0f}°")

    # 3. Générer les CSVs AIS (1 par seconde)
    print(f"\n[3/3] Génération des CSVs AIS ({DURATION_S} secondes)...")
    ais_dir = os.path.join(CLIP_DIR, "ais")
    os.makedirs(ais_dir, exist_ok=True)

    # Supprimer les anciens CSVs
    import glob
    for f in glob.glob(os.path.join(ais_dir, "*.csv")):
        os.remove(f)

    cur = {v['mmsi']: {'lon':v['init_lon'], 'lat':v['init_lat'],
                       'speed':v['speed'], 'course':v['course'],
                       'type':v['type']} for v in VESSELS}

    for sec in range(DURATION_S):
        t = start_dt + timedelta(seconds=sec)
        rows = []
        for v in VESSELS:
            mmsi = v['mmsi']
            pos  = cur[mmsi]
            if sec > 0:
                nl, nla = move_vessel(pos['lon'], pos['lat'],
                                      pos['speed']*KN_TO_MS, pos['course'], 1.0)
                cur[mmsi]['lon'] = nl
                cur[mmsi]['lat'] = nla
            rows.append({
                'mmsi'     : mmsi,
                'lon'      : round(cur[mmsi]['lon'], 8),
                'lat'      : round(cur[mmsi]['lat'], 8),
                'speed'    : round(pos['speed'], 1),
                'course'   : round(pos['course'], 1),
                'heading'  : round(pos['course'], 1),
                'type'     : pos['type'],
                'timestamp': int(t.timestamp() * 1000),
            })
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(ais_dir, t.strftime("%Y_%m_%d_%H_%M_%S")+".csv"))

    print(f"  ✓ {DURATION_S} CSVs générés dans {ais_dir}")
    print(f"\n{'='*60}")
    print(f"  ✅  Terminé !")
    print(f"  Relance maintenant ton modèle sur clip-02.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run()
