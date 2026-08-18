#!/usr/bin/env python3
"""
GT Generator Direct Pixel — clip-02
Génère les GT files directement depuis les positions pixel observées.
Pas besoin de projection GPS, fonctionne avec n'importe quel clip.
"""

import os
import math
import glob
import subprocess

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║              CONFIGURATION — basée sur les screenshots                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

CONFIG = {
    "clip_dir" : r"C:\Users\alach\OneDrive\Desktop\modele_1\goodluck\clip-02",
    "clip_name" : "clip-02",
    "img_w"     : 1920,
    "img_h"     : 1080,
    "fps"       : 30,

    # ── Navires observés dans la vidéo ──
    # u, v       : pixel du CENTRE du navire (frame 1)
    # w, h       : largeur et hauteur de la bounding box
    # vel_u      : vitesse horizontale en pixels/seconde
    #              (négatif = va vers la gauche)
    # vel_v      : vitesse verticale en pixels/seconde
    #              (positif = descend dans l'image)
    # mmsi       : identifiant AIS
    #
    #  Repère image :
    #   (0,0) ─────────────────→ u (droite)
    #     │
    #     ↓ v (bas)

    "vessels": [
        {
            "mmsi"  : 380000000,
            "u"     : 210,    # centre horizontal (pixel)
            "v"     : 375,    # centre vertical   (pixel)
            "w"     : 180,    # largeur bbox
            "h"     : 115,    # hauteur bbox
            "vel_u" : -2.0,   # se déplace légèrement vers la gauche
            "vel_v" :  0.5,   # descend très légèrement (vient vers caméra)
            "desc"  : "Grand cargo noir/blanc (gauche)",
        },
        {
            "mmsi"  : 290000000,
            "u"     : 380,
            "v"     : 375,
            "w"     : 90,
            "h"     : 60,
            "vel_u" : -1.0,
            "vel_v" :  0.0,
            "desc"  : "Navire moyen (centre-gauche)",
        },
        {
            "mmsi"  : 340000000,
            "u"     : 530,
            "v"     : 360,
            "w"     : 60,
            "h"     : 45,
            "vel_u" : -0.5,
            "vel_v" :  0.0,
            "desc"  : "Petit navire (centre)",
        },
        {
            "mmsi"  : 220000000,
            "u"     : 820,
            "v"     : 370,
            "w"     : 280,
            "h"     : 65,
            "vel_u" : -0.3,   # quasi immobile
            "vel_v" :  0.0,
            "desc"  : "Grande barge noire (droite)",
        },
        {
            "mmsi"  : 210000000,
            "u"     : 940,
            "v"     : 370,
            "w"     : 55,
            "h"     : 45,
            "vel_u" : -1.5,
            "vel_v" :  0.5,
            "desc"  : "Remorqueur rouge (droite)",
        },
    ],
}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         GÉNÉRATION GT                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def get_video_info(video_path):
    try:
        r = subprocess.run(
            ['ffprobe','-v','error','-select_streams','v:0',
             '-count_packets','-show_entries',
             'stream=nb_read_packets,r_frame_rate,width,height',
             '-of','csv=p=0', video_path],
            capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            p = r.stdout.strip().split(',')
            w,h = int(p[0]),int(p[1])
            n,d = map(int, p[2].split('/'))
            return int(p[3]), n/d, w, h
    except Exception:
        pass
    return None


def run(cfg):
    clip_dir  = cfg['clip_dir']
    clip_name = cfg['clip_name']
    fps       = cfg['fps']
    img_w     = cfg['img_w']
    img_h     = cfg['img_h']

    print(f"\n{'='*65}")
    print(f"  GT Direct Pixel  —  {clip_name}")
    print(f"{'='*65}")

    # Lire infos vidéo si disponible
    videos = (glob.glob(os.path.join(clip_dir,"*.mp4")) +
              glob.glob(os.path.join(clip_dir,"*.avi")))
    nb_frames = None
    if videos:
        info = get_video_info(videos[0])
        if info:
            nb_frames, fps, img_w, img_h = info
            print(f"[Video] {os.path.basename(videos[0])}  "
                  f"{img_w}x{img_h}  {fps:.0f}fps  {nb_frames} frames")

    if nb_frames is None:
        nb_frames = int(fps * 120)
        print(f"[Video] Non trouvee → {nb_frames} frames par défaut")

    # Préparer GT
    gt_dir = os.path.join(clip_dir, 'gt')
    os.makedirs(gt_dir, exist_ok=True)

    det_lines, trk_lines, fus_lines = [], [], []
    mmsi_to_tid = {v['mmsi']: i for i, v in enumerate(cfg['vessels'])}
    cnt = {v['mmsi']: 0 for v in cfg['vessels']}

    print(f"\n[Generation GT — {nb_frames} frames, {fps:.0f}fps]")

    for frame_id in range(2, nb_frames + 2):
        t = (frame_id - 2) / fps   # temps en secondes depuis frame 1

        for v in cfg['vessels']:
            mmsi = v['mmsi']

            # Position à ce frame
            u = v['u'] + v['vel_u'] * t
            vp = v['v'] + v['vel_v'] * t
            w  = v['w']
            h  = v['h']

            # Coin haut-gauche
            x = int(u - w / 2)
            y = int(vp - h / 2)

            # Vérifier visibilité
            cx, cy = u, vp
            if not (-50 < cx < img_w + 50 and -50 < cy < img_h + 50):
                continue

            tid  = mmsi_to_tid[mmsi]
            sfx  = ",1,1,1,1"
            coord = f"{x},{y},{w},{h}"

            det_lines.append(f"{frame_id},0,{coord}{sfx}")
            trk_lines.append(f"{frame_id},{tid},{coord}{sfx}")
            fus_lines.append(f"{frame_id},{mmsi},{coord}{sfx}")
            cnt[mmsi] += 1

    # Écriture fichiers
    for label, lines in [('detection', det_lines),
                         ('tracking',  trk_lines),
                         ('fusion',    fus_lines)]:
        path = os.path.join(gt_dir, f"{clip_name}_gt_{label}.txt")
        with open(path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"  ✓ {clip_name}_gt_{label}.txt  ({len(lines)} lignes)")

    # Résumé
    print(f"\n{'='*65}")
    print(f"  Navires :")
    for v in cfg['vessels']:
        n   = cnt[v['mmsi']]
        pct = n / nb_frames * 100 if nb_frames > 0 else 0
        bar = '#' * int(pct/5) + '-' * (20 - int(pct/5))
        print(f"  [{bar}] {pct:.0f}%  {v['desc']}")
    print(f"\n  Fichiers GT dans : {gt_dir}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    run(CONFIG)
