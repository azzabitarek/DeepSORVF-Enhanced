"""
convert_to_nir.py
Convertit une vidéo maritime couleur → style NIR proche (comme MVI_1523_NIR)
Usage: python convert_to_nir.py <video_input> [video_output]
Exemple: python convert_to_nir.py clip-01/2022_06_04_12_05_12_12_07_02.mp4
"""

import cv2
import numpy as np
import sys
import os

# ─── CONFIG ────────────────────────────────────────────────────────────────────
CLAHE_CLIP     = 3.0    # Contraste local (augmenter = plus de détail sur navires)
CLAHE_GRID     = (8, 8) # Taille grille CLAHE
NOISE_SIGMA    = 4      # Bruit capteur NIR (0 = aucun, 8 = fort)
GAMMA          = 1.3    # Correction gamma (>1 = plus sombre)
WATER_DARKEN   = 0.85   # Assombrir les zones sombres (eau) [0.5-1.0]
# ───────────────────────────────────────────────────────────────────────────────


def apply_gamma(img, gamma):
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def darken_water(gray):
    """Assombrit les pixels déjà sombres (eau) sans toucher les navires clairs."""
    mask = gray.astype(np.float32) / 255.0
    # Les pixels sombres (eau) sont encore assombris, les clairs restent clairs
    result = (mask ** (1.0 / WATER_DARKEN)) * 255.0
    return np.clip(result, 0, 255).astype(np.uint8)


def add_nir_noise(gray):
    """Ajoute du bruit gaussien typique capteur NIR."""
    if NOISE_SIGMA == 0:
        return gray
    noise = np.random.normal(0, NOISE_SIGMA, gray.shape).astype(np.float32)
    noisy = gray.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def convert_frame_to_nir(frame):
    # 1. Convertir en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. CLAHE - contraste adaptatif (fait ressortir les navires)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    gray = clahe.apply(gray)

    # 3. Assombrir l'eau
    gray = darken_water(gray)

    # 4. Correction gamma
    gray = apply_gamma(gray, GAMMA)

    # 5. Bruit capteur NIR
    gray = add_nir_noise(gray)

    # 6. Repasser en BGR pour écriture vidéo
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def convert_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir: {input_path}")
        sys.exit(1)

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    print(f"[INFO] {os.path.basename(input_path)}")
    print(f"       {W}x{H} @ {fps:.0f}fps  |  {total} frames")
    print(f"[INFO] Sortie → {output_path}")
    print(f"[INFO] Traitement en cours...")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        nir_frame = convert_frame_to_nir(frame)
        out.write(nir_frame)
        frame_idx += 1
        if frame_idx % 100 == 0:
            pct = frame_idx / total * 100
            print(f"       {frame_idx}/{total} frames ({pct:.0f}%)", end='\r')

    cap.release()
    out.release()
    print(f"\n[OK] Terminé ! Fichier: {output_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python convert_to_nir.py <video_input> [video_output]")
        print("Exemple: python convert_to_nir.py clip-01/2022_06_04_12_05_12_12_07_02.mp4")
        sys.exit(1)

    input_path = sys.argv[1]

    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        base, ext = os.path.splitext(input_path)
        output_path = base + '_NIR.avi'

    convert_video(input_path, output_path)
