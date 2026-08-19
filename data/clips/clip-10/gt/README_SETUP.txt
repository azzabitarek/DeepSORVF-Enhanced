=== SETUP CLIP MVI_1523_NIR pour DeepSORVF ===

1. CRÉER LE DOSSIER CLIP
   Créer: C:\Users\alach\OneDrive\Desktop\modele_1\goodluck\clip-XX\
   (remplacer XX par le numéro de clip suivant dans ton dataset)

2. STRUCTURE DU CLIP
   clip-XX\
   ├── ais\                          ← copier les 20 fichiers CSV ici
   ├── gt\                           ← copier les 3 fichiers GT ici
   │   ├── gt_detection.txt
   │   ├── gt_tracking.txt
   │   └── gt_fusion.txt
   ├── 2022_10_15_08_30_00_08_30_20.avi   ← RENOMMER MVI_1523_NIR.avi
   └── camera_para.txt               ← copier ici

3. RENOMMER LA VIDÉO
   MVI_1523_NIR.avi  →  2022_10_15_08_30_00_08_30_20.avi
   (format: YYYY_MM_DD_HH_MM_SS_HH_MM_SS)

4. PARAMÈTRES CAMÉRA (Singapore Strait)
   Format: [lon, lat, heading, pitch, roll, fov_h, alt, fx, fy, cx, cy]
   lon=103.79, lat=1.288 (Détroit de Singapour)
   heading=280°, pitch=-4°, FOV=50°, alt=20m

5. MMSI SYNTHÉTIQUES ASSIGNÉS
   MMSI 563000001 → Track 10 (Grand navire RoRo/car carrier - PRINCIPAL)
   MMSI 563000002 → Track 1  (Navire gauche, x: 0-447)
   MMSI 563000003 → Track 2  (Navire gauche fixe, x: 87-106)
   MMSI 563000004 → Track 3  (Navire gauche moyen, x: 146-152)
   MMSI 563000005 → Track 4  (Navire gauche mobile, x: 0-673)
   MMSI 563000006 → Track 5  (Navire centre-droit, x: 982-1074)
   MMSI 563000007 → Track 7  (Navire droite, x: 1302-1315)
   MMSI 563000008 → Track 8  (Navire droite, x: 1380-1398)
   MMSI 563000009 → Track 9  (Navire extrême droite, x: 1701-1730)
   MMSI 563000010 → Track 6  (Navire intermittent)

6. STATISTIQUES GT
   gt_detection.txt : 5343 lignes  (600 frames × ~9 objets)
   gt_tracking.txt  : 5022 lignes  (10 tracks × ~500 frames actives)
   gt_fusion.txt    : 5022 lignes  (même structure, MMSI au lieu track_id)
   AIS CSVs         : 20 fichiers  (1 par seconde, 2022_10_15_08_30_00 → 08_30_19)

NOTE: Les coordonnées GPS AIS sont synthétiques (rétro-projection estimée).
      La fusion AIS+visuelle devrait fonctionner pour le navire principal (MMSI 563000001).
