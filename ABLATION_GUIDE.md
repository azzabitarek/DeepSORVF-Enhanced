# DeepSORVF — Ablation Study Guide

## Architecture du projet

```
DeepSORVF_Project/
├── data/
│   ├── clips/                    ← vidéos + données AIS
│   │   ├── clip-01/              ← clip test principal
│   │   │   ├── *.mp4             ← vidéo source
│   │   │   ├── camera_para.txt   ← paramètres caméra
│   │   │   ├── ais/              ← données AIS (1 CSV/seconde)
│   │   │   └── gt/               ← ground truth (si disponible)
│   │   ├── clip-02/
│   │   ├── clip-10/
│   │   ├── Video-10/
│   │   ├── Video-28/
│   │   ├── Video-29/
│   │   └── Video-34/
│   ├── frames_cache/             ← cache inter-sessions
│   └── results/
│
├── config/
│   ├── paths.yaml                ← chemins centralisés
│   └── pipeline_config.yaml      ← paramètres pipeline
│
├── utils/
│   ├── AIS_utils.py              ← traitement données AIS
│   ├── VIS_utils.py              ← détection YOLOX + YOLOv8 + DeepSORT
│   ├── FUS_utils.py              ← fusion AIS/VIS (DTW + Hungarian)
│   ├── file_read.py              ← lecture clips + horodatage
│   ├── gen_result.py             ← écriture résultats
│   └── draw.py                   ← dessin boîtes
│
├── detection_yolox/              ← détecteur navires
├── detection_yolov8/             ← détecteur objets marins (KOLOMVERSE)
├── deep_sort/                    ← tracker DeepSORT
├── weights/
│   ├── best.pt                   ← YOLOv8 (42 MB)
│   ├── YOLOX-final.pth           ← YOLOX (34 MB)
│   └── ckpt.t7                   ← ReID DeepSORT (44 MB)
│
├── run_ablation.py               ← runner unitaire (1 config, 1 clip)
├── ablation_runner.py            ← runner batch (toutes configs × clips)
├── setup_colab.ipynb             ← notebook Colab (2 phases)
└── package_for_colab.py          ← crée le zip Colab
```

## Pipeline de traitement

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  AIS_utils   │    │  VIS_utils   │    │  FUS_utils   │
│  (AISPRO)    │    │  (VISPRO)    │    │  (FUSPRO)    │
├─────────────┤    ├─────────────┤    ├─────────────┤
│ - Lecture    │    │ - YOLOX     │    │ - DTW        │
│   AIS CSV    │    │ - YOLOv8    │    │ - Hungarian  │
│ - Kalman     │    │ - DeepSORT  │    │ - Binding    │
│   filter     │    │ - Ensemble  │    │ - OAR        │
│ - Projection │    │ - Static    │    │   anti-occl.  │
│   caméra     │    │   filter    │    │              │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │
       └──────────────────┴──────────────────┘
                          │
                    ┌─────▼─────┐
                    │  gen_result │
                    │  (CSV)      │
                    └─────────────┘
```

## Configurations d'ablation

| Config | YOLOX | Ensemble | Static Filter | AIS | OAR | Description |
|--------|-------|----------|---------------|-----|-----|-------------|
| **C0** | ✓ | ✗ | ✗ | ✗ | ✗ | YOLOX seul (baseline) |
| **C1** | ✓ | ✓ | ✗ | ✗ | ✗ | + KOLOMVERSE maritime |
| **C2** | ✓ | ✓ | ✓ | ✗ | ✗ | + filtre structures fixes |
| **C3** | ✓ | ✓ | ✓ | ✓ | ✗ | + fusion AIS (sans OAR) |
| **C4** | ✓ | ✓ | ✓ | ✓ | ✓ | + OAR anti-occlusion |
| **C5** | ✓ | ✓ | ✓ | ✓ | ✓ | Modèle complet (= C4) |

## Phases d'exécution sur Colab

### Phase 1 : C0, C1, C2 (sans AIS)

```
Pour chaque clip (7 clips) :
  Pour chaque config (C0, C1, C2) :
    → run_pipeline(clip, config, ais_enabled=False)
    → Sauvegarde dans ablation_results/phase1/{clip}/{config}/
```

**Durée estimée** : ~1.5 heure (GPU)

### Phase 2 : C3, C4, C5 (avec AIS)

```
Pour chaque clip (7 clips) :
  Pour chaque config (C3, C4, C5) :
    → run_pipeline(clip, config, ais_enabled=True)
    → Sauvegarde dans ablation_results/phase2/{clip}/{config}/
```

**Durée estimée** : ~2.5 heures (GPU)

**Total** : ~4 heures sur Colab GPU

## Fichiers de résultats

Chaque exécution produit :

```
ablation_results/
├── phase1/
│   ├── clip-01/
│   │   ├── C0/
│   │   │   ├── metric/
│   │   │   │   ├── clip-01_detection.txt    ← détections YOLOX
│   │   │   │   ├── clip-01_tracking.txt     ← pistes DeepSORT
│   │   │   │   └── clip-01_fusion.txt       ← fusion AIS/VIS
│   │   │   └── clip-01_C0_ablation_log.csv  ← logs détaillés
│   │   ├── C1/
│   │   └── C2/
│   └── phase1_summary.json                  ← stats résumées
└── phase2/
    ├── clip-01/
    │   ├── C3/
    │   ├── C4/
    │   └── C5/
    └── phase2_summary.json
```

### Format des métriques

Fichier `*_detection.txt` :
```
ID, timestamp, x1, y1, x2, y2, confidence
```

Fichier `*_tracking.txt` :
```
ID, timestamp, x1, y1, x2, y2, confidence
```

Fichier `*_fusion.txt` :
```
ID, mmsi, timestamp, x1, y1, x2, y2, match_score
```

### Format du summary JSON

```json
{
  "config": "C5",
  "clip": "clip-01",
  "total_frames": 1979,
  "detection_seconds": 79,
  "wall_time_s": 450.2,
  "avg_ms_per_frame": 32.0
}
```

## Paramètres clés

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| FPS | 25 | Fréquence d'image source |
| Detection interval | 1 frame/seconde | YOLOX s'exécute toutes les secondes |
| DeepSORT n_init | 1 | Tracks confirmés après 1 détection |
| ReID | Désactivé | Suivi spatial seul (Kalman + IoU) |
| Reinit | Toutes les 25 frames | Prévention crash Windows |
| DTW compression | Oui | Réduction trajectoires pour vitesse |
| Binding threshold | 1 | Verrouillage après 1 match consécutif |

## Structures testées (clips)

| Clip | Frames | Durée | Résolution | Description |
|------|--------|-------|------------|-------------|
| clip-01 | 1979 | 79s | 2560×1440 | Clip principal (vent par clair) |
| clip-02 | - | - | - | Clip test secondaire |
| clip-10 | - | - | - | Clip test |
| Video-10 | - | - | - | Vidéo longue |
| Video-28 | - | - | - | Vidéo longue |
| Video-29 | - | - | - | Vidéo longue |
| Video-34 | - | - | - | Vidéo longue |

## Commandes rapides

```bash
# Exécuter une seule config sur un clip
python run_ablation.py --clip clip-01 --config C5 --max-frames 200

# Exécuter toutes les configs sur tous les clips
python ablation_runner.py

# Exécuter uniquement Phase 1 (C0-C2)
python ablation_runner.py --configs C0 C1 C2

# Exécuter uniquement Phase 2 (C3-C5)
python ablation_runner.py --configs C3 C4 C5

# Exécuter sur un seul clip
python ablation_runner.py --sequences clip-01

# Limiter les frames (test rapide)
python ablation_runner.py --max-frames 100
```

## Prérequis Colab

- **GPU requis** : Runtime → Change runtime type → GPU
- **Espace Drive** : ~500 MB (vidéos) + ~130 MB (zip)
- **Durée session** : ~4 heures (free) / ~24 heures (Pro)
