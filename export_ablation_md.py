import zipfile, json, os

z = zipfile.ZipFile(r'F:\MyWork\Article02\Pfe\ablation_results-20260819T204150Z-1-001.zip')

md = []
md.append("# DeepSORVF Ablation Study — Full Results\n")
md.append("## Description de l'expérience\n")
md.append("""
### Objectif
Étude d'ablation du pipeline DeepSORVF Enhanced pour la détection et fusion AIS/VIS de navires.
Chaque configuration désactive un composant du pipeline pour mesurer sa contribution.

### Pipeline complet (C5 = production)
```
Video → YOLOX (détection navires) → KOLOMVERSE (ensemble NMS) → Static Filter → AIS Fusion (DTW+Hungarian) → OAR (anti-occlusion)
```

### Configurations ablation

| Config | YOLOX | KOLOMVERSE | Static Filter | AIS Fusion | OAR (anti) | DTW | Angle Penalty | Binding | Hungarian |
|--------|-------|------------|---------------|------------|------------|-----|---------------|---------|-----------|
| C0 | ✓ | ✗ | ✗ | ✗ | off | - | - | - | - |
| C1 | ✓ | ✓ | ✗ | ✗ | off | - | - | - | - |
| C2 | ✓ | ✓ | ✓ | ✗ | off | - | - | - | - |
| C3 | ✓ | ✓ | ✓ | ✓ | off(0) | ✓ | ✓ | ✓ | ✓ |
| C4 | ✓ | ✓ | ✓ | ✓ | on(1) | ✓ | ✓ | ✓ | ✓ |
| C5 | ✓ | ✓ | ✓ | ✓ | on(1) | ✓ | ✓ | ✓ | ✓ |
| C3a | ✓ | ✓ | ✓ | ✓ | off(0) | ✗ | ✓ | ✓ | ✓ |
| C3b | ✓ | ✓ | ✓ | ✓ | off(0) | ✓ | ✗ | ✓ | ✓ |
| C3c | ✓ | ✓ | ✓ | ✓ | off(0) | ✓ | ✓ | ✗ | ✓ |
| C3d | ✓ | ✓ | ✓ | ✓ | off(0) | ✓ | ✓ | ✓ | ✗ (greedy) |

### Jeux de données (7 clips)

| Clip | Frames | FPS | Résolution | Durée | Description |
|------|--------|-----|------------|-------|-------------|
| clip-01 | 1979 | 25 | 2560×1440 | ~79s | Vue côtière, 2-3 navires |
| clip-02 | 502 | 25 | 2560×1440 | ~20s | Court, 1 navire |
| clip-10 | 600 | 25 | 2560×1440 | ~24s | Format CSV AIS datetime (converti en epoch ms) |
| Video-10 | 604 | 25 | ? | ~24s | 1 navire |
| Video-28 | 35950 | 25 | ? | ~1438s | Plus long, 9 MMSI différents |
| Video-29 | 440 | 25 | ? | ~18s | Court, 1 navire |
| Video-34 | 417 | 25 | ? | ~17s | Court |

### Environnement
- **Local**: Windows, CPU-only (torch 2.13.0+cpu), ~15-20 ms/frame
- **Colab**: GPU (T4), ~8-15 ms/frame
- **DeepSORT**: reinit tous les 25 frames (Windows), persistant (Colab/Linux)
- **ReID**: activé (use_reid=True)
- **Seuil binding (bin_num)**: 1 (1er lock après 2 matchs consécutifs)
- **Fenêtre oubli (fog_num)**: 15 secondes

### Métriques
- **detection_seconds**: nombre de secondes avec détections
- **fusion_count**: nombre de lignes de fusion (AIS↔VIS appariées)
- **detection_fusion_rate**: fusion_count / detection_count
- **match_count**: nombre de matchs consécutifs pour une paire AIS↔VIS
- **is_new_lock**: True quand une paire entre pour la 1ère fois dans bin_cur (seuil atteint)
""")

# ── PHASE 1 ──
md.append("\n---\n# PHASE 1 — C0, C1, C2 (sans AIS)\n")

summary1 = json.loads(z.read('ablation_results/phase1/phase1_summary.json'))
md.append("## Résumé Phase 1\n")
md.append("| Config | Clip | Frames | Det-Sec | Wall Time | ms/frame | Erreur |")
md.append("|--------|------|--------|---------|-----------|----------|--------|")
for r in summary1:
    if 'error' in r:
        md.append(f"| {r['config']} | {r['clip']} | - | - | - | - | {r['error']} |")
    else:
        md.append(f"| {r['config']} | {r['clip']} | {r['total_frames']} | {r['detection_seconds']} | {r['wall_time_s']}s | {r['avg_ms_per_frame']} | - |")

# Phase 1 metric files
md.append("\n## Fichiers métriques Phase 1\n")
for name in sorted(z.namelist()):
    if name.startswith('ablation_results/phase1/') and name.endswith('_detection.txt'):
        content = z.read(name).decode('utf-8', errors='replace').strip()
        lines = [l for l in content.split('\n') if l.strip()]
        parts = name.split('/')
        clip, config = parts[2], parts[3]
        md.append(f"\n### {config} / {clip} — Detection ({len(lines)} lignes)\n")
        if len(lines) <= 20:
            md.append("```")
            md.append(content)
            md.append("```")
        else:
            md.append("```")
            md.append('\n'.join(lines[:10]))
            md.append(f"... ({len(lines)} lignes au total)")
            md.append('\n'.join(lines[-5:]))
            md.append("```")

    if name.startswith('ablation_results/phase1/') and name.endswith('_tracking.txt'):
        content = z.read(name).decode('utf-8', errors='replace').strip()
        lines = [l for l in content.split('\n') if l.strip()]
        parts = name.split('/')
        clip, config = parts[2], parts[3]
        md.append(f"\n### {config} / {clip} — Tracking ({len(lines)} lignes)\n")
        if len(lines) <= 20:
            md.append("```")
            md.append(content)
            md.append("```")
        else:
            md.append("```")
            md.append('\n'.join(lines[:10]))
            md.append(f"... ({len(lines)} lignes au total)")
            md.append('\n'.join(lines[-5:]))
            md.append("```")

# ── PHASE 2 ──
md.append("\n---\n# PHASE 2 — C3, C4, C5 (avec AIS)\n")

summary2 = json.loads(z.read('ablation_results/phase2/phase2_summary.json'))
md.append("## Résumé Phase 2\n")
md.append("| Config | Clip | Frames | Det-Sec | Wall Time | ms/frame | Erreur |")
md.append("|--------|------|--------|---------|-----------|----------|--------|")
for r in summary2:
    if 'error' in r:
        md.append(f"| {r['config']} | {r['clip']} | - | - | - | - | {r['error']} |")
    else:
        md.append(f"| {r['config']} | {r['clip']} | {r['total_frames']} | {r['detection_seconds']} | {r['wall_time_s']}s | {r['avg_ms_per_frame']} | - |")

# Phase 2 metric files
md.append("\n## Fichiers métriques Phase 2\n")
for name in sorted(z.namelist()):
    if name.startswith('ablation_results/phase2/') and name.endswith('_detection.txt'):
        content = z.read(name).decode('utf-8', errors='replace').strip()
        lines = [l for l in content.split('\n') if l.strip()]
        parts = name.split('/')
        clip, config = parts[2], parts[3]
        md.append(f"\n### {config} / {clip} — Detection ({len(lines)} lignes)\n")
        if len(lines) <= 20:
            md.append("```")
            md.append(content)
            md.append("```")
        else:
            md.append("```")
            md.append('\n'.join(lines[:10]))
            md.append(f"... ({len(lines)} lignes au total)")
            md.append('\n'.join(lines[-5:]))
            md.append("```")

    if name.startswith('ablation_results/phase2/') and name.endswith('_tracking.txt'):
        content = z.read(name).decode('utf-8', errors='replace').strip()
        lines = [l for l in content.split('\n') if l.strip()]
        parts = name.split('/')
        clip, config = parts[2], parts[3]
        md.append(f"\n### {config} / {clip} — Tracking ({len(lines)} lignes)\n")
        if len(lines) <= 20:
            md.append("```")
            md.append(content)
            md.append("```")
        else:
            md.append("```")
            md.append('\n'.join(lines[:10]))
            md.append(f"... ({len(lines)} lignes au total)")
            md.append('\n'.join(lines[-5:]))
            md.append("```")

    if name.startswith('ablation_results/phase2/') and name.endswith('_fusion.txt'):
        content = z.read(name).decode('utf-8', errors='replace').strip()
        lines = [l for l in content.split('\n') if l.strip()]
        parts = name.split('/')
        clip, config = parts[2], parts[3]
        md.append(f"\n### {config} / {clip} — Fusion ({len(lines)} lignes)\n")
        if len(lines) <= 30:
            md.append("```")
            md.append(content)
            md.append("```")
        else:
            md.append("```")
            md.append('\n'.join(lines[:15]))
            md.append(f"... ({len(lines)} lignes au total)")
            md.append('\n'.join(lines[-10:]))
            md.append("```")

# ── ABLATION LOGS ──
md.append("\n---\n# ABLATION LOGS (match_count / is_new_lock)\n")

for name in sorted(z.namelist()):
    if name.endswith('_ablation_log.csv'):
        content = z.read(name).decode('utf-8', errors='replace').strip()
        lines = content.split('\n')
        parts = name.split('/')
        phase, clip, config = parts[1], parts[2], parts[3]
        md.append(f"\n## {phase} / {config} / {clip} ({len(lines)-1} lignes)\n")
        if len(lines) <= 30:
            md.append("```csv")
            md.append(content)
            md.append("```")
        else:
            md.append("```csv")
            md.append('\n'.join(lines[:15]))
            md.append(f"... ({len(lines)-1} lignes au total)")
            md.append('\n'.join(lines[-10:]))
            md.append("```")
        # Stats
        if len(lines) > 1:
            import csv as csvmod
            import io
            reader = csvmod.DictReader(io.StringIO(content))
            rows = list(reader)
            total = len(rows)
            new_locks = sum(1 for r in rows if r['is_new_lock'] == 'True')
            unique_mmsi = len(set(r['mmsi'] for r in rows))
            match_counts = [int(r['match_count']) for r in rows]
            md.append(f"\n**Stats**: {total} entrées, {new_locks} is_new_lock=True, {unique_mmsi} MMSI uniques, match_count min={min(match_counts) if match_counts else 0} max={max(match_counts) if match_counts else 0} avg={sum(match_counts)/len(match_counts) if match_counts else 0:.1f}\n")

# ── KNOWN ISSUES ──
md.append("\n---\n# PROBLÈMES CONNUS\n")
md.append("""
1. **clip-10**: Erreur `float division by zero` dans les stats — les timestamps CSV datetime ont été convertis en epoch ms mais les anciens résultats datent d'avant la conversion
2. **is_new_lock toujours False**: Le bug a été corrigé (commit f491491) — le fix track les paires entrant dans `bin_cur` au lieu de vérifier `match == bin_num+1`
3. **Reinit every 25 frames** (Windows): Réinitialise `mat_las` à chaque seconde de détection, empêchant l'accumulation de `match_count`. Désactivé sur Colab/Linux
4. **Static Filter**: Aucune détection supprimée — seuils peut-être trop permissifs
5. **Detection = Tracking**: Le tracker ne filtre rien — seuils de confiance trop permissifs
""")

# Write
output = '\n'.join(md)
out_path = r'F:\MyWork\Article02\Pfe\ablation_results_FULL.md'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(output)
print(f'Written: {out_path}')
print(f'Size: {len(output)} chars, {len(md)} lines')
