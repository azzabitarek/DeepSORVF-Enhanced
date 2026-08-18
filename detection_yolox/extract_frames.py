import cv2
import os

VIDEO_DIR  = r"C:\Users\alach\OneDrive\Desktop\modele_1\VIS_Onshore\VIS_Onshore\Videos"
OUTPUT_DIR = r"C:\Users\alach\OneDrive\Desktop\modele_1\new\VOCdevkit_val\VOC2007\JPEGImages"
START_ID   = 100
os.makedirs(OUTPUT_DIR, exist_ok=True)
frame_id = START_ID
total_extracted = 0

for video_file in sorted(os.listdir(VIDEO_DIR)):
    if not video_file.lower().endswith(('.avi', '.mp4', '.mov', '.mkv')):
        continue

    video_path = os.path.join(VIDEO_DIR, video_file)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Impossible d'ouvrir : {video_file}")
        continue

    fps      = int(cap.get(cv2.CAP_PROP_FPS))
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(fps * 8, 1)

    print(f"Traitement : {video_file} — {total} frames — {fps} fps")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            out_path = os.path.join(OUTPUT_DIR, f"{frame_id}.png")
            cv2.imwrite(out_path, frame)
            frame_id += 1
            total_extracted += 1
        count += 1
    cap.release()

print(f"\nExtraction terminee")
print(f"Nouvelles frames : {total_extracted}")
print(f"IDs : {START_ID} a {frame_id - 1}")