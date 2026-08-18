import os, time, imutils, cv2, argparse
import pandas as pd
import numpy as np
from utils.file_read import read_all, ais_initial, update_time, time2stamp

from utils.VIS_utils import VISPRO, WakeDetector
from utils.AIS_utils import AISPRO
from utils.FUS_utils import FUSPRO
from utils.gen_result import gen_result
from utils.draw import DRAW
from utils.minimap import MINIMAP
import imageio
from utils.detection_logger import DetectionLogger


def main(arg):
    ais_file, timestamp0, time0 = ais_initial(arg.ais_path, arg.initial_time)
    Time = arg.initial_time.copy()

    cap      = cv2.VideoCapture(arg.video_path)
    im_shape = [cap.get(3), cap.get(4)]
    max_dis  = min(im_shape) // 2
    fps      = int(cap.get(5))
    t        = int(1000 / fps)

    # AIS max detection radius — radar range synced to same value
    AIS_MAX_DIS = 4 * 1852   # metres

    clip_name = os.path.splitext(os.path.basename(arg.result_video))[0]

    # ── Detection logger ────────────────────────────────────────────────────
    _result_dir = os.path.dirname(arg.result_video) or '.'
    LOGGER = DetectionLogger(
        result_dir=_result_dir,
        clip_name=clip_name
    )

    AIS  = AISPRO(arg.ais_path, ais_file, im_shape, t)
    VIS  = VISPRO(arg.anti, arg.anti_rate, t)
    FUS  = FUSPRO(max_dis, im_shape, t)
    DRA  = DRAW(im_shape, t)
    WAKE = WakeDetector(shoot_hdir=camera_para[2])
    MM   = MINIMAP(camera_para, im_shape,
                   range_m     = AIS_MAX_DIS,
                   map_size    = 700,
                   result_path = os.path.dirname(arg.result_video),
                   clip_name   = clip_name,
                   fps         = fps,
                   cpa_warn_m  = 400,
                   cpa_warn_s  = 120)

    name = 'DeepSORVF'

    # Détection automatique de la taille de l'écran
    import tkinter as tk
    _root = tk.Tk()
    SCREEN_W = _root.winfo_screenwidth()
    SCREEN_H = _root.winfo_screenheight()
    _root.destroy()

    # Fenêtre d'affichage en plein écran
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow(name, SCREEN_W, SCREEN_H)

    videoWriter = None
    bin_inf     = pd.DataFrame(columns=['ID', 'mmsi', 'timestamp', 'match'])

    print('Start Time: %s || Stamp: %d || fps: %d' % (time0, timestamp0, fps))
    print(f'Screen resolution detected: {SCREEN_W}x{SCREEN_H}')
    times  = 0
    time_i = 0
    sum_t  = []

    try:
      while True:
        _, im = cap.read()
        if im is None:
            break
        start = time.time()

        Time, timestamp, Time_name = update_time(Time, t)

        AIS_vis, AIS_cur = AIS.process(camera_para, timestamp, Time_name)
        Vis_tra, Vis_cur = VIS.feedCap(im, timestamp, AIS_vis, bin_inf)
        Fus_tra, bin_inf = FUS.fusion(AIS_vis, AIS_cur, Vis_tra, Vis_cur, timestamp)
        wake_angles      = WAKE.update(im, Vis_cur, Vis_tra)

        end    = time.time() - start
        time_i = time_i + end
        if timestamp % 1000 < t:
            gen_result(times, Vis_cur, Fus_tra, arg.result_metric, im_shape)

            # ── Log detections from both models ──────────────────────────────
            yolox_boxes = [(int(row.x1), int(row.y1), int(row.x2), int(row.y2),
                            'vessel', 1.0)
                           for _, row in Vis_cur.iterrows()] if len(Vis_cur) else []
            LOGGER.log_frame(
                timestamp   = timestamp,
                frame_idx   = times,
                time_name   = Time_name,
                yolox_bboxes               = yolox_boxes,
                maritime_bboxes_kept       = VIS.maritime_bboxes,
                maritime_bboxes_suppressed = VIS.maritime_bboxes_suppressed
            )
            times = times + 1
            sum_t.append(time_i)
            print('Time: %s || Stamp: %d || Process: %.4f || Avg: %.4f' % (
                Time_name, timestamp, time_i, np.mean(sum_t)))
            time_i = 0

        # Draw YOLOv8 KOLOMVERSE maritime detections (orange, display only)
        for x1, y1, x2, y2, cls_name, conf in VIS.maritime_bboxes:
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 165, 255), 2)  # orange
            label = f'{cls_name} {float(conf):.2f}'
            cv2.putText(im, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2, cv2.LINE_AA)

        im = DRA.draw_traj(im, AIS_vis, AIS_cur, Vis_tra, Vis_cur, Fus_tra, timestamp)
        im = MM.draw(im, AIS_vis, AIS_cur, Fus_tra, Vis_tra, Vis_cur,
                     timestamp, wake_angles=wake_angles)

        # Enregistrement vidéo à la résolution native
        if videoWriter is None:
            fourcc      = cv2.VideoWriter_fourcc('m','p','4','v')
            videoWriter = cv2.VideoWriter(
                arg.result_video, fourcc, fps,
                (im.shape[1], im.shape[0]))
        videoWriter.write(im)

        # Affichage : redimensionnement à la taille écran complète (étiré)
        display_frame = cv2.resize(im, (SCREEN_W, SCREEN_H), interpolation=cv2.INTER_LINEAR)

        cv2.imshow(name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Échap pour quitter
            break
if cv2.getWindowProperty(name, cv2.WND_PROP_FULLSCREEN) < 0:
            break

    except Exception as _loop_err:
        import traceback
        print(f'[ERROR] Simulation crashed: {_loop_err}')
        print(traceback.format_exc())
    finally:
        cap.release()
        if videoWriter:
            videoWriter.release()
        MM.release()
        cv2.destroyAllWindows()
        LOGGER.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepSORVF Enhanced')
    parser.add_argument('--anti',        type=int, default=1)
    parser.add_argument('--anti_rate',   type=int, default=0)
    parser.add_argument('--data_path',   type=str, default='./clip-01/')
    parser.add_argument('--result_path', type=str, default='./result/')

    video_path, ais_path, result_video, result_metric, initial_time, \
        camera_para = read_all(parser.parse_args().data_path,
                               parser.parse_args().result_path)

    parser.add_argument('--video_path',    type=str,  default=video_path)
    parser.add_argument('--ais_path',      type=str,  default=ais_path)
    parser.add_argument('--result_video',  type=str,  default=result_video)
    parser.add_argument('--result_metric', type=str,  default=result_metric)
    parser.add_argument('--initial_time',  type=list, default=initial_time)
    parser.add_argument('--camera_para',   type=list, default=camera_para)

    argspar = parser.parse_args()
    print('\nDeepSORVF — Enhanced')
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    arg = parser.parse_args()
    main(arg)