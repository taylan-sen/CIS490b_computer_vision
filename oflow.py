#!/usr/bin/env python3
"""
Record a video from your webcam, then run Lucas-Kanade (sparse) optical flow on it.
 
Usage:
    python lk_optical_flow.py                      # record, then run optical flow
    python lk_optical_flow.py --input myvideo.mp4  # skip recording, use an existing video
    python lk_optical_flow.py --live               # run optical flow live on the webcam
    python lk_optical_flow.py --save-flow flow.mp4 # also save the annotated output
 
Recording controls:   SPACE = start/stop recording    Q or ESC = done
Optical flow controls: SPACE = pause/resume   R = re-detect points   C = clear trails   Q or ESC = quit
"""
 
import argparse
import sys
import time
 
import cv2
import numpy as np
 
# Shi-Tomasi corner detection: picks "good features to track"
FEATURE_PARAMS = dict(maxCorners=200, qualityLevel=0.2, minDistance=7, blockSize=7)
 
# Lucas-Kanade parameters
#   winSize  - size of the window around each point used to solve for motion
#   maxLevel - number of pyramid levels (lets LK handle larger motions)
#   criteria - stop iterating after 10 iterations or when movement < 0.03 px
LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)
 
MIN_POINTS = 20  # re-detect corners when fewer than this many are still tracked
 
 
def open_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {index}. Try --camera 1, or check camera permissions.")
    return cap
 
 
def record_video(camera_index, out_path):
    """Show a webcam preview; SPACE toggles recording, Q/ESC finishes."""
    cap = open_camera(camera_index)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1 or fps > 120:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
    writer = None
    frames_written = 0
    start_time = None
    window = "Recorder - SPACE start/stop, Q to finish"
 
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera stopped sending frames.")
            break
 
        if writer is not None:
            writer.write(frame)
            frames_written += 1
 
        preview = frame.copy()
        if writer is not None:
            elapsed = time.time() - start_time
            cv2.circle(preview, (25, 25), 10, (0, 0, 255), -1)
            cv2.putText(preview, f"REC {elapsed:5.1f}s", (45, 33),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(preview, "SPACE to record, Q to finish", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(window, preview)
 
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
                if not writer.isOpened():
                    sys.exit(f"Could not open {out_path} for writing.")
                start_time = time.time()
                print("Recording...")
            else:
                writer.release()
                writer = None
                print(f"Stopped. Saved {frames_written} frames to {out_path}")
        elif key in (ord("q"), 27):
            break
 
    if writer is not None:
        writer.release()
        print(f"Saved {frames_written} frames to {out_path}")
    cap.release()
    cv2.destroyAllWindows()
    return frames_written > 0
 
 
def detect_points(gray):
    return cv2.goodFeaturesToTrack(gray, mask=None, **FEATURE_PARAMS)
 
 
def run_lucas_kanade(source, save_path=None, loop_delay_ms=30):
    """Track Shi-Tomasi corners through the video with pyramidal Lucas-Kanade."""
    cap = open_camera(source) if isinstance(source, int) else cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Could not open video source: {source}")
 
    ok, old_frame = cap.read()
    if not ok:
        sys.exit("Video has no frames.")
 
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = detect_points(old_gray)
    trails = np.zeros_like(old_frame)          # where we draw the motion paths
    rng = np.random.default_rng(0)
    colors = rng.integers(0, 255, (FEATURE_PARAMS["maxCorners"], 3)).tolist()
 
    writer = None
    if save_path:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        h, w = old_frame.shape[:2]
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
 
    window = "Lucas-Kanade optical flow - SPACE pause, R redetect, C clear, Q quit"
    paused = False
    frame_idx = 0
 
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break  # end of video
            frame_idx += 1
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
            if p0 is None or len(p0) < MIN_POINTS:
                p0 = detect_points(old_gray)
 
            good_new = np.empty((0, 2))
            if p0 is not None and len(p0) > 0:
                # The core call: where did each point in old_gray move to in frame_gray?
                p1, status, _err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **LK_PARAMS)
                if p1 is not None:
                    found = status.ravel() == 1
                    good_new = p1[found].reshape(-1, 2)
                    good_old = p0[found].reshape(-1, 2)
 
                    trails = (trails * 0.95).astype(np.uint8)   # slowly fade old trails
                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        a, b = map(int, new)
                        c, d = map(int, old)
                        color = colors[i % len(colors)]
                        cv2.line(trails, (a, b), (c, d), color, 2)
                        cv2.circle(frame, (a, b), 4, color, -1)
 
            output = cv2.add(frame, trails)
            cv2.putText(output, f"frame {frame_idx}  points {len(good_new)}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if writer is not None:
                writer.write(output)
            cv2.imshow(window, output)
 
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2).astype(np.float32) if len(good_new) else None
 
        key = cv2.waitKey(1 if isinstance(source, int) else loop_delay_ms) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("r"):
            p0 = detect_points(old_gray)
            trails[:] = 0
        elif key == ord("c"):
            trails[:] = 0
 
    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved annotated video to {save_path}")
    cv2.destroyAllWindows()
 
 
def main():
    parser = argparse.ArgumentParser(description="Record video and run Lucas-Kanade optical flow.")
    parser.add_argument("--camera", type=int, default=1, help="camera index (default 0)")
    parser.add_argument("--output", default="recording.mp4", help="where to save the recording")
    parser.add_argument("--input", help="run optical flow on this video instead of recording")
    parser.add_argument("--live", action="store_true", help="run optical flow live on the webcam")
    parser.add_argument("--save-flow", help="save the annotated optical-flow video to this file")
    args = parser.parse_args()
 
    if args.live:
        run_lucas_kanade(args.camera, args.save_flow)
    elif args.input:
        run_lucas_kanade(args.input, args.save_flow)
    else:
        if record_video(args.camera, args.output):
            print("Running Lucas-Kanade optical flow on the recording...")
            run_lucas_kanade(args.output, args.save_flow)
        else:
            print("Nothing was recorded.")
 
 
if __name__ == "__main__":
    main()
 
