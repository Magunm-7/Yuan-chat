import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def _sample_frames(cap, fps, t0, t1, sample_fps=5.0):
    step = max(1, int(fps / sample_fps))
    start = int(t0 * fps)
    end = int(t1 * fps)
    for fidx in range(start, end + 1, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            continue
        ts_ms = int((fidx / fps) * 1000)
        yield frame, ts_ms

def face_visible_and_microexpr(
    mp4_path: str,
    t0: float,
    t1: float,
    sample_fps: float = 5.0,
    model_path: str = "/home/qmn/Yuan-chat/MPSE_FULL_LOOP_PROJECT/models/mediapipe/face_landmarker.task",
):
    cap = cv2.VideoCapture(mp4_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        # 如果你的模型支持，可开启输出表情/blendshapes等：
        # output_face_blendshapes=True,
    )

    total = 0
    visible = 0
    motions = []
    prev = None

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for frame, ts_ms in _sample_frames(cap, fps, t0, t1, sample_fps=sample_fps):
            total += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result = landmarker.detect_for_video(mp_image, ts_ms)

            # result.face_landmarks: List[List[NormalizedLandmark]]
            if not result.face_landmarks:
                prev = None
                continue

            visible += 1
            lm = result.face_landmarks[0]
            pts = np.array([(p.x, p.y) for p in lm], dtype=np.float32)

            # face scale as inter-ocular distance (same heuristic as你原来写的)
            left = pts[33]
            right = pts[263]
            scale = float(np.linalg.norm(left - right) + 1e-6)

            if prev is not None:
                motion = float(np.mean(np.linalg.norm(pts - prev, axis=1)) / scale)
                motions.append(motion)

            prev = pts

    cap.release()

    q_video = (visible / total) if total > 0 else 0.0
    if motions:
        m = float(np.mean(motions))
        microexpr_rate = float(np.clip(m / 0.015, 0.0, 1.0))
    else:
        microexpr_rate = 0.0

    return float(q_video), float(microexpr_rate)
