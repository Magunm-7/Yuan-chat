import cv2
import numpy as np
import mediapipe as mp

def _sample_frames(cap, fps, t0, t1, sample_fps=5.0):
    step = max(1, int(fps / sample_fps))
    start = int(t0 * fps)
    end = int(t1 * fps)
    for fidx in range(start, end+1, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            continue
        yield frame

def face_visible_and_microexpr(mp4_path: str, t0: float, t1: float, sample_fps: float = 5.0):
    """Return (q_video, microexpr_rate) in [0,1] approximately.
    microexpr_rate is a proxy: average landmark motion magnitude per second normalized by face size.
    """
    cap = cv2.VideoCapture(mp4_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    mp_face = mp.solutions.face_mesh
    mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    total = 0
    visible = 0
    motions = []

    prev = None
    prev_scale = None

    for frame in _sample_frames(cap, fps, t0, t1, sample_fps=sample_fps):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mesh.process(rgb)
        total += 1
        if not res.multi_face_landmarks:
            prev = None
            prev_scale = None
            continue

        visible += 1
        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([(p.x, p.y) for p in lm], dtype=np.float32)

        # face scale as inter-ocular distance (approx)
        left = pts[33]   # approx left eye outer
        right = pts[263] # approx right eye outer
        scale = float(np.linalg.norm(left - right) + 1e-6)

        if prev is not None:
            # mean motion normalized by scale
            motion = float(np.mean(np.linalg.norm(pts - prev, axis=1)) / scale)
            motions.append(motion)

        prev = pts
        prev_scale = scale

    cap.release()

    q_video = (visible / total) if total > 0 else 0.0

    # map motion to [0,1] roughly
    if motions:
        m = float(np.mean(motions))
        microexpr_rate = float(np.clip(m / 0.015, 0.0, 1.0))  # 0.015 is heuristic
    else:
        microexpr_rate = 0.0

    return float(q_video), float(microexpr_rate)
