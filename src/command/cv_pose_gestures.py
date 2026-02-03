from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp_proc
import queue
import threading
import time
from typing import Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import pose_landmarker as pose_landmarker_lib

_DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


class CvPreviewer:
    def __init__(self, enabled: bool) -> None:
        self._enabled = enabled
        self._queue: mp_proc.Queue | None = None
        self._proc: mp_proc.Process | None = None

    def start(self) -> None:
        if not self._enabled:
            return
        if self._proc and self._proc.is_alive():
            return
        self._queue = mp_proc.Queue(maxsize=1)
        self._proc = mp_proc.Process(target=self._run, daemon=True)
        self._proc.start()

    def stop(self) -> None:
        if not self._enabled or not self._queue:
            return
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        if self._proc:
            self._proc.join(timeout=0.2)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=0.2)

    def show(self, frame) -> None:
        if not self._enabled or not self._queue:
            return
        try:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(frame)
        except Exception:
            pass

    def _run(self) -> None:
        import signal

        # Let the parent handle Ctrl-C; ignore SIGINT in the preview process.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        cv2.namedWindow("pose_gestures", cv2.WINDOW_NORMAL)
        while True:
            frame = self._queue.get()
            if frame is None:
                break
            cv2.imshow("pose_gestures", frame)
            cv2.waitKey(1)
        cv2.destroyAllWindows()


@dataclass
class WristState:
    x_px: float | None
    y_px: float | None
    visibility: float | None


@dataclass
class GestureState:
    cmd: str
    left: WristState
    right: WristState
    ts: float
    frame_h: int
    frame_w: int


class PoseGestureDetector:
    def __init__(
        self,
        cam_index: int = 0,
        fps: float = 15.0,
        midline_ratio: float = 0.5,
        hysteresis_px: float = 20.0,
        min_stable_ms: int = 150,
        visibility_threshold: float = 0.5,
        show_window: bool = True,
        model_path: str = "models/pose_landmarker_lite.task",
    ) -> None:
        self._cam_index = cam_index
        self._fps = fps
        self._midline_ratio = midline_ratio
        self._hysteresis_px = hysteresis_px
        self._min_stable_ms = min_stable_ms
        self._visibility_threshold = visibility_threshold
        self._show_window = show_window
        self._model_path = model_path
        self._preview = CvPreviewer(enabled=show_window)

        self._lock = threading.Lock()
        self._latest: Optional[GestureState] = None
        self._latest_frame = None
        self._latest_midline = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._use_tasks = not hasattr(mp, "solutions")
        if self._use_tasks:
            self._ensure_model(self._model_path)
            options = pose_landmarker_lib.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=self._model_path),
                running_mode=vision.RunningMode.VIDEO,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._pose_task = pose_landmarker_lib.PoseLandmarker.create_from_options(
                options
            )
        else:
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

        self._last_cmd = "stop"
        self._last_change_ts = time.time()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._preview.start()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self._preview.stop()

    def latest(self) -> Optional[GestureState]:
        with self._lock:
            return self._latest

    def render_latest(self) -> None:
        with self._lock:
            if self._latest_frame is None or self._latest is None:
                return
            frame = self._latest_frame.copy()
            state = self._latest
            midline = self._latest_midline
        self._render_overlay(frame, state, midline)

    def _run(self) -> None:
        cap = cv2.VideoCapture(self._cam_index)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self._cam_index}")

        target_dt = 0 if self._fps <= 0 else 1.0 / self._fps
        next_ts = time.time()

        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            frame_h, frame_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self._use_tasks:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(time.time() * 1000)
                result = self._pose_task.detect_for_video(mp_image, timestamp_ms)
                left, right = self._extract_wrists_task(result, frame_w, frame_h)
            else:
                result = self._pose.process(rgb)
                left, right = self._extract_wrists(result, frame_w, frame_h)
            midline = self._midline_ratio * frame_h
            cmd = self._classify(left, right, midline)
            cmd = self._debounce(cmd)

            state = GestureState(
                cmd=cmd,
                left=left,
                right=right,
                ts=time.time(),
                frame_h=frame_h,
                frame_w=frame_w,
            )
            with self._lock:
                self._latest = state

            if self._show_window:
                with self._lock:
                    self._latest_frame = frame
                    self._latest_midline = midline

            if target_dt > 0:
                next_ts += target_dt
                sleep_s = next_ts - time.time()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_ts = time.time()

        cap.release()

    def _extract_wrists(
        self, result, frame_w: int, frame_h: int
    ) -> Tuple[WristState, WristState]:
        if not result.pose_landmarks:
            return WristState(None, None, None), WristState(None, None, None)

        lm = result.pose_landmarks.landmark
        left_lm = lm[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
        right_lm = lm[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]

        left = WristState(
            x_px=left_lm.x * frame_w,
            y_px=left_lm.y * frame_h,
            visibility=left_lm.visibility,
        )
        right = WristState(
            x_px=right_lm.x * frame_w,
            y_px=right_lm.y * frame_h,
            visibility=right_lm.visibility,
        )
        return left, right

    def _extract_wrists_task(
        self, result, frame_w: int, frame_h: int
    ) -> Tuple[WristState, WristState]:
        if not result.pose_landmarks:
            return WristState(None, None, None), WristState(None, None, None)
        pose = result.pose_landmarks[0]
        left_lm = pose[pose_landmarker_lib.PoseLandmark.LEFT_WRIST]
        right_lm = pose[pose_landmarker_lib.PoseLandmark.RIGHT_WRIST]
        left = WristState(
            x_px=left_lm.x * frame_w,
            y_px=left_lm.y * frame_h,
            visibility=left_lm.visibility,
        )
        right = WristState(
            x_px=right_lm.x * frame_w,
            y_px=right_lm.y * frame_h,
            visibility=right_lm.visibility,
        )
        return left, right

    def _ensure_model(self, path: str) -> None:
        import os
        import urllib.request

        if os.path.exists(path):
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(_DEFAULT_MODEL_URL, path)

    def _is_up(self, wrist: WristState, midline: float) -> Optional[bool]:
        if wrist.y_px is None or wrist.visibility is None:
            return None
        if wrist.visibility < self._visibility_threshold:
            return None
        if wrist.y_px < (midline - self._hysteresis_px):
            return True
        if wrist.y_px > (midline + self._hysteresis_px):
            return False
        return None

    def _classify(self, left: WristState, right: WristState, midline: float) -> str:
        left_up = self._is_up(left, midline)
        right_up = self._is_up(right, midline)

        if left_up is True and right_up is True:
            return "forward"
        if left_up is True and right_up is not True:
            return "left"
        if right_up is True and left_up is not True:
            return "right"
        return "stop"

    def _debounce(self, cmd: str) -> str:
        now = time.time()
        if cmd != self._last_cmd:
            self._last_cmd = cmd
            self._last_change_ts = now
        if (now - self._last_change_ts) * 1000.0 < self._min_stable_ms:
            return "stop"
        return self._last_cmd

    def _render_overlay(self, frame, state: GestureState, midline: float) -> None:
        # Mirror preview for more intuitive interaction.
        frame = cv2.flip(frame, 1)
        cv2.line(
            frame,
            (0, int(midline)),
            (state.frame_w, int(midline)),
            (0, 255, 255),
            2,
        )

        def draw_wrist(wrist: WristState, color):
            if wrist.x_px is None or wrist.y_px is None:
                return
            x = int(state.frame_w - wrist.x_px)
            y = int(wrist.y_px)
            cv2.circle(frame, (x, y), 6, color, -1)

        draw_wrist(state.left, (255, 0, 0))
        draw_wrist(state.right, (0, 0, 255))

        cv2.putText(
            frame,
            f"cmd: {state.cmd}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        self._preview.show(frame)

