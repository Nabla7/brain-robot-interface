from __future__ import annotations

import argparse
import os
from pathlib import Path
import threading
import time

import platform
import signal
import glfw
import mujoco
import mujoco.viewer

from ..command.cmd_vel import CmdVel, CmdVelSmoother
from ..command.command_mux import CommandMux
from ..command.cv_pose_gestures import PoseGestureDetector
from ..command.csv_logger import CsvLogger, LogSample
from ..backends.sim_mjlab_onnx import CmdVelController, SimWalkConfig, build_sim_policy, load_sim_model


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SimWalk: MJLab ONNX in MuJoCo")
    parser.add_argument(
        "--bundle",
        default=str(_repo_root() / "bundles" / "g1_mjlab"),
        help="Path to MJLab bundle (model.xml + policy.onnx + assets/)",
    )
    parser.add_argument("--source", default="both", choices=["keyboard", "cv", "both"])
    parser.add_argument("--log-hz", type=float, default=30.0)
    parser.add_argument("--sim-dt", type=float, default=0.005)
    parser.add_argument("--viewer-dt", type=float, default=0.02)
    parser.add_argument("--ctrl-dt", type=float, default=0.02)
    parser.add_argument("--action-scale", type=float, default=0.5)
    parser.add_argument("--max-vx", type=float, default=0.6)
    parser.add_argument("--max-vy", type=float, default=0.4)
    parser.add_argument("--max-yaw", type=float, default=1.5)
    parser.add_argument("--cmd-smoothing", type=float, default=0.2)
    parser.add_argument("--key-timeout", type=float, default=0.35)
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--cv-fps", type=float, default=15.0)
    parser.add_argument("--midline", type=float, default=0.5)
    parser.add_argument("--hysteresis-px", type=float, default=20.0)
    parser.add_argument("--stable-ms", type=int, default=150)
    parser.add_argument("--show-cv", action="store_true")
    parser.add_argument("--pose-model", default=str(_repo_root() / "models" / "pose_landmarker_lite.task"))
    parser.add_argument("--logs", default=str(_repo_root() / "logs"))
    return parser


def _cmd_from_discrete(name: str, max_vx: float, max_vy: float, max_yaw: float) -> tuple[float, float, float]:
    if name == "forward":
        return max_vx, 0.0, 0.0
    if name == "left":
        return 0.0, 0.0, max_yaw
    if name == "right":
        return 0.0, 0.0, -max_yaw
    if name == "yaw_left":
        return 0.0, 0.0, max_yaw
    if name == "yaw_right":
        return 0.0, 0.0, -max_yaw
    return 0.0, 0.0, 0.0


def main() -> None:
    args = _build_arg_parser().parse_args()

    cfg = SimWalkConfig(ctrl_dt=args.ctrl_dt, sim_dt=args.sim_dt, action_scale=args.action_scale)
    model, data, default_qpos = load_sim_model(args.bundle, cfg)
    controller = CmdVelController()
    policy = build_sim_policy(args.bundle, cfg, controller, default_qpos)

    cmd_mux = CommandMux(key_timeout_s=args.key_timeout)
    smoother = CmdVelSmoother(alpha=args.cmd_smoothing)

    gesture_detector = None
    if args.source in ("cv", "both"):
        gesture_detector = PoseGestureDetector(
            cam_index=args.cam,
            fps=args.cv_fps,
            midline_ratio=args.midline,
            hysteresis_px=args.hysteresis_px,
            min_stable_ms=args.stable_ms,
            show_window=args.show_cv,
            model_path=args.pose_model,
        )
        gesture_detector.start()

    lock = threading.Lock()

    def key_callback(key: int) -> None:
        mapping = {
            glfw.KEY_W: "forward",
            glfw.KEY_UP: "forward",
            glfw.KEY_A: "left",
            glfw.KEY_LEFT: "left",
            glfw.KEY_D: "right",
            glfw.KEY_RIGHT: "right",
            glfw.KEY_Q: "yaw_left",
            glfw.KEY_E: "yaw_right",
            glfw.KEY_S: "stop",
            glfw.KEY_DOWN: "stop",
            glfw.KEY_SPACE: "stop",
        }
        cmd_name = mapping.get(key)
        if cmd_name:
            vx, vy, yaw = _cmd_from_discrete(cmd_name, args.max_vx, args.max_vy, args.max_yaw)
            cmd_mux.update_keyboard(CmdVel(vx, vy, yaw, ts=time.time(), source="keyboard"))

    try:
        viewer = mujoco.viewer.launch_passive(
            model,
            data,
            key_callback=key_callback,
            show_left_ui=False,
            show_right_ui=False,
        )
    except RuntimeError as exc:
        if platform.system() == "Darwin":
            raise RuntimeError(
                "On macOS, MuJoCo viewer must be launched via `mjpython`.\n"
                "Example: mjpython run_sim.py --bundle bundles/g1_mjlab --source both --show-cv"
            ) from exc
        raise

    def sim_loop() -> None:
        while viewer.is_running():
            start = time.perf_counter()
            with lock:
                cmd = smoother.update(cmd_mux.sample())
                controller.set_cmd(cmd)
                policy.step(model, data)
                mujoco.mj_step(model, data)
            dt = model.opt.timestep - (time.perf_counter() - start)
            if dt > 0:
                time.sleep(dt)

    def viewer_loop() -> None:
        while viewer.is_running():
            with lock:
                viewer.sync()
            time.sleep(args.viewer_dt)

    logger = CsvLogger(log_hz=args.log_hz, out_dir=args.logs)

    def log_sample() -> LogSample:
        state = gesture_detector.latest() if gesture_detector else None
        cmd = cmd_mux.sample()
        return LogSample(
            ts=time.time(),
            vx=cmd.vx,
            vy=cmd.vy,
            yaw_rate=cmd.yaw_rate,
            source=cmd.source,
            left_x=None if not state else state.left.x_px,
            left_y=None if not state else state.left.y_px,
            right_x=None if not state else state.right.x_px,
            right_y=None if not state else state.right.y_px,
            left_vis=None if not state else state.left.visibility,
            right_vis=None if not state else state.right.visibility,
        )

    log_path = logger.start(log_sample)
    print(f"[INFO] Logging to {log_path}")

    sim_thread = threading.Thread(target=sim_loop, daemon=True)
    view_thread = threading.Thread(target=viewer_loop, daemon=True)
    sim_thread.start()
    view_thread.start()

    stop_event = threading.Event()

    def _handle_sigint(signum, frame) -> None:  # noqa: ARG001
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        while viewer.is_running() and not stop_event.is_set():
            if gesture_detector:
                latest = gesture_detector.latest()
                if latest:
                    vx, vy, yaw = _cmd_from_discrete(
                        latest.cmd, args.max_vx, args.max_vy, args.max_yaw
                    )
                    cmd_mux.update_cv(CmdVel(vx, vy, yaw, ts=latest.ts, source="cv"))
                if args.show_cv:
                    gesture_detector.render_latest()
            time.sleep(0.01)
    finally:
        try:
            if viewer.is_running():
                viewer.close()
        except Exception:
            pass
        logger.stop()
        if gesture_detector:
            gesture_detector.stop()


if __name__ == "__main__":
    main()

