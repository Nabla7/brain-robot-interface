from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party" / "unitree_sdk2_python"))

import mujoco
import mujoco.viewer

from ..backends.unitree_lowstate_mirror import MirrorConfig, UnitreeLowStateMirror
from ..command.csv_logger import CsvLogger, LogSample


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MirrorView: robot LowState -> MuJoCo")
    parser.add_argument(
        "--xml",
        default="bundles/g1_mjlab/scene.xml",
        help="MuJoCo scene XML for visualization",
    )
    parser.add_argument("--interface", default="en0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--viewer-dt", type=float, default=0.02)
    parser.add_argument("--log-hz", type=float, default=10.0)
    parser.add_argument("--logs", default="logs")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    mirror = UnitreeLowStateMirror(
        model, data, MirrorConfig(domain_id=args.domain_id, interface=args.interface)
    )

    viewer = mujoco.viewer.launch_passive(
        model,
        data,
        show_left_ui=False,
        show_right_ui=False,
    )
    logger = CsvLogger(log_hz=args.log_hz, out_dir=args.logs)
    log_path = logger.start(
        lambda: LogSample(
            ts=time.time(),
            vx=0.0,
            vy=0.0,
            yaw_rate=0.0,
            source="mirror",
            left_x=None,
            left_y=None,
            right_x=None,
            right_y=None,
            left_vis=None,
            right_vis=None,
        )
    )
    print(f"[INFO] Logging to {log_path}")

    try:
        while viewer.is_running():
            updated = mirror.apply_to_mujoco()
            if updated:
                mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(args.viewer_dt)
    finally:
        logger.stop()


if __name__ == "__main__":
    main()

