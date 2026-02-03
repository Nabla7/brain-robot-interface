# Brain-Robot Interface (Hack Nation)

Single `cmd_vel` pipeline with three modes:

- **SimWalk**: MJLab ONNX policy in MuJoCo (walks in sim).
- **RobotWalk**: Unitree built-in locomotion via Sport RPC (real robot).
- **MirrorView**: subscribe to LowState and mirror the robot in MuJoCo.

## Setup

```bash
git clone --recurse-submodules git@github.com:Nabla7/brain-robot-interface.git
cd brain-robot-interface

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you already cloned without submodules:
```bash
git submodule update --init --recursive
```

## Run

### SimWalk (MJLab ONNX in MuJoCo)
```bash
mjpython run_sim.py \
  --bundle bundles/g1_mjlab \
  --source both \
  --log-hz 30 \
  --show-cv
```

### RobotWalk (real Unitree G1, Sport RPC)
```bash
python -m src.apps.run_robot_walk \
  --interface en0 \
  --domain-id 0 \
  --source cv \
  --log-hz 30 \
  --show-cv
```

### MirrorView (robot → MuJoCo)
```bash
mjpython -m src.apps.run_mirror_view \
  --xml bundles/g1_mjlab/scene.xml \
  --interface en0 \
  --domain-id 0
```

## Gesture mapping (MediaPipe Pose)

- Left wrist above midline → **left**
- Right wrist above midline → **right**
- Both wrists above midline → **forward**
- Otherwise → **stop**

Tune thresholds:
```bash
mjpython run_sim.py --midline 0.5 --hysteresis-px 20 --stable-ms 150
```

MediaPipe auto-downloads the pose model to `models/pose_landmarker_lite.task` on first run.

## Keyboard controls (SimWalk only)

- `W` / `Up`: forward
- `A` / `Left`: left
- `D` / `Right`: right
- `Q` / `E`: yaw left/right
- `S` / `Down` / `Space`: stop

Keyboard inputs override CV for `--key-timeout` seconds.

## Troubleshooting

- **macOS viewer**: use `mjpython` for MuJoCo windows.
- **macOS camera permissions**: allow Terminal/Python in System Settings → Privacy → Camera.
- **Black CV window**: try `--cam 1` or unplug/replug the webcam.
- **CV preview**: the preview runs in a separate process to avoid conflicts with MuJoCo on macOS.
- **MediaPipe errors**: ensure the model download succeeded or pass `--pose-model`.

## Output

CSV logs are written to `logs/session_<timestamp>.csv` with:

```
ts,vx,vy,yaw_rate,source,left_x,left_y,right_x,right_y,left_vis,right_vis
```

