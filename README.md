# Brain Robot Interface

Minimal API for controlling Unitree robots in sim or on hardware using discrete actions.

The only intended entrypoint is `examples/minimal_policy.py`.

## Setup

```bash
cd ~
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install
```

```bash
git clone https://github.com/Nabla7/brain-robot-interface.git
cd brain-robot-interface
uv sync
source .venv/bin/activate
```

## Run

### Sim (macOS)

```bash
CYCLONEDDS_HOME="$HOME/cyclonedds/install" uv run mjpython examples/minimal_policy.py
```

### Sim (Linux)

```bash
CYCLONEDDS_HOME="$HOME/cyclonedds/install" uv run python examples/minimal_policy.py
```

### Robot + Mirror

Set `backend="robot"` in the example script and run it normally:

```bash
CYCLONEDDS_HOME="$HOME/cyclonedds/install" uv run python examples/minimal_policy.py
```

## API

```python
from bri import Action, Controller

ctrl = Controller(
    backend="sim",
    hold_s=0.3,
    forward_speed=0.6,
    yaw_rate=1.5,
    smooth_alpha=0.2,
)
ctrl.start()
ctrl.set_action(Action.FORWARD)
```

## Parameters

- `backend`: `"sim"` or `"robot"`
- `hold_s`: seconds to keep last action before auto-STOP
- `forward_speed`: forward velocity
- `yaw_rate`: rotation speed
- `smooth_alpha`: smoothing factor (0–1)
- `ctrl_hz`: internal control loop rate
- `interface`, `domain_id`: DDS settings for real robot
- `mirror`: when `backend="robot"`, mirror view is enabled by default

## Notes

- On macOS, use `mjpython` for any MuJoCo window.
- Robot mode requires DDS deps (e.g., `cyclonedds`) from Unitree SDK.
