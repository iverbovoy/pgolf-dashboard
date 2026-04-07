# pgolf dashboard

Real-time training monitor for [Parameter Golf](https://github.com/openai/parameter-golf) runs. Single Python file, no build step, no configuration needed.

![screenshot](screenshot.png)

## Quick Start

```bash
pip install fastapi uvicorn
python3 dashboard.py --ssh "root@host -p 1234" --remote-log LATEST
```

Open `http://localhost:8050`. Replace `root@host -p 1234` with your RunPod TCP SSH connection string.

No changes to your training script needed — the dashboard parses standard `train_gpt.py` output.

## Usage

```bash
# Monitor remote RunPod pod (auto-detects latest .log in /workspace/)
python3 dashboard.py --ssh "root@host -p 1234" --remote-log LATEST

# Monitor specific log file on remote pod
python3 dashboard.py --ssh "root@host -p 1234" --remote-log /workspace/train.log

# View local log file
python3 dashboard.py train.log

# Open empty UI, connect to pod via browser input field
python3 dashboard.py

# Auto-stop pod when training completes (saves log locally first)
python3 dashboard.py --ssh "root@host -p 1234" --remote-log LATEST \
  --auto-stop --save-dir ./logs

# Custom port
python3 dashboard.py --port 3000
```

## What It Shows

### During Training
- Val BPB chart with baseline reference line (1.2244, configurable)
- Step count, current BPB, speed (ms/step)
- Progress bar counting down from max wallclock (10 min default)
- Warmup detection

### During Eval
- Progress bar switches to 10-min eval budget countdown
- Turns red if eval exceeds the time limit
- Stage tracker — each phase appears as it completes:

```
● Training         10m 00s · 4,600 steps
● Eval             4m 15s
  ● SWA              43 ckpts
  ● Roundtrip        1.2168 · 14s
  ● Sliding Window   1.1832 · 76s
  ● Hedge Mixer      1.1324 · 159s
  ● Artifact         15.40 MB
```

Only stages present in the log are shown — works with any eval pipeline.

### Themes
Click the logo to cycle: **mono** (black/white) → **kitty** (pink/black) → **emerald** (green/peach). Each theme has its own font, chart colors, and progress bar style. Preference saved to localStorage.

### LATEST Mode
`--remote-log LATEST` finds the most recently modified `.log` in `/workspace/`. When a new run starts, the dashboard switches to it automatically — no restart needed.

### Auto-stop
With `--auto-stop`, the dashboard saves the log locally and stops the RunPod pod when training completes. Requires [runpodctl](https://github.com/runpod/runpodctl) installed locally. Useful for overnight runs.

## Requirements

- Python 3.10+
- `fastapi`, `uvicorn` (`pip install fastapi uvicorn`)
- Internet for CDN (Plotly.js, Google Fonts) on first page load
- SSH access to remote pod (optional, for remote monitoring)

## Log Compatibility

Parses standard `train_gpt.py` output from the parameter-golf challenge:

```
model_params:23662344
world_size:8 gpu:NVIDIA H100 80GB HBM3
max_wallclock_seconds:600.000
warmup_step:5/20
step:100/20000 train_loss:2.5000 train_time:10000ms step_avg:100.00ms
step:100/20000 val_loss:2.3000 val_bpb:1.3600
swa: averaging 44 checkpoints
Serialized model int8+zstd22: 15403955 bytes
final_roundtrip_exact val_loss:2.0545 val_bpb:1.2168
final_sliding_window_exact val_loss:1.9978 val_bpb:1.1832
final_hedge_mixer_exact val_loss:1.9120 val_bpb:1.1324
```

Unrecognized lines are silently ignored — safe to use with any fork or modified script.

## Tips

- Use the **TCP SSH** connection from RunPod (not the proxy one), e.g. `root@203.0.113.1 -p 12345`
- Baseline value is configurable: edit `const BASELINE=1.2244` at the top of the `<script>` section
- Refresh rate: 10s for short runs (≤10min), 3min for long runs — auto-detected
- If SSH drops, dashboard shows cached data until reconnection

## Architecture

Single `dashboard.py` file (~1200 lines). FastAPI backend, inline HTML/CSS/JS frontend with Plotly.js. No build step, no dependencies beyond FastAPI.
