#!/usr/bin/env python3
"""Parameter Golf — Universal Training Dashboard.

Monitor any parameter-golf training run. Supports local logs, remote SSH,
drag & drop, and multi-run comparison.

Usage:
    python3 dashboard.py train.log                        # local log file
    python3 dashboard.py train.log --compare baseline.log # compare two runs
    python3 dashboard.py --ssh "root@host -p 1234" --remote-log /workspace/train.log
    python3 dashboard.py                                  # drag & drop in browser

Options:
    --port 8050           Server port (default: 8050)
    --refresh 180         Auto-refresh interval in seconds (default: 180)
    --ssh "user@host"     SSH connection string for remote log
    --remote-log PATH     Path to log on remote host
    --compare FILE        Load a second log for comparison overlay
"""

import argparse
import json
import re
import secrets
import shlex
import subprocess
import time
from pathlib import Path

__version__ = "0.2.0"
_start_ts = time.time()

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
_runs: dict[str, dict] = {}  # name -> parsed data
_ssh_cache = {"text": "", "ts": 0}
_auth_token: str = secrets.token_urlsafe(24)
_ssh_health = {"last_ok_ts": 0, "consecutive_failures": 0, "last_error": "", "status": "unknown"}
_ssh_state = {"byte_offset": 0, "local_cache": ""}
_webhook_sent: dict[str, set] = {}  # run_key -> set of events already sent
_config = {
    "ssh_args": None,   # list[str] — parsed SSH arguments (no shell interpolation)
    "ssh_host": None,    # str — extracted host for pod matching
    "remote_log": None,
    "refresh": 180,
    "local_files": [],
    "compare_files": [],
    "auto_stop": False,
    "auto_stop_done": False,
    "save_dir": ".",
    "baseline": None,
    "webhook_url": None,
    "webhook_threshold": None,
    "ssh_targets": [],   # list of dicts for multi-SSH
}


# ---------------------------------------------------------------------------
# Log parser (works with any train_gpt.py log)
# ---------------------------------------------------------------------------
def parse_log(text: str) -> dict:
    d = {"name": "", "steps": [], "train_loss": [], "step_avg": [], "train_time": [],
         "val_steps": [], "val_bpb": [],
         "pert_steps": [], "pert_chi": [],
         "chi_steps": [], "chi_vals": [],
         "phase_switches": [], "finals": {},
         "config": {}, "status": "warmup",
         "warmup_cur": 0, "warmup_total": 0,
         "swa_count": 0, "peak_mem": ""}
    if not text:
        return d
    for line in text.splitlines():
        # Config
        # Parse key:value pairs from config lines
        m = re.search(r"gpu:([\w.\-:/()+ ]+?)(?:\s{2,}|$)", line)
        if m:
            d["config"]["gpu"] = re.sub(r"[^\w.\-:/()+ ]", "", m.group(1).strip())[:80]
        for key in ["world_size", "model_params", "seed", "num_repeats", "grad_accum_steps"]:
            m = re.search(rf"\b{key}:(\d+)", line)
            if m:
                d["config"][key] = m.group(1)
        m = re.search(r"max_wallclock_seconds:([\d.]+)", line)
        if m:
            d["config"]["max_wallclock"] = float(m.group(1))
        m = re.search(r"iterations:(\d+)", line)
        if m:
            d["config"]["iterations"] = int(m.group(1))
        m = re.search(r"train_batch_tokens:(\d+)", line)
        if m:
            d["config"]["train_batch_tokens"] = m.group(1)

        # Warmup steps
        m = re.search(r"warmup_step:(\d+)/(\d+)", line)
        if m:
            d["warmup_cur"] = int(m.group(1))
            d["warmup_total"] = int(m.group(2))
            d["status"] = "warmup"
            continue

        # Perturbation sensitivity (EoC diagnostic — optional)
        m = re.match(r"step:(\d+) perturbation_sensitivity pert_div:\[(.+?)\] pert_chi:\[(.+?)\]", line)
        if m:
            d["pert_steps"].append(int(m.group(1)))
            d["pert_chi"].append([float(x) for x in m.group(3).split(",")])
            continue

        # Phase switch
        m = re.search(r"prog_depth: switched to (\d+) repeats at step:(\d+)", line)
        if m:
            d["phase_switches"].append([int(m.group(2)), int(m.group(1))])
            continue

        # Prog depth schedule (extract from log start)
        m = re.search(r"prog_depth: schedule=\[(.+?)\] starting_repeats=(\d+)", line)
        if m:
            d["config"]["prog_schedule"] = m.group(1)
            d["config"]["num_repeats"] = m.group(2)
            continue

        # Val bpb
        m = re.search(r"step:(\d+)/\d+ val_loss:([\d.]+) val_bpb:([\d.]+)", line)
        if m:
            s, b = int(m.group(1)), float(m.group(3))
            if s > 0:
                d["val_steps"].append(s)
                d["val_bpb"].append(b)
                d["status"] = "training"
            continue

        # Train step
        m = re.search(r"step:(\d+)/\d+ train_loss:([\d.]+) train_time:(\d+)ms step_avg:([\d.]+)ms(?: chi:([\d.]+))?", line)
        if m:
            d["status"] = "training"
            d["steps"].append(int(m.group(1)))
            d["train_loss"].append(float(m.group(2)))
            d["train_time"].append(int(m.group(3)))
            d["step_avg"].append(float(m.group(4)))
            if m.group(5):
                d["chi_steps"].append(int(m.group(1)))
                d["chi_vals"].append(float(m.group(5)))

        # Finals
        for key, pat in [
            ("roundtrip", r"final_roundtrip_exact val_loss:([\d.]+) val_bpb:([\d.]+)"),
            ("sliding", r"final_sliding_window_exact val_loss:([\d.]+) val_bpb:([\d.]+)"),
            ("hedge", r"final_hedge_mixer_exact val_loss:([\d.]+) val_bpb:([\d.]+)"),
            ("ttt", r"final_ttt_exact val_loss:([\d.]+) val_bpb:([\d.]+)"),
            ("roundtrip", r"final_int8_zlib_roundtrip_exact val_loss:([\d.]+) val_bpb:([\d.]+)"),
        ]:
            m2 = re.search(pat, line)
            if m2:
                d["finals"][key] = float(m2.group(2))
        # Eval times
        for key, pat in [
            ("rt_time", r"final_roundtrip .*eval_time:(\d+)ms"),
            ("sw_time", r"final_sliding_window .*eval_time:(\d+)ms"),
            ("hm_time", r"final_hedge_mixer .*eval_time:(\d+)ms"),
            ("ttt_time", r"final_ttt .*eval_time:(\d+)ms"),
        ]:
            m2 = re.search(pat, line)
            if m2:
                d["finals"][key] = int(m2.group(1))
        # Status tracking
        if "swa: averaging" in line:
            d["status"] = "swa"
        m = re.search(r"swa: averaging (\d+) checkpoints", line)
        if m:
            d["swa_count"] = int(m.group(1))
        if "stopping_early" in line or "step:" in line and f"/{d['config'].get('iterations', 0)}" in line:
            pass  # status updated below
        if "peak memory" in line:
            d["peak_mem"] = line.strip()
        if "Serialized model int" in line:
            m = re.search(r"Serialized model .+?: (\d+) bytes", line)
            if m:
                d["finals"]["artifact_bytes"] = int(m.group(1))
        if "final_roundtrip " in line and "exact" not in line:
            d["status"] = "eval_sliding"
        if "final_roundtrip_exact" in line or "final_int8_zlib_roundtrip_exact" in line:
            d["status"] = "done"  # default done after roundtrip
        if "final_sliding_window_exact" in line:
            d["status"] = "done"  # done after sliding if no hedge
        if "final_hedge_mixer " in line and "exact" not in line:
            d["status"] = "eval_hedge"
        if "final_hedge_mixer_exact" in line:
            d["status"] = "done"
        if "final_ttt " in line and "exact" not in line:
            d["status"] = "eval_ttt"
        if "final_ttt_exact" in line:
            d["status"] = "done"

    # Crash/stall detection: check last 30 lines for error patterns
    lines = text.splitlines()
    crash_patterns = re.compile(r"(Traceback|CUDA out of memory|RuntimeError|torch\.cuda\.OutOfMemoryError|Killed|oom-kill)")
    tail = lines[-30:] if len(lines) >= 30 else lines
    for tline in reversed(tail):
        cm = crash_patterns.search(tline)
        if cm:
            d["status"] = "crashed"
            d["error_excerpt"] = tline.strip()[:200]
            break

    return d


def _parse_ssh_string(ssh_str: str, insecure: bool = False) -> list[str]:
    """Parse an SSH connection string into a safe argv list."""
    parts = shlex.split(ssh_str)
    host_key_policy = "no" if insecure else "accept-new"
    ssh_opts = [
        "-o", f"StrictHostKeyChecking={host_key_policy}",
        "-o", "UserKnownHostsFile=~/.ssh/pgolf_known_hosts",
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
    ]
    return ["ssh"] + parts + ssh_opts


def _extract_ssh_host(ssh_args: list[str]) -> str | None:
    """Extract the target host from parsed SSH args for pod matching."""
    skip_next = False
    for arg in ssh_args[1:]:  # skip "ssh"
        if skip_next:
            skip_next = False
            continue
        if arg in ("-p", "-i", "-o", "-l", "-F", "-J"):
            skip_next = True
            continue
        if arg.startswith("-"):
            continue
        # First non-flag arg is [user@]host
        return arg.split("@")[-1] if "@" in arg else arg
    return None


_SAFE_LOG_PATH = re.compile(r"^/[\w./_-]+\.log$")


def _fetch_ssh_target(target: dict) -> str:
    """Fetch log from a single SSH target with incremental tailing and health tracking."""
    ssh_args = target["ssh_args"]
    remote_log = target["remote_log"]
    health = target["health"]
    if not ssh_args or not remote_log:
        return ""
    now = time.time()
    try:
        remote = remote_log
        if remote == "LATEST":
            lr = subprocess.run(
                ssh_args + ["ls -t /workspace/*.log 2>/dev/null | head -1"],
                capture_output=True, text=True, timeout=10)
            new_remote = lr.stdout.strip() or "/workspace/train.log"
            if not _SAFE_LOG_PATH.match(new_remote):
                print(f"[warn] Ignoring suspicious log filename: {new_remote!r}")
                return ""
            if new_remote != target.get("_resolved_log"):
                target["local_cache"] = ""
                target["byte_offset"] = 0
                _config["auto_stop_done"] = False  # re-arm for new run
            remote = new_remote
            target["_resolved_log"] = remote
        # Incremental fetch: get remote file size first
        sr = subprocess.run(
            ssh_args + [f"stat -c %s {remote}"],
            capture_output=True, text=True, timeout=10)
        if sr.returncode == 0 and sr.stdout.strip().isdigit():
            remote_size = int(sr.stdout.strip())
            cur_offset = target["byte_offset"]
            if remote_size < cur_offset:
                # File rotated/truncated — reset
                target["byte_offset"] = 0
                target["local_cache"] = ""
                cur_offset = 0
            if remote_size > cur_offset:
                # Fetch only new bytes
                r = subprocess.run(
                    ssh_args + [f"tail -c +{cur_offset + 1} {remote}"],
                    capture_output=True, text=True, timeout=20)
                if r.returncode == 0:
                    target["local_cache"] += r.stdout
                    target["byte_offset"] = remote_size
            # Success
            health["last_ok_ts"] = now
            health["consecutive_failures"] = 0
            health["last_error"] = ""
            health["status"] = "ok"
            result = target["local_cache"]
            if result:
                Path("/tmp/pgolf_ssh_cache.log").write_text(result)
            return result
        else:
            # stat failed — fallback to full cat
            r = subprocess.run(
                ssh_args + [f"cat {remote}"],
                capture_output=True, text=True, timeout=20)
            if r.returncode == 0 and r.stdout.strip():
                target["local_cache"] = r.stdout
                target["byte_offset"] = len(r.stdout.encode())
                health["last_ok_ts"] = now
                health["consecutive_failures"] = 0
                health["last_error"] = ""
                health["status"] = "ok"
                Path("/tmp/pgolf_ssh_cache.log").write_text(r.stdout)
                return r.stdout
            raise RuntimeError(f"cat failed: rc={r.returncode}")
    except Exception as e:
        health["consecutive_failures"] += 1
        health["last_error"] = str(e)[:200]
        cf = health["consecutive_failures"]
        if cf >= 5:
            health["status"] = "failing"
        elif cf >= 2:
            health["status"] = "stale"
    # Fallback: local cache file
    cache = Path("/tmp/pgolf_ssh_cache.log")
    if cache.exists():
        t = cache.read_text()
        target["local_cache"] = t
        return t
    return target.get("local_cache", "")


def fetch_ssh_log() -> str:
    """Legacy single-SSH fetch — delegates to first SSH target."""
    if not _config.get("ssh_targets"):
        return ""
    now = time.time()
    if now - _ssh_cache["ts"] < 5 and _ssh_cache["text"]:
        return _ssh_cache["text"]
    target = _config["ssh_targets"][0]
    text = _fetch_ssh_target(target)
    # Sync global health from primary target
    _ssh_health.update(target["health"])
    if text:
        _ssh_cache.update(text=text, ts=now)
    return text


def _send_webhook(run_data: dict, event: str):
    """Send notification to Discord/Slack webhook."""
    if not _config.get("webhook_url"):
        return
    import urllib.request
    finals = run_data.get("finals", {})
    best_bpb = finals.get("hedge") or finals.get("sliding") or finals.get("roundtrip") or "N/A"
    artifact_mb = f'{finals["artifact_bytes"]/1e6:.2f} MB' if "artifact_bytes" in finals else "N/A"
    payload = {
        "content": f"**{event}** | {run_data.get('name', 'unknown')} | bpb: {best_bpb} | artifact: {artifact_mb}",
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(_config["webhook_url"], data=data, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
        print(f"[webhook] Sent: {event}")
    except Exception as e:
        print(f"[webhook] Failed: {e}")


def _check_webhook_events(run_key: str, data: dict):
    """Check and fire webhook events for a run, avoiding duplicates."""
    if not _config.get("webhook_url"):
        return
    if run_key not in _webhook_sent:
        _webhook_sent[run_key] = set()
    sent = _webhook_sent[run_key]
    # Training complete
    if data.get("status") == "done" and "done" not in sent:
        sent.add("done")
        _send_webhook(data, "Training complete")
    # BPB threshold crossing
    threshold = _config.get("webhook_threshold")
    if threshold is not None and data.get("val_bpb"):
        last_bpb = data["val_bpb"][-1]
        if last_bpb < threshold and "threshold" not in sent:
            sent.add("threshold")
            _send_webhook(data, f"BPB crossed {threshold}")


def load_all_runs():
    """Reload all data sources."""
    # Local files
    for i, f in enumerate(_config["local_files"]):
        p = Path(f)
        if p.exists():
            name = p.stem
            data = parse_log(p.read_text())
            data["name"] = name
            data["source"] = "local"
            _runs[f"main_{i}"] = data

    # Compare files
    for i, f in enumerate(_config["compare_files"]):
        p = Path(f)
        if p.exists():
            name = p.stem + " (compare)"
            data = parse_log(p.read_text())
            data["name"] = name
            data["source"] = "compare"
            _runs[f"compare_{i}"] = data

    # Multi-SSH: loop over all targets
    for idx, target in enumerate(_config.get("ssh_targets", [])):
        now = time.time()
        text = _fetch_ssh_target(target)
        # Sync global health from first target
        if idx == 0:
            _ssh_health.update(target["health"])
            _config["_resolved_log"] = target.get("_resolved_log", "")
            if text:
                _ssh_cache.update(text=text, ts=now)
        if text:
            run_key = f"ssh_{idx}" if len(_config["ssh_targets"]) > 1 else "ssh"
            data = parse_log(text)
            label = target.get("label", f"pod-{idx}")
            data["name"] = f"live ({label})"
            data["source"] = "ssh"
            _runs[run_key] = data
            _check_webhook_events(run_key, data)
            # Auto-stop: save log + stop pod when done (only for first target)
            if idx == 0 and _config["auto_stop"] and not _config["auto_stop_done"] and data["status"] == "done":
                _auto_stop(text, data)


def _auto_stop(log_text: str, parsed: dict):
    """Save log locally and stop the RunPod pod matching our SSH target."""
    import threading
    _config["auto_stop_done"] = True

    # Verify all expected eval stages completed (not just a substring match)
    finals = parsed.get("finals", {})
    if "roundtrip" not in finals:
        print("[auto-stop] Roundtrip eval not found — refusing to stop pod.")
        _config["auto_stop_done"] = False
        return

    def do_stop():
        # Save log
        resolved = _config.get("_resolved_log", "train.log")
        log_name = Path(resolved).name
        save_path = Path(_config["save_dir"]) / log_name
        save_path.write_text(log_text)
        print(f"[auto-stop] Log saved to {save_path} ({len(log_text)} bytes)")

        # Wait a bit for final dashboard refresh
        time.sleep(30)

        # Match pod by SSH target host
        target_host = _config.get("ssh_host")
        if not target_host:
            print("[auto-stop] No SSH host configured — cannot identify pod. Refusing to stop.")
            return

        try:
            r = subprocess.run(
                ["runpodctl", "get", "pod"],
                capture_output=True, text=True, timeout=10)
            if r.returncode != 0:
                print(f"[auto-stop] runpodctl failed: {r.stderr}")
                return
            # Find pod matching our SSH target IP
            matched_pods = []
            for line in r.stdout.strip().split("\n")[1:]:
                parts = line.split()
                if len(parts) >= 2 and parts[1] == "RUNNING":
                    pod_id = parts[0]
                    # Check if any column contains our target host
                    if any(target_host in p for p in parts):
                        matched_pods.append(pod_id)
            if not matched_pods:
                print(f"[auto-stop] No RUNNING pod matches SSH host {target_host}. Refusing to stop any pod.")
                return
            if len(matched_pods) > 1:
                print(f"[auto-stop] Multiple pods match {target_host}: {matched_pods}. Refusing to guess.")
                return
            pod_id = matched_pods[0]
            print(f"[auto-stop] Stopping pod {pod_id} (matched {target_host})...")
            subprocess.run(["runpodctl", "stop", "pod", pod_id], timeout=10)
            print(f"[auto-stop] Pod {pod_id} stopped.")
        except Exception as e:
            print(f"[auto-stop] Error: {e}")

    threading.Thread(target=do_stop, daemon=True).start()


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
app = FastAPI()


def _check_token(token: str | None):
    """Verify auth token. Raises 403 if invalid."""
    from fastapi import HTTPException
    if token != _auth_token:
        raise HTTPException(status_code=403, detail="Invalid or missing token")


@app.get("/healthz")
def healthz():
    return JSONResponse({
        "version": __version__,
        "uptime_seconds": int(time.time() - _start_ts),
        "ssh_health": _ssh_health,
        "runs_loaded": len(_runs),
        "ssh_configured": bool(_config.get("ssh_args") or _config.get("ssh_targets")),
        "auto_stop_enabled": _config.get("auto_stop", False),
        "auto_stop_triggered": _config.get("auto_stop_done", False),
    })


@app.get("/api/data")
def api_data(token: str = None):
    _check_token(token)
    load_all_runs()
    return JSONResponse({
        "runs": _runs,
        "config": {"refresh": _config["refresh"], "log": _config.get("_resolved_log", "")},
        "ssh_health": _ssh_health,
    })


@app.post("/api/upload")
async def upload_log(file: UploadFile = File(...), token: str = None):
    _check_token(token)
    text = (await file.read()).decode("utf-8", errors="replace")
    data = parse_log(text)
    raw_name = Path(file.filename or "uploaded").name
    name = re.sub(r"[^\w.\-]", "_", raw_name)[:128]
    data["name"] = name
    data["source"] = "upload"
    key = f"upload_{name}_{int(time.time())}"
    _runs[key] = data
    return JSONResponse({"status": "ok", "key": key, "name": name,
                         "steps": len(data["steps"]), "val_evals": len(data["val_steps"])})


@app.get("/", response_class=HTMLResponse)
def index():
    baseline_val = str(_config["baseline"]) if _config["baseline"] is not None else "null"
    return HTML.replace("__PGOLF_TOKEN__", _auth_token).replace("__PGOLF_BASELINE__", baseline_val)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>pgolf dash</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>⛳</text></svg>">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Switzer:wght@400;500;600;700&family=Nunito+Sans:wght@400;500;600;700&family=Urbanist:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
:root{--font-ui:'Switzer'}
:root,
[data-theme="dark"]{
  --bg:#09090b;--surface:#18181b;--surface2:#27272a;
  --border:#27272a;--border-hover:#3f3f46;
  --text:#fafafa;--text2:#a1a1aa;--text3:#71717a;
  --blue:#60a5fa;--red:#f87171;--green:#4ade80;--amber:#fbbf24;--purple:#c084fc;
  --plot-bg:#18181b;--plot-grid:#27272a;--plot-zero:#3f3f46;--plot-text:#a1a1aa;
  --bar-color:rgba(161,161,170,.5);--bar-glow:rgba(0,0,0,0);--bar-glow2:rgba(0,0,0,0);
}
[data-theme="kitty"]{
  --bg:#0d0409;--surface:#1a0a14;--surface2:#2a1020;
  --border:#3d1830;--border-hover:#522040;
  --text:#f9a8d4;--text2:#d4749e;--text3:#8c4a6a;
  --blue:#f9a8d4;--red:#fb7185;--green:#f9a8d4;--amber:#fda4af;--purple:#e9d5ff;
  --plot-bg:#0d0409;--plot-grid:#2a1020;--plot-zero:#3d1830;--plot-text:#d4749e;
  --font-ui:'Nunito Sans';
  --bar-color:rgba(249,168,212,.5);--bar-glow:rgba(0,0,0,0);--bar-glow2:rgba(0,0,0,0);
}
[data-theme="emerald"]{
  --bg:#061210;--surface:#0c201a;--surface2:#143028;
  --border:#1e4438;--border-hover:#28584a;
  --text:#4ade80;--text2:#f0a878;--text3:#3a7a5a;
  --blue:#4ade80;--red:#f0a878;--green:#4ade80;--amber:#f0a878;--purple:#86efac;
  --font-ui:'Urbanist';
  --plot-bg:#061210;--plot-grid:#1e4438;--plot-zero:#28584a;--plot-text:#f0a878;
  --font-ui:'Urbanist';
  --bar-color:rgba(240,168,120,.5);--bar-glow:rgba(0,0,0,0);--bar-glow2:rgba(0,0,0,0);
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:var(--font-ui),system-ui,-apple-system,sans-serif;
  font-size:13px;-webkit-font-smoothing:antialiased}

/* Header */
header{display:flex;justify-content:space-between;align-items:center;padding:16px 24px 20px}
.hdr-left{display:flex;gap:8px;align-items:center;flex:1}
header h1{font-size:15px;font-weight:600;color:var(--text2);letter-spacing:-.01em;cursor:pointer;user-select:none;transition:color .2s;flex:1;text-align:right}
header h1:hover{color:var(--text)}
header h1 span{color:var(--text);transition:color .2s}
#status{color:var(--text3);font-size:11px;font-family:var(--font-ui),system-ui,sans-serif}

/* Toolbar (in header) */
#upload-btn{background:transparent;border:none;padding:5px 8px;
  color:var(--text3);font-size:13px;cursor:pointer;font-family:var(--font-ui),sans-serif;font-weight:500;transition:color .15s;display:inline-block}
#upload-btn:hover{color:var(--text)}
.run-list{display:flex;gap:6px;flex-wrap:wrap}
.run-tag{background:var(--surface);border:1px solid var(--border);border-radius:20px;padding:5px 14px;
  font-size:11px;font-weight:500;cursor:default;transition:all .15s}
.run-tag .x{margin-left:8px;color:var(--text3);cursor:pointer;font-size:13px}
.run-tag .x:hover{color:var(--red)}
#ssh-health{display:inline-flex;align-items:center;gap:4px;font-size:10px;padding:3px 10px;border-radius:12px;background:var(--surface);border:1px solid var(--border);color:var(--text2);margin-left:8px}
#ssh-health .sh-dot{font-size:8px}
.progress-fill.crashed{background:var(--red) !important}
.error-excerpt{font-size:11px;color:var(--red);padding:4px 24px 12px;font-family:'JetBrains Mono',monospace;word-break:break-all}

/* Progress */
.progress{padding:0 24px 20px}
.progress-wrap{background:var(--surface);border-radius:10px;padding:12px 16px;display:flex;align-items:center;gap:16px}
.progress-bar{flex:1;height:6px;background:var(--surface2);border-radius:3px;overflow:hidden;position:relative}
.progress-fill{height:100%;border-radius:3px;background:var(--bar-color);box-shadow:0 0 8px var(--bar-glow),0 0 20px var(--bar-glow2);transition:width .8s ease}
.progress-text{display:flex;gap:20px;font-size:11px;font-family:var(--font-ui),system-ui,sans-serif;color:var(--text2);white-space:nowrap}
.progress-text .dim{color:var(--text3)}

/* Metrics */
.metrics{display:flex;padding:0 24px 18px;align-items:center;justify-content:space-between}
.metric{display:flex;align-items:baseline;gap:6px;font-family:var(--font-ui),system-ui,sans-serif}
.metric .mlabel{font-size:10px;color:var(--text3);text-transform:uppercase;letter-spacing:.05em;font-weight:500}
.metric .mval{font-size:15px;font-weight:700;letter-spacing:-.02em;white-space:nowrap}
.metric .munit{font-size:10px;color:var(--text3)}
.metric .msub{font-size:10px;color:var(--text3);margin-left:2px}
.msep{width:1px;height:16px;background:var(--border);flex-shrink:0}
.good{color:var(--green)}.bad{color:var(--red)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.pulsing{animation:pulse 2s ease-in-out infinite}

/* Stages */
.stages{display:flex;flex-direction:column;gap:6px;padding:4px 24px 20px}
.stage{display:flex;align-items:center;gap:10px;font-size:15px;color:var(--text3);font-family:var(--font-ui),system-ui,sans-serif}
.stage.active{color:var(--text)}
.stage.done{color:var(--text2)}
.stage .stg-icon{font-size:12px;width:16px;text-align:center;flex-shrink:0}
.stage .stg-name{width:130px;flex-shrink:0;font-weight:500}
.stage .stg-time{font-size:13px;color:var(--text3)}
.stage.sub{padding-left:22px;font-size:13px}

/* Below chart */
.bc-results{display:flex;gap:24px;flex-wrap:wrap}
.bc-item{font-family:var(--font-ui),system-ui,sans-serif}
.bc-item>div:first-child{display:flex;align-items:baseline;gap:8px}
.bc-label{font-size:10px;color:var(--text3);letter-spacing:.03em}
.bc-val{font-size:14px;font-weight:700;color:var(--green)}
.bc-desc{font-size:10px;color:var(--text3);font-family:var(--font-ui),sans-serif;margin-top:2px;opacity:.7}

/* Main chart */
.main-charts{padding:16px 16px 24px}
.mc{overflow:hidden}
.mc .mplot{width:100%;height:480px}
/* Loading overlay */
@keyframes pgolf-spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
@keyframes pgolf-glow{0%,100%{text-shadow:none;opacity:0.5}50%{text-shadow:0 0 6px currentColor;opacity:1}}
@keyframes pgolf-fadein{from{opacity:0}to{opacity:1}}
#loading-overlay{position:fixed;inset:0;z-index:99999;background:var(--bg);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:20px;transition:opacity 0.8s ease}
#loading-overlay.hide{opacity:0;pointer-events:none}
#loading-spinner{width:48px;height:48px;border:2px solid var(--surface2);border-top-color:var(--bar-color,var(--text));border-radius:50%;animation:pgolf-spin 1s linear infinite}
#loading-text{font-family:var(--font-ui),system-ui,sans-serif;font-size:13px;font-weight:500;color:var(--text3);height:18px;animation:pgolf-fadein 0.5s ease}
</style></head><body>

<div id="loading-overlay">
  <div id="loading-logo" style="position:absolute;top:20px;right:24px;font-size:15px;font-weight:600;letter-spacing:-.01em;line-height:1;font-family:var(--font-ui),system-ui,sans-serif;cursor:default"></div>
  <div id="loading-spinner"></div>
  <div id="loading-text"></div>
</div>
<header>
  <div class="hdr-left">
    <div class="run-list" id="run-list"></div>
    <label id="upload-btn" title="Upload .log file">
      <input type="file" id="file-input" accept=".log,.txt" multiple style="display:none">
      <span style="cursor:pointer">+</span>
    </label>
  </div>
  <span id="status"><span id="s-time" style="display:none"></span></span>
  <h1 id="logo-h1"><span id="logo-text">pgolf dashboard</span></h1>
</header>

<div class="metrics" id="metrics-bar">
  <div class="metric"><span class="mlabel">step</span><span class="mval" id="m-step">—</span><span class="munit" id="m-step-u"></span></div>
  <div class="metric"><span class="mlabel">bpb</span><span class="mval" id="m-bpb" style="color:var(--text)">—</span><span class="msub" id="m-bpb-s"></span></div>
  <div class="metric"><span class="mlabel">speed</span><span class="mval" id="m-spd" style="color:var(--text)">—</span><span class="munit">ms</span><span class="msub" id="m-spd-s"></span></div>
</div>

<div class="main-charts">
  <div class="mc"><div id="c-bpb" class="mplot"></div></div>
</div>

<div class="progress">
  <div class="progress-wrap">
    <div class="progress-bar"><div class="progress-fill" id="prog-fill" style="width:0%"></div></div>
    <div class="progress-text">
      <span id="prog-elapsed">—</span>
      <span class="dim" id="prog-remaining">—</span>
      <span id="prog-pct">—</span>
    </div>
  </div>
</div>

<div class="stages" id="stages">
  <div class="stage" id="stg-train"><span class="stg-icon">○</span><span class="stg-name">Training</span><span class="stg-time" id="stg-train-t"></span></div>
  <div class="stage" id="stg-eval"><span class="stg-icon">○</span><span class="stg-name">Eval</span><span class="stg-time" id="stg-eval-t"></span></div>
  <div class="stage sub" id="stg-swa"><span class="stg-icon">○</span><span class="stg-name">SWA</span><span class="stg-time" id="stg-swa-t"></span></div>
  <div class="stage sub" id="stg-rt"><span class="stg-icon">○</span><span class="stg-name">Roundtrip</span><span class="stg-time" id="stg-rt-t"></span></div>
  <div class="stage sub" id="stg-sw"><span class="stg-icon">○</span><span class="stg-name">Sliding Window</span><span class="stg-time" id="stg-sw-t"></span></div>
  <div class="stage sub" id="stg-hm"><span class="stg-icon">○</span><span class="stg-name">Hedge Mixer</span><span class="stg-time" id="stg-hm-t"></span></div>
  <div class="stage sub" id="stg-ttt"><span class="stg-icon">○</span><span class="stg-name">TTT</span><span class="stg-time" id="stg-ttt-t"></span></div>
  <div class="stage sub" id="stg-size" style="display:none"><span class="stg-icon">●</span><span class="stg-name">Artifact</span><span class="stg-time" id="stg-size-t"></span></div>
</div>



<script>
const TOKEN='__PGOLF_TOKEN__';
const BASELINE_CLI='__PGOLF_BASELINE__';
let BASELINE=BASELINE_CLI!=='null'?parseFloat(BASELINE_CLI):null;
let _baselineAutoSet=false;

function getPlotTheme(){
  const s=getComputedStyle(document.documentElement);
  const g=v=>s.getPropertyValue(v).trim();
  return {paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
    font:{color:g('--plot-text')||g('--text2'),size:11,family:'Inter,system-ui,sans-serif'},
    margin:{l:50,r:24,t:28,b:36},legend:{bgcolor:'rgba(0,0,0,0)',font:{size:10,family:'Inter'},x:1,y:0.99,xanchor:'right'},
    xaxis:{showgrid:false,zerolinecolor:'rgba(0,0,0,0)',fixedrange:false,tickfont:{family:'JetBrains Mono',size:10}},
    yaxis:{showgrid:false,zerolinecolor:'rgba(0,0,0,0)',fixedrange:false,tickfont:{family:'JetBrains Mono',size:10}}};
}
let DL=getPlotTheme();
const THEME_LINE={dark:'#e4e4e7',kitty:'#f9a8d4',emerald:'#f0a878'};
function getLineColor(){const t=document.documentElement.getAttribute('data-theme')||'dark';return THEME_LINE[t]||'#e4e4e7'}
const DASH_STYLES=['solid','dash','dot','dashdot','longdash','longdashdot'];
let use24h=localStorage.getItem('pgolf-24h')!=='false';
const RATE_OPTIONS=[5000,10000,30000,60000];
let rateIdx=parseInt(localStorage.getItem('pgolf-rate-idx'))||0;
let rateManual=!!localStorage.getItem('pgolf-rate-idx');
let refreshInterval=rateManual?RATE_OPTIONS[rateIdx]:180000;
let refreshTimer=null;
let lastElapsed=0, lastMaxWall=600000, lastFetchTime=0, clockTimer=null, lastStatus='warmup', lastStep=0, lastCurPh=1, lastEvalTotal=0;

function phS(pp){return pp.filter(p=>p[0]>0).map(p=>({type:'line',x0:p[0],x1:p[0],y0:0,y1:1,yref:'paper',line:{color:'#3f3f46',width:1,dash:'dot'}}))}
function phA(pp){return pp.filter(p=>p[0]>0).map(p=>({x:p[0],y:1.02,yref:'paper',text:p[1]+'×',showarrow:false,font:{size:9,color:'#52525b',family:'Inter'}}))}

function fmtTime(ms){
  const s=Math.floor(ms/1000),m=Math.floor(s/60),h=Math.floor(m/60);
  if(h>0)return h+'h '+String(m%60).padStart(2,'0')+'m '+String(s%60).padStart(2,'0')+'s';
  return m+'m '+String(s%60).padStart(2,'0')+'s';
}
const EVAL_LABELS={eval_roundtrip:'Roundtrip eval',eval_sliding:'Sliding window eval',eval_hedge:'Hedge mixer eval',swa:'Averaging weights'};
let lastWarmupCur=0, lastWarmupTotal=0;
let evalStartTime=parseInt(localStorage.getItem('pgolf-eval-start'))||0;
let evalPhase=localStorage.getItem('pgolf-eval-phase')||'';
function tickClock(){
  const t=new Date();
  const te=document.getElementById('s-time');
  if(te)te.innerText=use24h?t.toLocaleTimeString('en-GB'):t.toLocaleTimeString('en-US');
  if(lastStatus==='warmup'){
    document.getElementById('prog-elapsed').innerText='Warmup '+lastWarmupCur+'/'+lastWarmupTotal;
    document.getElementById('prog-remaining').innerText='compiling...';
    document.getElementById('prog-pct').innerText='';
    document.getElementById('prog-fill').style.width='0%';
    return;
  }
  const now=Date.now();
  const isEval=lastStatus.startsWith('eval_')||lastStatus==='swa';
  const isDone=lastStatus==='done';
  if(lastStatus==='training'){
    // Training phase: interpolate from last known train_time
    const el=lastElapsed+(now-lastFetchTime)*0.85;
    const rem=Math.max(0,lastMaxWall-el);
    const pct=Math.min(100,el/lastMaxWall*100);
    document.getElementById('prog-elapsed').innerText='';
    document.getElementById('prog-remaining').innerText=rem>0?fmtTime(rem)+' remaining':'finishing...';
    document.getElementById('prog-pct').innerText=pct.toFixed(1)+'%';
    document.getElementById('prog-fill').style.width=pct.toFixed(1)+'%';
    // Update training stage time
    const stgT=document.getElementById('stg-train-t');
    if(stgT)stgT.textContent=fmtTime(el)+' · step '+lastStep.toLocaleString()+(lastCurPh>1?' · '+lastCurPh+'×rep':'');
    evalStartTime=0;evalPhase='';
    localStorage.removeItem('pgolf-eval-start');localStorage.removeItem('pgolf-eval-phase');
  } else if(isEval){
    // Eval phase: 10 min eval budget countdown
    if(!evalStartTime||evalPhase===''){
      evalStartTime=now;evalPhase='eval';
      localStorage.setItem('pgolf-eval-start',evalStartTime);
      localStorage.setItem('pgolf-eval-phase',evalPhase);
    }
    const evalEl=now-evalStartTime;
    const evalBudget=600000; // 10 min
    const evalRem=Math.max(0,evalBudget-evalEl);
    const evalPct=Math.min(100,evalEl/evalBudget*100);
    document.getElementById('prog-elapsed').innerText='';
    document.getElementById('prog-remaining').innerText=evalRem>0?fmtTime(evalRem)+' remaining':'over budget!';
    document.getElementById('prog-pct').innerText=evalPct.toFixed(1)+'%';
    document.getElementById('prog-fill').style.width=evalPct.toFixed(1)+'%';
    if(evalRem<=0)document.getElementById('prog-remaining').style.color='var(--red)';
    else document.getElementById('prog-remaining').style.color='';
    // Tick eval stage time
    const evalStgT=document.getElementById('stg-eval-t');
    if(evalStgT)evalStgT.textContent=fmtTime(lastEvalTotal+evalEl);
  } else if(isDone){
    document.getElementById('prog-elapsed').innerText='';
    document.getElementById('prog-remaining').innerText='';
    document.getElementById('prog-pct').innerText='';
    document.getElementById('prog-fill').style.width='100%';
  }
}

function render(data){
  try{ return _render(data) }catch(e){ console.error('render error:',e); const st=document.getElementById('s-time'); if(st)st.innerText='render error: '+e.message }
}
function _render(data){
  const {runs,config}=data;
  const keys=Object.keys(runs);
  // Auto-detect refresh rate: short runs (<=10min) = 5s, long runs = 3min
  if(keys.length&&!rateManual){
    const mw=runs[keys[0]].config.max_wallclock||600;
    const newInterval=mw<=600?10000:180000;
    if(newInterval!==refreshInterval){
      refreshInterval=newInterval;
      rateIdx=RATE_OPTIONS.indexOf(refreshInterval);if(rateIdx<0)rateIdx=0;
      if(refreshTimer)clearInterval(refreshTimer);
      refreshTimer=setInterval(()=>refresh(),refreshInterval);
    }
  }
  if(!keys.length){const st=document.getElementById('s-time');if(st)st.innerText='No data';return}

  const hasSSH=keys.some(k=>runs[k].source==='ssh');

  // Run list
  const rl=document.getElementById('run-list');
  rl.innerHTML='';
  keys.forEach((k,i)=>{
    const r=runs[k];
    const tag=document.createElement('div');
    tag.className='run-tag active';
    tag.dataset.key=k;
    const ri=refreshInterval/1000;
    const rateStr=ri>=60?(ri/60)+'m':ri+'s';
    const gpuInfo=r.config.gpu?' · '+r.config.gpu:'';
    const wsInfo=r.config.world_size?r.config.world_size+'×':'';
    const hwInfo=r.config.gpu?(wsInfo+r.config.gpu):'';
    // Build tag content safely via DOM API (no innerHTML — prevents XSS via filename/gpu)
    const dot=document.createElement('span');
    dot.id='live-dot';
    dot.style.cssText='color:var(--green);opacity:0.25;transition:opacity 0.8s ease,text-shadow 0.8s ease';
    dot.textContent='●';
    tag.appendChild(dot);
    tag.appendChild(document.createTextNode('\u00a0 '));
    const label=document.createElement('span');
    if(r.source==='ssh'){
      label.textContent='live';
      if(hwInfo){
        const hw=document.createElement('span');
        hw.style.cssText='color:var(--text3);font-size:10px';
        hw.textContent=' · '+hwInfo;
        label.appendChild(hw);
      }
    } else {
      label.textContent=r.name||k;
    }
    tag.appendChild(label);
    if(r.source!=='ssh'){
      const x=document.createElement('span');
      x.className='x';
      x.textContent='×';
      x.addEventListener('click',function(e){e.stopPropagation();removeRun(k)});
      tag.appendChild(x);
    }
    rl.appendChild(tag);
  });

  // SSH health pill
  let shEl=document.getElementById('ssh-health');
  if(data.ssh_health && data.ssh_health.status!=='unknown'){
    if(!shEl){shEl=document.createElement('span');shEl.id='ssh-health';rl.parentElement.appendChild(shEl)}
    shEl.textContent='';
    const dot=document.createElement('span');dot.className='sh-dot';
    const lbl=document.createElement('span');
    const h=data.ssh_health;
    if(h.status==='ok'){dot.style.color='var(--green)';dot.textContent='●';lbl.textContent='connected'}
    else if(h.status==='stale'){dot.style.color='var(--amber)';dot.textContent='●';const ago=h.last_ok_ts?Math.round(Date.now()/1000-h.last_ok_ts)+'s ago':'';lbl.textContent='stale'+(ago?' ('+ago+')':'')}
    else if(h.status==='failing'){dot.style.color='var(--red)';dot.textContent='●';lbl.textContent=h.last_error||'failing'}
    shEl.appendChild(dot);shEl.appendChild(lbl);
  } else if(shEl){shEl.remove()}

  // Auto-derive baseline from first val_bpb if not set via CLI
  if(BASELINE===null&&!_baselineAutoSet){
    const pr=runs[keys[0]];
    if(pr&&pr.val_bpb&&pr.val_bpb.length>0){BASELINE=pr.val_bpb[0];_baselineAutoSet=true}
  }

  // Use first run for metrics
  const primary=runs[keys[0]];
  const ns=primary.steps.length, nv=primary.val_steps.length, np=primary.pert_chi.length;
  const nt=primary.train_time.length;

  // Progress bar
  const maxWall=(primary.config.max_wallclock||600)*1000; // ms, default 10min
  const elapsed=nt?primary.train_time[nt-1]:0;
  const pct=Math.min(100,elapsed/maxWall*100);
  const remaining=Math.max(0,maxWall-elapsed);
  document.getElementById('prog-fill').style.width=pct.toFixed(1)+'%';

  // Save for real-time clock
  lastWarmupCur=primary.warmup_cur||0; lastWarmupTotal=primary.warmup_total||0;
  lastElapsed=elapsed; lastMaxWall=maxWall; lastFetchTime=Date.now();
  lastStep=ns?primary.steps[ns-1]:0;
  const ph=primary.phase_switches;
  lastCurPh=ph.length?ph[ph.length-1][1]:parseInt(primary.config.num_repeats||'1');
  if(!clockTimer){
    clockTimer=setInterval(tickClock,1000);
  }
  tickClock();

  // Metrics — estimate real total steps from wallclock budget
  const lastSpd=ns?primary.step_avg[ns-1]:100;
  const estTotalSteps=Math.floor(maxWall/lastSpd);
  const realMax=Math.min(primary.config.iterations||200000, estTotalSteps);
  document.getElementById('m-step').innerText=lastStep.toLocaleString();
  document.getElementById('m-step-u').innerText='/ '+Math.round(realMax/1000)+'K';

  if(nv){
    const bpbEl=document.getElementById('m-bpb');
    bpbEl.innerText=primary.val_bpb[nv-1].toFixed(4);
    bpbEl.style.color='var(--text)';
    if(keys.length>1){
      const cmp=runs[keys[1]];
      if(cmp.val_bpb.length){
        let bi=cmp.val_steps.findIndex(s=>s>=primary.val_steps[nv-1]);
        if(bi<0)bi=cmp.val_steps.length-1;
        const diff=primary.val_bpb[nv-1]-cmp.val_bpb[bi];
        const el=document.getElementById('m-bpb-s');
        el.innerText='vs '+cmp.name+': '+(diff>=0?'+':'')+diff.toFixed(4);
        el.style.color=diff<=0?'var(--green)':'var(--red)';
      }
    } else {document.getElementById('m-bpb-s').innerText=''}
  }

  // Efficiency: bpb improvement — window auto-adapts to run length
  if(ns){
    const spd=primary.step_avg[ns-1];
    document.getElementById('m-spd').innerText=spd.toFixed(1);
    document.getElementById('m-spd-s').innerText='';
  }

  const curPh=lastCurPh;

  // Status & results
  const f=primary.finals;
  const st=primary.status||'training';
  lastStatus=st;
  // Status is now shown via stages timeline below

  // Stages timeline
  const stageOrder=['eval','swa','rt','sw','hm','ttt'];
  const stageStatus={train:'pending',eval:'pending',swa:'pending',rt:'pending',sw:'pending',hm:'pending',ttt:'pending'};
  const mw=primary.config.max_wallclock||600;
  const isEvalPhase=st==='swa'||st.startsWith('eval_')||st==='done';
  // Determine stage states
  if(st==='warmup'||st==='training'){stageStatus.train='active'}
  else{stageStatus.train='done'}
  if(isEvalPhase&&st!=='done'){stageStatus.eval='active'}
  else if(st==='done'){stageStatus.eval='done'}
  if(st==='swa'){stageStatus.swa='active'}
  else if(f.roundtrip||st==='eval_roundtrip'||st==='eval_sliding'||st==='eval_hedge'||st==='done'){stageStatus.swa='done'}
  if(st==='eval_roundtrip'){stageStatus.rt='active'}
  else if(f.roundtrip){stageStatus.rt='done'}
  if(st==='eval_sliding'){stageStatus.sw='active'}
  else if(f.sliding){stageStatus.sw='done'}
  if(st==='eval_hedge'){stageStatus.hm='active'}
  else if(f.hedge){stageStatus.hm='done'}
  if(st==='eval_ttt'){stageStatus.ttt='active'}
  else if(f.ttt){stageStatus.ttt='done'}
  // Times
  const trainTime=nt?primary.train_time[nt-1]:0;
  const fmtEval=(bpb,ms)=>{let s='';if(bpb)s+=bpb.toFixed(4);if(ms)s+=' · '+Math.round(ms/1000)+'s';return s};
  // Training info
  let trainInfo='';
  if(stageStatus.train==='active'){
    trainInfo=fmtTime(trainTime)+' · step '+lastStep.toLocaleString();
    if(curPh>1)trainInfo+=' · '+curPh+'×rep';
  } else if(stageStatus.train==='done'){
    trainInfo=fmtTime(mw*1000)+' · '+lastStep.toLocaleString()+' steps';
  }
  // Eval total time (sum of all eval times, or ticking)
  let evalTotalMs=0;
  if(f.rt_time)evalTotalMs+=f.rt_time;
  if(f.sw_time)evalTotalMs+=f.sw_time;
  if(f.hm_time)evalTotalMs+=f.hm_time;
  if(f.ttt_time)evalTotalMs+=f.ttt_time;
  // Store eval info for tickClock
  lastEvalTotal=evalTotalMs;
  const stgTimes={
    eval:stageStatus.eval==='done'?fmtTime(evalTotalMs):'',
    swa:primary.swa_count?primary.swa_count+' ckpts':'',
    rt:fmtEval(f.roundtrip,f.rt_time),
    sw:fmtEval(f.sliding,f.sw_time),
    hm:fmtEval(f.hedge,f.hm_time),
    ttt:fmtEval(f.ttt,f.ttt_time)
  };
  // Training stage (always visible)
  const trainEl=document.getElementById('stg-train');
  const trainIcon=trainEl.querySelector('.stg-icon');
  const trainTEl=document.getElementById('stg-train-t');
  trainEl.className='stage'+(stageStatus.train==='active'?' active':' done');
  trainIcon.textContent='●';
  trainIcon.style.color=stageStatus.train==='done'?'var(--text3)':'var(--text)';
  trainIcon.style.animation=stageStatus.train==='active'?'pgolf-glow 2s ease-in-out infinite':'none';
  trainTEl.textContent=trainInfo;
  trainEl.style.display='flex';
  // Eval + sub-stages
  for(const s of stageOrder){
    const el=document.getElementById('stg-'+s);
    const tEl=document.getElementById('stg-'+s+'-t');
    const isSub=s!=='eval';
    el.className='stage'+(isSub?' sub':'')+(stageStatus[s]==='active'?' active':'')+(stageStatus[s]==='done'?' done':'');
    const icon=el.querySelector('.stg-icon');
    if(stageStatus[s]==='done'){icon.textContent='●';icon.style.color='var(--text3)';icon.style.animation='none';el.style.display='flex'}
    else if(stageStatus[s]==='active'){icon.textContent='●';icon.style.color='var(--text)';icon.style.animation='pgolf-glow 2s ease-in-out infinite';el.style.display='flex'}
    else if(s==='eval'&&stageStatus.train==='done'){el.style.display='flex';icon.textContent='○';icon.style.color='var(--text3)'}
    else{el.style.display='none'}
    tEl.textContent=stgTimes[s]||'';
  }
  // Artifact size stage
  const sizeEl=document.getElementById('stg-size');
  const sizeTEl=document.getElementById('stg-size-t');
  sizeEl.style.display='none';
  if(f.artifact_bytes&&st==='done'){
    sizeEl.style.display='flex';
    sizeEl.className='stage done';
    sizeEl.querySelector('.stg-icon').style.color=f.artifact_bytes<=16000000?'var(--text3)':'var(--red)';
    sizeTEl.style.color=f.artifact_bytes<=16000000?'var(--text2)':'var(--red)';
    sizeTEl.textContent=(f.artifact_bytes/1000000).toFixed(2)+' MB';
  }

  // Crash display: red progress bar + error excerpt
  const progFill=document.getElementById('prog-fill');
  let errEl=document.getElementById('error-excerpt');
  if(st==='crashed'){
    progFill.classList.add('crashed');
    if(!errEl){errEl=document.createElement('div');errEl.id='error-excerpt';errEl.className='error-excerpt';document.getElementById('stages').after(errEl)}
    errEl.textContent=primary.error_excerpt||'Unknown error';
  } else {
    progFill.classList.remove('crashed');
    if(errEl)errEl.remove();
  }

  const logName=config.log?config.log.split('/').pop():'';
  const t=new Date();
  const timeStr=use24h?t.toLocaleTimeString('en-GB'):t.toLocaleTimeString('en-US');
  const stEl=document.getElementById('s-time');
  if(stEl)stEl.innerText=timeStr;
  const ri=refreshInterval/1000;
  const sRate=document.getElementById('s-rate');
  if(sRate)sRate.innerText=ri>=60?(ri/60)+'m':ri+'s';

  // ---- CHARTS ----
  const allPh=[...ph]; // phase lines from primary

  // 1: val_bpb — all runs overlaid + naive baseline as horizontal line
  const bpbXmax=realMax>0?Math.ceil(realMax/1000)*1000:6000;
  const bpbTraces=[];
  // Check if primary run has crossed below baseline
  const BL=BASELINE;
  let blShapes=[];
  let blAnnotations=[];
  if(BL!==null){
    const crossed=nv>0&&primary.val_bpb[nv-1]<BL;
    const blC=crossed?'rgba(255,255,255,':'rgba(255,255,255,';
    const blStyles={
      'dot':[{type:'line',x0:0,x1:bpbXmax,y0:BL,y1:BL,line:{color:blC+'0.2)',width:0.5,dash:'dot'}}],
      'dash':[{type:'line',x0:0,x1:bpbXmax,y0:BL,y1:BL,line:{color:blC+'0.2)',width:0.5,dash:'dash'}}],
      'longdash':[{type:'line',x0:0,x1:bpbXmax,y0:BL,y1:BL,line:{color:blC+'0.15)',width:0.7,dash:'longdash'}}],
      'solid':[{type:'line',x0:0,x1:bpbXmax,y0:BL,y1:BL,line:{color:blC+'0.1)',width:0.5}}],
      'glow':[
        {type:'line',x0:0,x1:bpbXmax,y0:BL,y1:BL,line:{color:blC+'0.03)',width:10}},
        {type:'line',x0:0,x1:bpbXmax,y0:BL,y1:BL,line:{color:blC+'0.06)',width:4}},
        {type:'line',x0:0,x1:bpbXmax,y0:BL,y1:BL,line:{color:blC+'0.15)',width:0.5}},
      ],
      'zone':[{type:'rect',x0:0,x1:bpbXmax,y0:BL-0.002,y1:BL+0.002,fillcolor:blC+'0.06)',line:{width:0}}],
      'dashdot':[{type:'line',x0:0,x1:bpbXmax,y0:BL,y1:BL,line:{color:blC+'0.2)',width:0.5,dash:'dashdot'}}],
      'thick':[{type:'line',x0:0,x1:bpbXmax,y0:BL,y1:BL,line:{color:blC+'0.08)',width:2}}],
      'double':[
        {type:'line',x0:0,x1:bpbXmax,y0:BL-0.001,y1:BL-0.001,line:{color:blC+'0.12)',width:0.3}},
        {type:'line',x0:0,x1:bpbXmax,y0:BL+0.001,y1:BL+0.001,line:{color:blC+'0.12)',width:0.3}},
      ],
      'none':[],
    };
    const themeBl={dark:'solid',kitty:'solid',emerald:'solid'};
    const curTheme=document.documentElement.getAttribute('data-theme')||'dark';
    blShapes=blStyles[themeBl[curTheme]||'solid'];
    const blAnnotColor=crossed?'rgba(255,255,255,0.15)':'rgba(255,255,255,0.25)';
    blAnnotations=[{x:bpbXmax,y:BL,xanchor:'right',yanchor:'bottom',
      text:'baseline '+BL,showarrow:false,font:{size:9,color:blAnnotColor}}];
  }
  const bpbLayout={...DL,title:'',
    xaxis:{...DL.xaxis,title:'',range:[0,bpbXmax],tick0:1000,dtick:1000},yaxis:{...DL.yaxis,title:'',range:[1.1,1.45]},
    shapes:[...blShapes],
    annotations:[...blAnnotations]};
  keys.forEach((k,i)=>{
    const r=runs[k];
    if(r.val_steps.length){
      const lc=getLineColor();
      const opacity=i===0?1:0.45;
      const rgba=lc.startsWith('#')?'rgba('+parseInt(lc.slice(1,3),16)+','+parseInt(lc.slice(3,5),16)+','+parseInt(lc.slice(5,7),16)+','+opacity+')':lc;
      bpbTraces.push({x:r.val_steps,y:r.val_bpb,mode:'lines+markers',marker:{size:i===0?4:3,color:rgba},
        line:{color:rgba,width:i===0?2.5:1.5,dash:DASH_STYLES[i%DASH_STYLES.length]},name:r.name||k});
    }
  });
  safeUpdate('c-bpb',bpbTraces,bpbLayout);

}

function safeUpdate(id,traces,layout){
  const el=document.getElementById(id);
  if(!el)return;
  if(id==='c-bpb'){
    let xr=null;
    if(el.layout&&el.layout.xaxis&&el.layout.xaxis.range&&el.layout.xaxis.autorange===false)xr=el.layout.xaxis.range.slice();
    Plotly.react(id,traces,{...layout,dragmode:'pan'},{responsive:true,scrollZoom:true,displayModeBar:false});
    if(xr)Plotly.relayout(id,{'xaxis.range':xr,'xaxis.autorange':false});
  }
}


// Theme switcher — click pgolf logo, persist in localStorage
const themes=['dark','kitty','emerald'];
let themeIdx=parseInt(localStorage.getItem('pgolf-theme-idx'))||0;
document.documentElement.setAttribute('data-theme',themes[themeIdx]);
DL=getPlotTheme();
document.getElementById('logo-h1').addEventListener('click',()=>{
  themeIdx=(themeIdx+1)%themes.length;
  localStorage.setItem('pgolf-theme-idx',themeIdx);
  localStorage.removeItem('pgolf-font');
  document.documentElement.setAttribute('data-theme',themes[themeIdx]);
  document.documentElement.style.removeProperty('--font-ui');
  DL=getPlotTheme();
  // Show loading overlay during theme transition
  location.reload();
});

// Diffusing text effect on logo (per-char spans with glow + idle effects)
(function(){
  const NOISE='░▒▓█▀▄▌▐╌╍─│┌┐└┘├┤┬┴┼▪▫◊●○■□▲△◆◇∆∇≈±×÷αβγδεζηθλΣΩ';
  const pick=()=>NOISE[Math.floor(Math.random()*NOISE.length)];
  const el=document.getElementById('logo-text');
  const target='pgolf dashboard';
  const len=target.length;
  let idle=false,hovered=false,frame=0;
  const glitch=new Set();
  const nonSpace=[];
  for(let i=0;i<len;i++)if(target[i]!==' ')nonSpace.push(i);
  const pickIdx=()=>nonSpace[Math.floor(Math.random()*nonSpace.length)];

  // Build per-char spans ("pgolf" in --text, " dashboard" in --text2)
  const splitAt=5; // "pgolf" length
  function initSpans(){
    el.innerHTML='';
    for(let i=0;i<len;i++){
      const s=document.createElement('span');
      const col=i<splitAt?'var(--text)':'var(--text2)';
      s.style.cssText='display:inline-block;width:1ch;text-align:center;transition:text-shadow 0.4s ease,opacity 0.3s ease;color:'+col;
      s.textContent=target[i]===' '?' ':pick();
      el.appendChild(s);
    }
  }
  function setChar(i,ch,active){
    const s=el.children[i];if(!s)return;
    s.textContent=ch;
    s.style.opacity=active?'0.85':'1';
    s.style.textShadow=active?'0 0 4px currentColor,0 0 10px currentColor':'none';
    s.style.transition=active?'none':'text-shadow 0.4s ease,opacity 0.3s ease';
  }

  // Crystallize: noise → target
  function crystallize(){
    idle=false;glitch.clear();
    const locks=target.split('').map(c=>c===' '?0:(0.15+Math.random()*0.85)*1400);
    const start=performance.now();
    function tick(){
      const elapsed=performance.now()-start;
      let done=true;
      for(let i=0;i<len;i++){
        if(target[i]===' '){setChar(i,' ',false);continue}
        if(elapsed>=locks[i]){setChar(i,target[i],false)}
        else{done=false;const p=Math.min(elapsed/locks[i],1);setChar(i,Math.random()<p*0.4?target[i]:pick(),true)}
      }
      if(!done)frame=requestAnimationFrame(tick);
      else{idle=true;idleTimer=setTimeout(runIdle,4000+Math.random()*6000)}
    }
    frame=requestAnimationFrame(tick);
  }

  // Hover: continuous noise with letter peeks
  function startNoise(){
    idle=false;clearTimeout(idleTimer);
    const peekUntil=new Float64Array(len);
    function tick(){
      if(!hovered)return;
      const now=performance.now();
      for(let i=0;i<len;i++){
        if(target[i]===' ')continue;
        if(peekUntil[i]<now&&Math.random()<0.015)peekUntil[i]=now+200+Math.random()*400;
        setChar(i,now<peekUntil[i]?target[i]:pick(),now>=peekUntil[i]);
      }
      frame=requestAnimationFrame(tick);
    }
    frame=requestAnimationFrame(tick);
  }

  // Idle: spread glitch effect
  let idleTimer=0;
  function runIdle(){
    if(hovered||!idle||nonSpace.length===0)return;
    glitch.clear();glitch.add(pickIdx());
    const dur=1600*(0.6+Math.random()*0.8);
    const start=performance.now();
    let spreadLast=start;
    function tick(){
      const now=performance.now();
      if(now-start>=dur||hovered){
        glitch.clear();for(let i=0;i<len;i++)setChar(i,target[i],false);
        if(!hovered)idleTimer=setTimeout(runIdle,4000+Math.random()*6000);
        return;
      }
      // spread
      if(now-spreadLast>150){
        spreadLast=now;
        const add=[];
        for(const idx of glitch){
          if(idx>0&&!glitch.has(idx-1)&&target[idx-1]!==' '&&Math.random()<0.4)add.push(idx-1);
          if(idx<len-1&&!glitch.has(idx+1)&&target[idx+1]!==' '&&Math.random()<0.4)add.push(idx+1);
        }
        for(const i of add)glitch.add(i);
      }
      for(let i=0;i<len;i++){
        if(glitch.has(i)){setChar(i,pick(),true)}
        else{setChar(i,target[i],false)}
      }
      frame=requestAnimationFrame(tick);
    }
    frame=requestAnimationFrame(tick);
  }

  initSpans();
  el.parentElement.addEventListener('mouseenter',()=>{hovered=true;cancelAnimationFrame(frame);clearTimeout(idleTimer);startNoise()});
  el.parentElement.addEventListener('mouseleave',()=>{hovered=false;cancelAnimationFrame(frame);crystallize()});
  crystallize();
})();

// SSH is now configured via CLI --ssh flag only (no browser-side SSH to prevent RCE)
async function removeRun(key){
  await fetch(`/api/remove?key=${encodeURIComponent(key)}&token=${encodeURIComponent(TOKEN)}`,{method:'POST'});
  refresh();
}

// File upload (button + drag&drop)
async function uploadFiles(files){
  for(const f of files){
    const fd=new FormData();
    fd.append('file',f);
    await fetch(`/api/upload?token=${encodeURIComponent(TOKEN)}`,{method:'POST',body:fd});
  }
  refresh();
}
document.getElementById('file-input').addEventListener('change',function(e){
  if(e.target.files.length)uploadFiles(e.target.files);
  e.target.value='';
});
// Drag & drop anywhere on page
document.addEventListener('dragover',function(e){e.preventDefault();e.dataTransfer.dropEffect='copy'});
document.addEventListener('drop',function(e){
  e.preventDefault();
  const files=[...e.dataTransfer.files].filter(f=>f.name.endsWith('.log')||f.name.endsWith('.txt'));
  if(files.length)uploadFiles(files);
});
document.getElementById('s-time').addEventListener('click',()=>{use24h=!use24h;localStorage.setItem('pgolf-24h',use24h);refresh()});
function cycleRate(){
  rateManual=true;
  rateIdx=(rateIdx+1)%RATE_OPTIONS.length;
  localStorage.setItem('pgolf-rate-idx',rateIdx);
  refreshInterval=RATE_OPTIONS[rateIdx];
  if(refreshTimer)clearInterval(refreshTimer);
  refreshTimer=setInterval(()=>refresh(),refreshInterval);
  refresh();
}

async function refresh(){
  try{
    const r=await fetch(`/api/data?token=${encodeURIComponent(TOKEN)}`);const d=await r.json();render(d);
    if(Object.keys(d.runs).length&&window._hideLoading)window._hideLoading();
    const dot=document.getElementById('live-dot');
    if(dot){dot.style.opacity='1';dot.style.textShadow='0 0 8px var(--green),0 0 16px var(--green)';setTimeout(()=>{dot.style.opacity='0.25';dot.style.textShadow='none'},800)}
  }catch(e){const st=document.getElementById('s-time');if(st)st.innerText='error: '+e}
}

refresh();
refreshTimer=setInterval(()=>refresh(), refreshInterval);

// Loading overlay — diffusing logo + typewriter phrases, auto-hide on first data
(function(){
  const NOISE='░▒▓█▀▄▌▐╌╍─│┌┐└┘├┤┬┴┼▪▫◊●○■□▲△◆◇∆∇≈±×÷αβγδεζηθλΣΩ';
  const pickN=()=>NOISE[Math.floor(Math.random()*NOISE.length)];
  // Diffusing logo in loading overlay
  const logoEl=document.getElementById('loading-logo');
  if(logoEl){
    const target='pgolf dashboard';
    const splitAt=5;
    // Init spans
    for(let i=0;i<target.length;i++){
      const s=document.createElement('span');
      const col=i<splitAt?'var(--text)':'var(--text3)';
      s.style.cssText='display:inline-block;width:1ch;text-align:center;color:'+col+';transition:text-shadow 0.4s ease,opacity 0.3s ease';
      s.textContent=target[i]===' '?' ':pickN();
      logoEl.appendChild(s);
    }
    // Crystallize then idle loop
    const len=target.length;
    const nonSpace=[];for(let i=0;i<len;i++)if(target[i]!==' ')nonSpace.push(i);
    const pickIdx=()=>nonSpace[Math.floor(Math.random()*nonSpace.length)];
    function setC(i,ch,active){
      const s=logoEl.children[i];if(!s)return;
      s.textContent=ch;
      s.style.opacity=active?'0.85':'1';
      s.style.textShadow=active?'0 0 4px currentColor,0 0 10px currentColor':'none';
      s.style.transition=active?'none':'text-shadow 0.4s ease,opacity 0.3s ease';
    }
    // Crystallize
    const locks=target.split('').map(c=>c===' '?0:(0.15+Math.random()*0.85)*1400);
    const cStart=performance.now();
    let idleTimer=0,cDone=false;
    function cTick(){
      const elapsed=performance.now()-cStart;
      let done=true;
      for(let i=0;i<len;i++){
        if(target[i]===' '){setC(i,' ',false);continue}
        if(elapsed>=locks[i]){setC(i,target[i],false)}
        else{done=false;const p=Math.min(elapsed/locks[i],1);setC(i,Math.random()<p*0.4?target[i]:pickN(),true)}
      }
      if(!done)requestAnimationFrame(cTick);
      else{cDone=true;idleTimer=setTimeout(idleSpread,3000+Math.random()*4000)}
    }
    // Idle spread effect (continuous while loading)
    function idleSpread(){
      if(!cDone)return;
      const glitch=new Set();glitch.add(pickIdx());
      const dur=1600*(0.6+Math.random()*0.8);
      const iStart=performance.now();
      let spreadLast=iStart;
      function iTick(){
        const now=performance.now();
        if(now-iStart>=dur){
          glitch.clear();for(let i=0;i<len;i++)setC(i,target[i],false);
          idleTimer=setTimeout(idleSpread,2000+Math.random()*3000);
          return;
        }
        if(now-spreadLast>150){
          spreadLast=now;
          const add=[];
          for(const idx of glitch){
            if(idx>0&&!glitch.has(idx-1)&&target[idx-1]!==' '&&Math.random()<0.4)add.push(idx-1);
            if(idx<len-1&&!glitch.has(idx+1)&&target[idx+1]!==' '&&Math.random()<0.4)add.push(idx+1);
          }
          for(const i of add)glitch.add(i);
        }
        for(let i=0;i<len;i++){
          if(glitch.has(i)){setC(i,pickN(),true)}else{setC(i,target[i],false)}
        }
        requestAnimationFrame(iTick);
      }
      requestAnimationFrame(iTick);
    }
    requestAnimationFrame(cTick);
  }

  const PHRASES=['Connecting...','Fetching log...','Parsing data...','Almost there...','One moment...','Getting ready...'];
  const el=document.getElementById('loading-text');
  const overlay=document.getElementById('loading-overlay');
  if(!el||!overlay)return;
  let pi=0,typing=false,hidden=false;

  function typePhrase(phrase,cb){
    typing=true;
    let i=0;
    el.textContent='';
    const timer=setInterval(()=>{
      if(hidden){clearInterval(timer);return}
      if(i<=phrase.length){el.textContent=phrase.slice(0,i);i++}
      else{clearInterval(timer);typing=false;cb&&cb()}
    },40);
  }

  function cycle(){
    if(hidden)return;
    typePhrase(PHRASES[pi%PHRASES.length],()=>{
      setTimeout(()=>{
        if(hidden)return;
        el.style.transition='opacity 0.3s ease';
        el.style.opacity='0';
        setTimeout(()=>{
          el.style.transition='none';
          el.style.opacity='1';
          pi++;
          cycle();
        },400);
      },2000);
    });
  }
  setTimeout(cycle,300);

  // Hide overlay on first successful data
  window._hideLoading=function(){
    if(hidden)return;
    hidden=true;
    overlay.style.transition='opacity 0.8s ease';
    overlay.style.opacity='0';
    setTimeout(()=>overlay.remove(),1000);
  };
})();

// Restore saved font override
(function(){
  const saved=localStorage.getItem('pgolf-font');
  if(saved)document.documentElement.style.setProperty('--font-ui',saved);
})();
</script></body></html>"""


# ---------------------------------------------------------------------------
# Remove uploaded run
# ---------------------------------------------------------------------------
@app.post("/api/remove")
def remove_run(key: str, token: str = None):
    _check_token(token)
    _runs.pop(key, None)
    return JSONResponse({"status": "ok"})


@app.get("/api/log")
def get_log(token: str = None):
    """Return raw log text for download."""
    _check_token(token)
    text = fetch_ssh_log()
    if not text:
        for f in _config["local_files"]:
            p = Path(f)
            if p.exists():
                text = p.read_text()
                break
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(text or "no log available", media_type="text/plain")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Parameter Golf Training Dashboard")
    parser.add_argument("logs", nargs="*", help="Local .log files to display")
    parser.add_argument("--compare", nargs="*", default=[], help="Comparison log files (overlay)")
    parser.add_argument("--ssh", action="append", default=None, help='SSH connection: "root@host -p 1234 -i key" (can be repeated)')
    parser.add_argument("--remote-log", default=None, help="Log path on remote host")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address (default: 127.0.0.1; use 0.0.0.0 to expose — NOT RECOMMENDED without firewall)")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--refresh", type=int, default=180, help="Auto-refresh seconds")
    parser.add_argument("--auto-stop", action="store_true", help="Auto-save log and stop RunPod pod when training completes")
    parser.add_argument("--save-dir", default=".", help="Directory to save logs on auto-stop")
    parser.add_argument("--insecure-host-key", action="store_true",
                        help="Use StrictHostKeyChecking=no (NOT RECOMMENDED — vulnerable to MITM)")
    parser.add_argument("--baseline", type=float, default=None, help="Baseline BPB value for chart reference line")
    parser.add_argument("--notify-webhook", default=None, help="Discord/Slack webhook URL for notifications")
    parser.add_argument("--notify-threshold", type=float, default=None, help="BPB threshold for webhook notification")
    args = parser.parse_args()

    _config["local_files"] = args.logs
    _config["compare_files"] = args.compare
    _config["refresh"] = args.refresh
    _config["auto_stop"] = args.auto_stop
    _config["save_dir"] = args.save_dir
    _config["baseline"] = args.baseline
    _config["webhook_url"] = args.notify_webhook
    _config["webhook_threshold"] = args.notify_threshold
    if args.ssh:
        remote_log = args.remote_log or "LATEST"
        for idx, ssh_str in enumerate(args.ssh):
            ssh_args = _parse_ssh_string(ssh_str, insecure=args.insecure_host_key)
            ssh_host = _extract_ssh_host(ssh_args)
            target = {
                "ssh_args": ssh_args,
                "ssh_host": ssh_host,
                "label": f"pod-{idx}" if len(args.ssh) > 1 else "SSH",
                "remote_log": remote_log,
                "byte_offset": 0,
                "local_cache": "",
                "health": {"last_ok_ts": 0, "consecutive_failures": 0, "last_error": "", "status": "unknown"},
            }
            _config["ssh_targets"].append(target)
        # Keep backward compat: set ssh_args/ssh_host from first target
        _config["ssh_args"] = _config["ssh_targets"][0]["ssh_args"]
        _config["ssh_host"] = _config["ssh_targets"][0]["ssh_host"]
        _config["remote_log"] = remote_log

    if args.host == "0.0.0.0":
        print("\n\033[93m  WARNING: Dashboard bound to 0.0.0.0 — anyone on your network can access it.\033[0m")
        print("  Use --host 127.0.0.1 to restrict to this machine.\n")

    print(f"Dashboard: http://localhost:{args.port}?token={_auth_token}")
    print(f"  Auth token: {_auth_token}")
    if args.logs:
        print(f"  Local logs: {args.logs}")
    if args.compare:
        print(f"  Compare: {args.compare}")
    if args.ssh:
        for idx, ssh_str in enumerate(args.ssh):
            label = _config["ssh_targets"][idx]["label"]
            print(f"  SSH [{label}]: {ssh_str} -> {_config['remote_log']}")
    if args.notify_webhook:
        print(f"  Webhook: {args.notify_webhook}")
        if args.notify_threshold:
            print(f"  Notify threshold: {args.notify_threshold}")
    if args.baseline is not None:
        print(f"  Baseline BPB: {args.baseline}")
    if not args.logs and not args.ssh:
        print("  No data source — drag & drop .log files in browser")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
