# Changelog

## [0.2.0] - 2026-04-08

### Added
- SSH health indicator (green/amber/red) in header
- Crash/stall detection (OOM, Traceback, Killed) with error excerpt
- Dynamic baseline via `--baseline` CLI arg (auto-derives if unset)
- Incremental SSH log fetching (`tail -c` instead of full `cat`)
- `/healthz` endpoint with version, uptime, SSH status
- Multi-SSH support (`--ssh` repeatable for multiple pods)
- Discord/Slack webhook notifications (`--notify-webhook`, `--notify-threshold`)
- File upload button (+) and drag & drop in browser
- Auto-stop re-arm on log file rotation
- 38 pytest tests + GitHub Actions CI (Python 3.10-3.12)
- MIT LICENSE, requirements.txt

### Changed
- Charts now monochrome with dash styles instead of colored lines
- Removable runs via x button (not just uploads)

### Security
- Fix RCE via `/api/data?ssh=` query parameter (#1)
- Fix 0.0.0.0 bind with no auth (#2) — now 127.0.0.1 + token
- Fix auto-stop targeting wrong pod (#3)
- Fix StrictHostKeyChecking=no (#4) — now accept-new
- Fix command injection via LATEST filename (#5)
- Fix stored XSS via innerHTML (#6) — now textContent/createElement
- Upload filenames sanitized, 50 MB size limit
- SSH cache moved from /tmp to ~/.pgolf_cache/ (mode 700)

## [0.1.0] - 2026-04-07

### Added
- Initial release
- Real-time val BPB chart with Plotly.js
- SSH remote log monitoring with LATEST mode
- Local log viewing and comparison overlay
- Progress bar with wallclock countdown
- Eval stage tracker (SWA, roundtrip, sliding window, hedge, TTT)
- Auto-stop RunPod pod on training completion
- Three themes: dark, kitty, emerald
- Animated logo with diffusing text effect
