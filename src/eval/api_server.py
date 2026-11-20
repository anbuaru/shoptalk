"""Start / stop the Phase 7 FastAPI service as a subprocess (Colab-friendly)."""
from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


def detect_app_module(repo: str | Path, pkg: str = "src/api") -> tuple[str, bool]:
    """Find `<var> = FastAPI(` (or a `create_app` factory) under src/api.
    Returns ("src.api.main:app", is_factory)."""
    repo = Path(repo)
    files = sorted((repo / pkg).glob("*.py"),
                   key=lambda p: (p.stem not in ("main", "app", "server", "api"), p.stem))
    dotted = pkg.strip("/").replace("/", ".")
    for f in files:
        txt = f.read_text(errors="ignore")
        m = re.search(r"^(\w+)\s*(?::\s*\w+\s*)?=\s*FastAPI\(", txt, re.M)
        if m:
            return f"{dotted}.{f.stem}:{m.group(1)}", False
    for f in files:
        if re.search(r"^def create_app\(", f.read_text(errors="ignore"), re.M):
            return f"{dotted}.{f.stem}:create_app", True
    raise FileNotFoundError(f"No FastAPI app found under {repo / pkg}; set API_APP in the config cell")


def free_port(port: int) -> None:
    """Kill whatever is listening on `port` (left over from a previous run)."""
    try:
        import psutil
        for c in psutil.net_connections(kind="inet"):
            if c.laddr and c.laddr.port == port and c.status == psutil.CONN_LISTEN and c.pid:
                try:
                    os.kill(c.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        time.sleep(1)
    except Exception:
        subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)


class ApiServer:
    def __init__(self, repo: str | Path, app: str, port: int = 8000, env: dict | None = None,
                 log_path: str | Path | None = None, factory: bool = False):
        self.repo, self.app, self.port, self.factory = Path(repo), app, port, factory
        self.env = {**os.environ, **{k: str(v) for k, v in (env or {}).items()}}
        self.env["PYTHONPATH"] = f"{self.repo}:{self.env.get('PYTHONPATH', '')}"
        self.log_path = Path(log_path or self.repo / "logs" / "phase9_uvicorn.log")
        self.proc: subprocess.Popen | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> subprocess.Popen:
        free_port(self.port)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "uvicorn", self.app, "--host", "127.0.0.1",
               "--port", str(self.port), "--log-level", "warning"]
        if self.factory:
            cmd.append("--factory")
        self._log = open(self.log_path, "a")
        self._log.write(f"\n===== start {time.strftime('%F %T')} : {' '.join(cmd)}\n")
        self._log.flush()
        self.proc = subprocess.Popen(cmd, cwd=self.repo, env=self.env, stdout=self._log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        return self.proc

    def stop(self, timeout: float = 30) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=timeout)
            except Exception:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except Exception:
                    pass
        self.proc = None
        try:
            self._log.close()
        except Exception:
            pass

    def tail(self, n: int = 40) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="ignore").splitlines()[-n:])
        except FileNotFoundError:
            return "(no log yet)"
