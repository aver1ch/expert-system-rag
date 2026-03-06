import json
import os
import re
import subprocess
import time
import argparse
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class SessionInfo:
    job_id: str
    node: str


class HpcTunnelManager:
    def __init__(self) -> None:
        self.enabled = os.getenv("HPC_TUNNEL_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
        self.user = os.getenv("HPC_SSH_USER", "averichie")
        self.login_host = os.getenv("HPC_LOGIN_HOST", "login1.hpc.spbstu.ru")
        self.slurm_cluster = os.getenv("HPC_SLURM_CLUSTER", "nv")
        self.slurm_constraint = os.getenv("HPC_SLURM_CONSTRAINT", "ollama").strip()
        self.local_port = int(os.getenv("HPC_LOCAL_PORT", "11434"))
        self.remote_port = int(os.getenv("HPC_REMOTE_PORT", "11434"))
        self.max_weekly_hours = float(os.getenv("HPC_MAX_WEEKLY_HOURS", "10"))
        self.requested_job_hours = float(os.getenv("HPC_REQUESTED_JOB_HOURS", "2"))
        self.job_name = os.getenv("HPC_JOB_NAME", "ollama")
        self.gpu_count = int(os.getenv("HPC_GPU_COUNT", "3"))
        self.mem_per_gpu = os.getenv("HPC_MEM_PER_GPU", "32G")
        self.slurm_time = os.getenv("HPC_SLURM_TIME", "2:00:00")
        self.remote_slurm_path = os.getenv("HPC_REMOTE_SLURM_PATH", "~/ollama.slurm")
        self.ollama_ready_timeout_sec = int(os.getenv("HPC_OLLAMA_READY_TIMEOUT_SEC", "0"))
        self.ollama_ready_interval_sec = int(os.getenv("HPC_OLLAMA_READY_INTERVAL_SEC", "5"))
        self.identity_file = os.getenv("HPC_SSH_IDENTITY_FILE", "").strip()
        self.identity_agent = os.getenv("SSH_AUTH_SOCK", "").strip()

        runtime_dir = Path(os.getenv("HPC_RUNTIME_DIR", "/app/runtime"))
        runtime_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = runtime_dir / "hpc_usage.json"
        self.ssh_socket_path = Path(os.getenv("HPC_SSH_SOCKET_PATH", "/tmp/hpc_tunnel.sock"))
        self.known_hosts_path = runtime_dir / "known_hosts"
        self.strict_host_key_checking = os.getenv("HPC_SSH_STRICT_HOST_CHECKING", "accept-new")

        self.session: Optional[SessionInfo] = None

    def start(self) -> None:
        if not self.enabled:
            print("[HPC] Tunnel automation disabled (HPC_TUNNEL_ENABLED=0).")
            return
        state = self._load_state()
        state = self._normalize_state_for_week(state)
        self._settle_stale_session(state)
        self._check_weekly_budget(state)
        charged_seconds = int(self.requested_job_hours * 3600)
        state["used_seconds"] = int(state.get("used_seconds", 0)) + charged_seconds

        try:
            session = self._find_existing_running_session()
            reuse_existing = session is not None
            if session is not None:
                checked = self._get_job_session(session.job_id)
                if checked is None:
                    print(f"[HPC] Previously found job {session.job_id} is no longer running, requesting a new job.")
                    session = None
                    reuse_existing = False
                else:
                    session = checked
            if session is None:
                self._upload_slurm_script()
                session = self._submit_and_wait_for_running()
            self._open_tunnel(session.node)
            self._wait_ollama_ready()
        except Exception:
            # Откатываем списание квоты, если запуск сессии не удался
            state["used_seconds"] = max(0, int(state.get("used_seconds", 0)) - charged_seconds)
            self._save_state(state)
            raise

        planned_seconds = int(self.requested_job_hours * 3600)
        state["active_session"] = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "planned_seconds": planned_seconds,
            "job_id": session.job_id,
            "node": session.node,
            "reuse_existing": reuse_existing,
        }
        self._save_state(state)
        self.session = session
        print(f"[HPC] Tunnel is ready: localhost:{self.local_port} -> {session.node}:{self.remote_port}")

    def stop(self) -> None:
        if not self.enabled:
            return

        state = self._normalize_state_for_week(self._load_state())
        active = state.get("active_session")

        if active:
            job_id = str(active.get("job_id", "")).strip()
            reuse_existing = bool(active.get("reuse_existing", False))
            if job_id and not reuse_existing:
                self._run_ssh(f"scancel -M {self.slurm_cluster} {job_id}", check=False)
            state["active_session"] = None
            self._save_state(state)

        self._close_tunnel()

    def _normalize_state_for_week(self, state: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        week_start = (now - timedelta(days=now.weekday())).date().isoformat()
        limit_seconds = int(self.max_weekly_hours * 3600)
        if state.get("week_start") != week_start:
            return {"week_start": week_start, "used_seconds": 0, "limit_seconds": limit_seconds, "active_session": None}
        if "used_seconds" not in state:
            state["used_seconds"] = 0
        if "limit_seconds" not in state:
            state["limit_seconds"] = limit_seconds
        return state

    def _settle_stale_session(self, state: Dict[str, Any]) -> None:
        active = state.get("active_session")
        if not active:
            return
        state["active_session"] = None
        self._save_state(state)

    def _check_weekly_budget(self, state: Dict[str, Any]) -> None:
        used_seconds = int(state.get("used_seconds", 0))
        limit_seconds = int(state.get("limit_seconds", int(self.max_weekly_hours * 3600)))
        if used_seconds + int(self.requested_job_hours * 3600) > limit_seconds:
            raise RuntimeError(
                "[HPC] Weekly GPU quota exceeded: "
                f"used={used_seconds/3600.0:.2f}h, requested={self.requested_job_hours:.2f}h, limit={limit_seconds/3600.0:.2f}h"
            )

    def _upload_slurm_script(self) -> None:
        script = self._render_slurm_script()
        target = self.remote_slurm_path
        ssh_cmd = self._ssh_base_cmd() + [f"{self.user}@{self.login_host}", f"cat > {target}"]
        result = subprocess.run(
            ssh_cmd,
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"[HPC] Failed to upload slurm script: {result.stderr.strip()}")

    def _submit_and_wait_for_running(self) -> SessionInfo:
        submit_out = self._run_ssh(f"sbatch {self.remote_slurm_path}")
        match = re.search(r"Submitted batch job\s+(\d+)(?:\s+on\s+cluster\s+\S+)?", submit_out)
        if not match:
            raise RuntimeError(f"[HPC] Unable to parse sbatch output: {submit_out.strip()}")

        job_id = match.group(1)
        deadline = time.time() + int(os.getenv("HPC_START_TIMEOUT_SEC", "600"))

        while time.time() < deadline:
            status_out = self._run_ssh(
                f"squeue -M {self.slurm_cluster} -h -j {job_id} -o '%T|%N'",
                check=False,
            ).strip()

            if not status_out:
                time.sleep(5)
                continue

            status = ""
            node = ""
            for line in status_out.splitlines():
                if "|" not in line:
                    continue
                parts = line.split("|", 1)
                if len(parts) != 2:
                    continue
                status = parts[0].strip()
                node = parts[1].strip()
                break

            if not status:
                time.sleep(5)
                continue

            if status == "RUNNING" and node and node not in {"(null)", "n/a"}:
                print(f"[HPC] Job {job_id} is RUNNING on node {node}")
                return SessionInfo(job_id=job_id, node=node)

            if status in {"CANCELLED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}:
                raise RuntimeError(f"[HPC] Job {job_id} failed with status {status}")

            time.sleep(5)

        self._run_ssh(f"scancel -M {self.slurm_cluster} {job_id}", check=False)
        raise RuntimeError(f"[HPC] Timeout while waiting for job {job_id} to start (job {job_id} cancelled)")

    def _find_existing_running_session(self) -> Optional[SessionInfo]:
        out = self._run_ssh(
            f"squeue -M {self.slurm_cluster} -h -u {self.user} -n {self.job_name} -o '%i|%T|%N'",
            check=False,
        )
        for line in out.splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 3:
                continue
            job_id, status, node = parts
            if status == "RUNNING" and node and node not in {"(null)", "n/a"}:
                print(f"[HPC] Reusing running job {job_id} on node {node}")
                return SessionInfo(job_id=job_id, node=node)
        return None

    def _get_job_session(self, job_id: str) -> Optional[SessionInfo]:
        out = self._run_ssh(
            f"squeue -M {self.slurm_cluster} -h -j {job_id} -o '%i|%T|%N'",
            check=False,
        ).strip()
        if not out:
            return None
        out_job_id = ""
        status = ""
        node = ""
        for line in out.splitlines():
            if "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|", 2)]
            if len(parts) != 3:
                continue
            out_job_id, status, node = parts
            break
        if not out_job_id:
            return None
        if out_job_id != job_id:
            return None
        if status != "RUNNING" or not node or node in {"(null)", "n/a"}:
            return None
        return SessionInfo(job_id=out_job_id, node=node)

    def _open_tunnel(self, node: str) -> None:
        if self.ssh_socket_path.exists():
            self.ssh_socket_path.unlink(missing_ok=True)
        cmd = [
            *self._ssh_base_cmd(),
            "-f",
            "-N",
            "-M",
            "-S",
            str(self.ssh_socket_path),
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30",
            "-L",
            f"{self.local_port}:{node}:{self.remote_port}",
            f"{self.user}@{self.login_host}",
        ]
        result = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(f"[HPC] SSH tunnel failed to start: {stderr}")

    def _close_tunnel(self) -> None:
        if not self.ssh_socket_path.exists():
            return
        cmd = [
            *self._ssh_base_cmd(),
            "-S",
            str(self.ssh_socket_path),
            "-O",
            "exit",
            f"{self.user}@{self.login_host}",
        ]
        subprocess.run(cmd, text=True, capture_output=True, check=False)
        self.ssh_socket_path.unlink(missing_ok=True)

    def _wait_ollama_ready(self) -> None:
        started = time.time()
        deadline = time.time() + self.ollama_ready_timeout_sec if self.ollama_ready_timeout_sec > 0 else None
        url = f"http://127.0.0.1:{self.local_port}/api/tags"
        last_error = ""
        attempts = 0

        while True:
            if deadline is not None and time.time() >= deadline:
                break
            attempts += 1
            try:
                with urllib.request.urlopen(url, timeout=6) as resp:
                    if resp.status == 200:
                        elapsed_ms = int((time.time() - started) * 1000)
                        print(f"[HPC] Ollama ready on tunnel after {elapsed_ms}ms attempts={attempts}", flush=True)
                        return
                    last_error = f"status={resp.status}"
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
                last_error = str(exc)
            if attempts % 6 == 0:
                elapsed_ms = int((time.time() - started) * 1000)
                print(f"[HPC] Waiting Ollama on tunnel attempts={attempts} elapsed_ms={elapsed_ms} last_error={last_error}", flush=True)
            time.sleep(max(1, self.ollama_ready_interval_sec))

        raise RuntimeError(f"[HPC] Ollama did not become ready on tunnel {url}: {last_error}")

    def _run_ssh(self, remote_cmd: str, check: bool = True) -> str:
        cmd = self._ssh_base_cmd() + [f"{self.user}@{self.login_host}", remote_cmd]
        result = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if check and result.returncode != 0:
            raise RuntimeError(
                f"[HPC] SSH command failed: '{remote_cmd}'\n"
                f"stdout: {result.stdout.strip()}\n"
                f"stderr: {result.stderr.strip()}"
            )
        return result.stdout

    def _ssh_base_cmd(self) -> list[str]:
        cmd = [
            "ssh",
            "-o", f"StrictHostKeyChecking={self.strict_host_key_checking}",
            "-o", f"UserKnownHostsFile={self.known_hosts_path}",
        ]
        if self.identity_file:
            cmd.extend(["-o", "IdentitiesOnly=yes", "-i", self.identity_file])
        if self.identity_agent:
            cmd.extend(["-o", f"IdentityAgent={self.identity_agent}"])
        return cmd

    def _render_slurm_script(self) -> str:
        sleep_seconds = max(1, int(self.requested_job_hours * 3600))
        return (
            "#!/bin/bash\n"
            f"#SBATCH -M {self.slurm_cluster}\n"
            f"#SBATCH -C {self.slurm_constraint}\n"
            f"#SBATCH -G {self.gpu_count}\n"
            f"#SBATCH --mem-per-gpu={self.mem_per_gpu}\n"
            f"#SBATCH -t {self.slurm_time}\n"
            f"#SBATCH -J {self.job_name}\n"
            "echo \"Ollama started on $(hostname -s):11434\" > ollama.out\n"
            f"sleep {sleep_seconds}\n"
        )

    def _load_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_state(self, state: Dict[str, Any]) -> None:
        self.state_path.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")

    @staticmethod
    def _parse_iso(value: Any) -> Optional[datetime]:
        if not isinstance(value, str) or not value:
            return None
        try:
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage HPC Ollama SLURM job and SSH tunnel")
    parser.add_argument("action", choices=["start", "stop"])
    args = parser.parse_args()

    manager = HpcTunnelManager()
    if args.action == "start":
        manager.start()
    else:
        manager.stop()


def load_quota_state(runtime_dir: str = "/app/runtime") -> Dict[str, Any]:
    state_path = Path(runtime_dir) / "hpc_usage.json"
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
