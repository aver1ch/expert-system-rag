#!/bin/sh
set -eu

cleanup() {
  python /app/hpc_tunnel.py stop >/dev/null 2>&1 || true
}

if ! python /app/hpc_tunnel.py start; then
  echo "[HPC] Tunnel startup failed."
  if [ "${HPC_TUNNEL_REQUIRED:-0}" = "1" ] || [ "${HPC_TUNNEL_REQUIRED:-0}" = "true" ]; then
    echo "[HPC] Tunnel is required, exiting."
    exit 1
  fi
  echo "[HPC] Continue without tunnel (HPC_TUNNEL_REQUIRED=0)."
fi

trap cleanup INT TERM EXIT
uvicorn service:app --host 0.0.0.0 --port 8000
