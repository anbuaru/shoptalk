#!/usr/bin/env bash
# ShopTalk on one EC2 GPU instance (g4dn.xlarge, "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)").
# Run from the cloned repo on the instance:
#
#   ./deploy/ec2.sh setup "<presigned S3 URL of shoptalk_artifacts.zip>"   # once: models + index, build image
#   ./deploy/ec2.sh start      # run the container (first start downloads ~8 GB of models: 5–10 min)
#   ./deploy/ec2.sh smoke      # end-to-end checks + a 20-request latency sample
#   ./deploy/ec2.sh status     # container, GPU memory, /health
#   ./deploy/ec2.sh logs       # follow the container output (Ctrl+C to leave)
#   ./deploy/ec2.sh tunnel     # optional public https link (Cloudflare quick tunnel) instead of an SSH tunnel
#   ./deploy/ec2.sh save-logs  # pack requests/feedback logs into ~/shoptalk/shoptalk_logs_<time>.tgz
#   ./deploy/ec2.sh stop       # stop the container — then STOP THE INSTANCE in the AWS console
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${SHOPTALK_HOME:-$HOME/shoptalk}"          # artifacts, model cache and logs live here (survive rebuilds)
IMAGE="${SHOPTALK_IMAGE:-shoptalk:gpu}"
NAME=shoptalk
DOCKER=docker
docker ps >/dev/null 2>&1 || DOCKER="sudo docker"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
die() { printf '\n❌ %s\n' "$*" >&2; exit 1; }

preflight() {
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found — use the 'Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)'"
  nvidia-smi -L || die "no GPU visible — is this a g4dn instance?"
  $DOCKER info >/dev/null 2>&1 || die "Docker is not running"
  $DOCKER info 2>/dev/null | grep -qi nvidia || echo "⚠️ NVIDIA runtime not listed by 'docker info' — '--gpus all' may still work via CDI"
}

cmd_setup() {
  local url="${1:-}"
  [[ -n "$url" ]] || die "usage: $0 setup \"<presigned URL of shoptalk_artifacts.zip>\"  (keep the quotes)"
  preflight
  mkdir -p "$BASE"/{artifacts,hf-cache,logs}
  say "1/3 Downloading models + Chroma index"
  curl -fL --retry 3 -o "$BASE/artifacts.zip" "$url" || die "download failed — the presigned URL may have expired (make a new one in S3)"
  python3 -m zipfile -e "$BASE/artifacts.zip" "$BASE/artifacts/"
  [[ -f "$BASE/artifacts/data/chroma/chroma.sqlite3" ]] || die "artifacts.zip has no data/chroma/chroma.sqlite3"
  ls -d "$BASE"/artifacts/models/*/ >/dev/null 2>&1 || die "artifacts.zip has no models/<embedding model>/"
  du -sh "$BASE/artifacts"/*
  say "2/3 Building the image (GPU build of torch, ~5–8 min the first time)"
  $DOCKER build -t "$IMAGE" "$REPO_DIR"
  say "3/3 Done. Next: $0 start"
}

cmd_start() {
  [[ -d "$BASE/artifacts/data/chroma" ]] || die "run '$0 setup <url>' first"
  $DOCKER rm -f "$NAME" >/dev/null 2>&1 || true
  # Ports are bound to 127.0.0.1 only: reach them through an SSH tunnel (or '$0 tunnel'), never an open port.
  $DOCKER run -d --name "$NAME" --gpus all --restart unless-stopped --shm-size 2g \
    -p 127.0.0.1:7860:7860 -p 127.0.0.1:8000:8000 \
    -v "$BASE/artifacts/models:/app/models:ro" \
    -v "$BASE/artifacts/data:/app/data:ro" \
    -v "$BASE/hf-cache:/cache/huggingface" \
    -v "$BASE/logs:/app/logs" \
    -e SHOPTALK_TTS="${SHOPTALK_TTS:-1}" \
    "$IMAGE" >/dev/null
  say "Container started. Models load in the background (first start downloads ~8 GB: 5–10 min)."
  echo "Watch: $0 logs     ·  then: $0 smoke"
}

cmd_smoke()  { $DOCKER exec "$NAME" python /app/docker/smoke_test.py --latency "${LATENCY_N:-20}" "$@"; }
cmd_logs()   { $DOCKER logs -f --tail 100 "$NAME"; }
cmd_status() {
  $DOCKER ps --filter "name=$NAME" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
  curl -s --max-time 5 http://127.0.0.1:8000/health && echo || echo "(API not answering yet)"
}
cmd_tunnel() {
  local exe="$BASE/cloudflared"
  [[ -x "$exe" ]] || { curl -fsSL -o "$exe" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x "$exe"; }
  say "Public https link for the UI below (Ctrl+C closes it). Anyone with the link can use ShopTalk — close it after the demo."
  "$exe" tunnel --url http://127.0.0.1:7860 --protocol http2 --no-autoupdate
}
cmd_save_logs() {
  local f
  f="$BASE/shoptalk_logs_$(date +%Y%m%d-%H%M).tgz"
  tar -czf "$f" -C "$BASE" logs && echo "saved $f  (download: scp -i <key.pem> ubuntu@<ip>:$f .)"
}
cmd_stop() {
  $DOCKER stop "$NAME" >/dev/null 2>&1 && echo "container stopped" || echo "container was not running"
  say "Now STOP the instance: AWS console ▸ EC2 ▸ Instances ▸ select ▸ Instance state ▸ Stop (GPU billing ends; the disk is kept)."
}

case "${1:-}" in
  setup) shift; cmd_setup "$@" ;;
  start) cmd_start ;;
  smoke) shift; cmd_smoke "$@" ;;
  logs) cmd_logs ;;
  status) cmd_status ;;
  tunnel) cmd_tunnel ;;
  save-logs) cmd_save_logs ;;
  stop) cmd_stop ;;
  *) sed -n '2,13p' "$0"; exit 1 ;;
esac
