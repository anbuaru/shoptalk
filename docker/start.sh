#!/usr/bin/env bash
# Starts the ShopTalk API (port 8000) and UI (port 7860) in one container.
# If either process exits, the container exits too, so Docker's restart policy can bring it back.
set -euo pipefail
cd "${APP_DIR:-/app}"

python -m uvicorn src.api.main:app --host 0.0.0.0 --port "${SHOPTALK_API_PORT:-8000}" --log-level warning &
API_PID=$!
python -m src.ui.app &
UI_PID=$!

stop() { kill -TERM "$API_PID" "$UI_PID" 2>/dev/null || true; wait || true; }
trap stop TERM INT

echo "[start.sh] API pid $API_PID on :${SHOPTALK_API_PORT:-8000} · UI pid $UI_PID on :${SHOPTALK_UI_PORT:-7860}"
echo "[start.sh] models load in the background; /health reports 'loading' until they are ready (first start downloads ~8 GB)"
set +e
wait -n "$API_PID" "$UI_PID"
code=$?
echo "[start.sh] a process exited with code $code — stopping the container"
stop
exit "$code"
