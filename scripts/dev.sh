#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ML_PORT="${ML_PORT:-8000}"
BACKEND_PORT="${PORT:-4000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
WITH_FRONTEND="${WITH_FRONTEND:-0}"
DATABASE_URL="${DATABASE_URL:-postgresql://adaptive_learning:adaptive_learning@127.0.0.1:54329/adaptive_learning?schema=public}"
ALP_POLICY="${ALP_POLICY:-rule_improved}"
ML_PYTHON="$ROOT_DIR/ml-service/venv/bin/python"

if [[ ! -x "$ML_PYTHON" ]]; then
    ML_PYTHON="python3"
fi

# Bash 3 treats an empty array as unset under `set -u`; the sentinel keeps
# cleanup safe before any application process has been started.
pids=("")
cleanup_done=0

cleanup() {
    if [[ "$cleanup_done" == "1" ]]; then
        return
    fi

    cleanup_done=1

    echo
    echo "Stopping dev services..."

    for pid in "${pids[@]}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done

    wait 2>/dev/null || true
}

trap cleanup EXIT INT TERM

require_port_free() {
    local port="$1"
    local name="$2"
    local env_name="$3"

    if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "Port $port is already in use for $name."
        echo "Stop that process or run with a different port, for example:"
        echo "  $env_name=<port> ./scripts/dev.sh"
        exit 1
    fi
}

start_service() {
    local name="$1"
    local workdir="$2"
    shift 2

    echo "Starting $name..."
    (
        cd "$workdir"
        "$@"
    ) &
    pids+=("$!")
}

require_port_free "$ML_PORT" "ML service" "ML_PORT"
require_port_free "$BACKEND_PORT" "backend" "PORT"

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required to start the local PostgreSQL database."
    exit 1
fi

echo "Starting local PostgreSQL..."
docker compose -f "$ROOT_DIR/docker-compose.yml" up -d postgres

echo "Waiting for PostgreSQL..."
for _ in {1..30}; do
    if docker compose -f "$ROOT_DIR/docker-compose.yml" exec -T postgres pg_isready -U adaptive_learning -d adaptive_learning >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

if ! docker compose -f "$ROOT_DIR/docker-compose.yml" exec -T postgres pg_isready -U adaptive_learning -d adaptive_learning >/dev/null 2>&1; then
    echo "PostgreSQL did not become ready. Check: docker compose -f docker-compose.yml logs postgres"
    exit 1
fi

if [[ ! -d "$ROOT_DIR/backend/node_modules" ]]; then
    echo "Installing backend dependencies..."
    (cd "$ROOT_DIR/backend" && npm install)
fi

echo "Applying database schema..."
(
    cd "$ROOT_DIR/backend"
    DATABASE_URL="$DATABASE_URL" npx prisma db push
    DATABASE_URL="$DATABASE_URL" npx prisma generate
)

if [[ "$WITH_FRONTEND" == "1" ]]; then
    require_port_free "$FRONTEND_PORT" "frontend" "FRONTEND_PORT"
fi

start_service "ML service on http://localhost:$ML_PORT" \
    "$ROOT_DIR/ml-service" \
    "$ML_PYTHON" -m uvicorn app.main:app --host 127.0.0.1 --port "$ML_PORT"

start_service "backend on http://localhost:$BACKEND_PORT" \
    "$ROOT_DIR/backend" \
    env PORT="$BACKEND_PORT" DATABASE_URL="$DATABASE_URL" ALP_POLICY="$ALP_POLICY" ML_SERVICE_URL="http://127.0.0.1:$ML_PORT" node server.js

if [[ "$WITH_FRONTEND" == "1" ]]; then
    start_service "frontend on http://localhost:$FRONTEND_PORT" \
        "$ROOT_DIR/files" \
        python3 -m http.server "$FRONTEND_PORT"
fi

echo
echo "Ready."
echo "App:     http://localhost:$BACKEND_PORT/index.html"
echo "API:     http://localhost:$BACKEND_PORT"
echo "ML:      http://localhost:$ML_PORT"
echo "Policy:  $ALP_POLICY"

if [[ "$WITH_FRONTEND" == "1" ]]; then
    echo "Frontend: http://localhost:$FRONTEND_PORT"
fi

echo
echo "Press Ctrl-C to stop everything."

wait
