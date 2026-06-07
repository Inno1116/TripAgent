#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ACTION="${1:-up}"
COMPOSE_FILE="${KYURIAGENTS_COMPOSE_FILE:-docker-compose.full.yml}"
ENV_FILE="${KYURIAGENTS_ENV_FILE:-runtime.env}"

compose() {
  docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" "$@"
}

ensure_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is not installed or not in PATH." >&2
    exit 1
  fi
  if ! docker compose version >/dev/null 2>&1; then
    echo "Docker Compose v2 is required." >&2
    exit 1
  fi
}

ensure_env() {
  if [[ ! -f "$ENV_FILE" ]]; then
    cp runtime.env.example "$ENV_FILE"
    echo "Created $ENV_FILE from runtime.env.example."
    echo "Edit $ENV_FILE first, then run: bash deploy.sh up"
    exit 1
  fi
}

require_configured_env() {
  local missing=0
  if grep -Eq '^DASHSCOPE_API_KEY=(replace-me)?$' "$ENV_FILE"; then
    echo "Please set DASHSCOPE_API_KEY in $ENV_FILE." >&2
    missing=1
  fi
  if grep -Eq '^POSTGRES_PASSWORD=(change-this-postgres-password|change-me)?$' "$ENV_FILE"; then
    echo "Please set a strong POSTGRES_PASSWORD in $ENV_FILE." >&2
    missing=1
  fi
  if ! grep -Eq '^(KYURIAGENTS_API_ADMIN_KEY|DEEPAGENTS_API_ADMIN_KEY)=.+' "$ENV_FILE" \
    || grep -Eq '^(KYURIAGENTS_API_ADMIN_KEY|DEEPAGENTS_API_ADMIN_KEY)=(replace-this-admin-key)?$' "$ENV_FILE"; then
    echo "Please set KYURIAGENTS_API_ADMIN_KEY in $ENV_FILE." >&2
    missing=1
  fi
  if [[ "$missing" -ne 0 ]]; then
    exit 1
  fi
}

warn_linux_limits() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    return
  fi
  local current
  current="$(sysctl -n vm.max_map_count 2>/dev/null || echo 0)"
  if [[ "$current" -lt 262144 ]]; then
    echo "Warning: Elasticsearch usually needs vm.max_map_count=262144."
    echo "Run: sudo sysctl -w vm.max_map_count=262144"
  fi
}

ensure_docker

case "$ACTION" in
  init)
    if [[ -f "$ENV_FILE" ]]; then
      echo "$ENV_FILE already exists."
    else
      cp runtime.env.example "$ENV_FILE"
      echo "Created $ENV_FILE. Edit it before deployment."
    fi
    ;;
  up | start)
    ensure_env
    require_configured_env
    warn_linux_limits
    compose up -d --build
    compose ps
    ;;
  update)
    ensure_env
    require_configured_env
    compose up -d --build bootstrap api worker web
    compose ps
    ;;
  bootstrap)
    ensure_env
    require_configured_env
    compose up --build bootstrap
    ;;
  ps)
    ensure_env
    compose ps
    ;;
  logs)
    ensure_env
    service="${2:-api}"
    compose logs -f "$service"
    ;;
  down | stop)
    ensure_env
    compose down
    ;;
  reset)
    ensure_env
    if [[ "${KYURIAGENTS_CONFIRM_RESET:-}" != "yes" ]]; then
      echo "This deletes Docker volumes. Re-run with KYURIAGENTS_CONFIRM_RESET=yes bash deploy.sh reset" >&2
      exit 1
    fi
    compose down -v
    ;;
  *)
    echo "Usage: bash deploy.sh [init|up|update|bootstrap|ps|logs SERVICE|down|reset]" >&2
    exit 1
    ;;
esac
