"""Run the KyuriAgents FastAPI server."""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8000


def main() -> None:
    """Start the API server with `uvicorn`."""
    _load_runtime_env()
    try:
        import uvicorn  # noqa: PLC0415

        from kyuriagents.server.app import create_app  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install `kyuriagents-backend[api]` to run the API server."
        raise ImportError(msg) from exc

    host = os.environ.get("KYURIAGENTS_API_HOST") or os.environ.get("DEEPAGENTS_API_HOST", _DEFAULT_HOST)
    port = int(os.environ.get("KYURIAGENTS_API_PORT") or os.environ.get("DEEPAGENTS_API_PORT", str(_DEFAULT_PORT)))
    uvicorn.run(create_app(), host=host, port=port)


def _load_runtime_env() -> None:
    root = Path(__file__).resolve().parents[1]
    for env_path in (root / "runtime.env", root / "kyuriagents" / "runtime" / "runtime.env"):
        if env_path.exists():
            _load_env_file(env_path)


def _load_env_file(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip())


if __name__ == "__main__":
    main()
