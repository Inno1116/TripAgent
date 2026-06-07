"""ASGI entrypoint for KyuriAgents."""

from kyuriagents.server.app import create_app

app = create_app()
