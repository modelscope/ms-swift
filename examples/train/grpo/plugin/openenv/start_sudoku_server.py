#!/usr/bin/env python3
"""Start the TextArena Sudoku server with configurable concurrent sessions.

The default OpenEnv server only allows 1 concurrent session because
TextArenaEnvironment is not marked as SUPPORTS_CONCURRENT_SESSIONS.
Since each WebSocket session creates an independent game instance,
it is safe to enable concurrent sessions.

Usage:
    TEXTARENA_ENV_ID=Sudoku-v0 python start_sudoku_server.py
    TEXTARENA_ENV_ID=Sudoku-v0 MAX_CONCURRENT_ENVS=8 python start_sudoku_server.py
"""
import os
import uvicorn

# Read config from environment
env_id = os.getenv("TEXTARENA_ENV_ID", "Sudoku-v0")
max_concurrent_envs = int(os.getenv("MAX_CONCURRENT_ENVS", "8"))
host = os.getenv("HOST", "0.0.0.0")
port = int(os.getenv("PORT", "8000"))

# Mark TextArenaEnvironment as supporting concurrent sessions.
# Each WebSocket session creates an independent game instance via the factory,
# so concurrent sessions are safe.
from textarena_env.server.environment import TextArenaEnvironment
TextArenaEnvironment.SUPPORTS_CONCURRENT_SESSIONS = True

# Build the app with custom max_concurrent_envs
from textarena_env.server.app import (
    create_textarena_environment,
    TextArenaAction,
    TextArenaObservation,
    build_textarena_gradio_app,
)
from openenv.core.env_server.http_server import create_app

app = create_app(
    create_textarena_environment,
    TextArenaAction,
    TextArenaObservation,
    env_name="textarena_env",
    max_concurrent_envs=max_concurrent_envs,
    gradio_builder=build_textarena_gradio_app,
)

if __name__ == "__main__":
    print(f"Starting Sudoku server: env={env_id}, max_concurrent_envs={max_concurrent_envs}")
    uvicorn.run(app, host=host, port=port)
