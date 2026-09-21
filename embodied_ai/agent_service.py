"""Local HTTP control plane for launching Python providers against the Rust authority.

This service deliberately owns no simulation state. It is an optional third local
process that lets the browser start an orchestrated provider run without exposing
Gemini credentials to Next.js or letting the browser mutate the world directly.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .cli import provider_for
from .datasets import export_jsonl, export_parquet
from .experiments import ExperimentManifest
from .paths import ROOT
from .runner import RemoteRunResult, run_remote


class AgentRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    provider: Literal["scripted", "mock_reasoning", "random_valid", "cautious", "explorer", "gemini"] = "scripted"
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    model: str | None = Field(default=None, max_length=128)
    memory_mode: Literal["none", "recent"] = "recent"
    memory_window: int = Field(default=5, ge=1, le=100, strict=True)
    policy_state_mode: Literal["reset", "preserve"] = "reset"
    observation_mode: Literal["minimal", "normal", "rich", "oracle", "noisy"] = "normal"
    max_steps: int | None = Field(default=None, ge=1, le=10_000)
    max_wall_seconds: float | None = Field(default=None, gt=0, le=86_400)
    max_total_tokens: int | None = Field(default=None, ge=1, le=100_000_000)


class AgentRunManager:
    def __init__(self, simulation_url: str, output_directory: Path):
        self.simulation_url = simulation_url.rstrip("/")
        self.output_directory = output_directory
        self._runs: dict[str, dict[str, object]] = {}
        self._lock = threading.Lock()

    def start(self, request: AgentRunRequest, timeout: float = 5) -> dict[str, object]:
        created = threading.Event()
        holder: dict[str, object] = {}

        def on_created(run_id: str) -> None:
            with self._lock:
                self._runs[run_id] = {"run_id": run_id, "status": "running", "provider": request.provider}
            holder["run_id"] = run_id
            created.set()

        def worker() -> None:
            try:
                manifest = ExperimentManifest(
                    scenario_id="survival_room",
                    seed=request.seed,
                    provider=request.provider,
                    model=request.model,
                    observation_mode=request.observation_mode,
                    memory_mode=request.memory_mode,
                    memory_window=request.memory_window,
                    max_steps=request.max_steps,
                    max_wall_seconds=request.max_wall_seconds,
                    max_total_tokens=request.max_total_tokens,
                    agent_config={"launch": "agent_service", "provider": request.provider, "policy_state_mode": request.policy_state_mode},
                )
                manifest_path = manifest.persist(self.output_directory)
                result = run_remote(
                    provider_for(request.provider, request.seed, request.model),
                    request.seed,
                    self.simulation_url,
                    request.memory_mode,
                    request.max_steps,
                    request.observation_mode,
                    on_created,
                    request.max_wall_seconds,
                    request.max_total_tokens,
                    memory_window=request.memory_window,
                    experiment_id=manifest.experiment_id,
                    policy_state_mode=request.policy_state_mode,
                )
                self._finish(result, manifest_path)
            except Exception as error:  # surfaced to the local observer; no secret-bearing request data is retained
                holder["error"] = "Provider orchestration failed: " + str(error)
                run_id = holder.get("run_id")
                if isinstance(run_id, str):
                    with self._lock:
                        self._runs[run_id] = {"run_id": run_id, "status": "failed", "error": holder["error"]}
                created.set()

        threading.Thread(target=worker, name="embodied-agent-run", daemon=True).start()
        if not created.wait(timeout):
            raise TimeoutError("Timed out while creating the authoritative simulation run.")
        if "error" in holder:
            raise RuntimeError(str(holder["error"]))
        run_id = holder.get("run_id")
        if not isinstance(run_id, str):
            raise RuntimeError("The provider did not return a run identifier.")
        return self.status(run_id)

    def _finish(self, result: RemoteRunResult, manifest_path: Path | None = None) -> None:
        exports: dict[str, str] = {}
        if manifest_path is not None:
            exports["experiment_manifest"] = str(manifest_path)
        if result.records:
            self.output_directory.mkdir(parents=True, exist_ok=True)
            jsonl = export_jsonl(result.records, self.output_directory / f"{result.run_id}.trajectory.jsonl")
            export_parquet(
                [
                    {
                        **record,
                        "observation": json.dumps(record["observation"]),
                        "next_observation": json.dumps(record.get("next_observation")),
                        "agent_context": json.dumps(record["agent_context"]),
                        "agent_metadata": json.dumps(record.get("agent_metadata")),
                        "events": json.dumps(record["events"]),
                        "chosen_action": json.dumps(record["chosen_action"]),
                        "metrics": json.dumps(record["metrics"]),
                    }
                    for record in result.records
                ],
                self.output_directory / f"{result.run_id}.parquet",
            )
            exports.update({"jsonl": str(jsonl), "parquet": str(self.output_directory / f"{result.run_id}.parquet")})
        with self._lock:
            self._runs[result.run_id] = {
                "run_id": result.run_id,
                "status": "completed",
                "terminal_reason": result.terminal_reason,
                "steps": result.steps,
                "exports": exports,
            }

    def status(self, run_id: str) -> dict[str, object]:
        with self._lock:
            status = self._runs.get(run_id)
            if status is None:
                raise KeyError(run_id)
            return dict(status)

    def export_path(self, run_id: str, export_type: str) -> Path:
        if export_type not in {"jsonl", "parquet", "experiment_manifest"}:
            raise KeyError(export_type)
        status = self.status(run_id)
        exports = status.get("exports")
        if not isinstance(exports, dict) or not isinstance(exports.get(export_type), str):
            raise KeyError(export_type)
        root = self.output_directory.resolve()
        candidate = Path(exports[export_type]).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise KeyError(export_type)
        return candidate


def make_handler(manager: AgentRunManager):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: HTTPStatus, body: dict[str, object]) -> None:
            encoded = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            origin = self.headers.get("Origin", "")
            if origin in {"http://localhost:3000", "http://127.0.0.1:3000"}:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.end_headers()
            self.wfile.write(encoded)

        def _send_file(self, path: Path) -> None:
            content = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
            origin = self.headers.get("Origin", "")
            if origin in {"http://localhost:3000", "http://127.0.0.1:3000"}:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.end_headers()
            self.wfile.write(content)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            origin = self.headers.get("Origin", "")
            if origin in {"http://localhost:3000", "http://127.0.0.1:3000"}:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send(HTTPStatus.OK, {"status": "ok"})
                return
            prefix = "/api/agent-runs/"
            if self.path.startswith(prefix) and "/exports/" in self.path:
                run_id, export_type = self.path.removeprefix(prefix).split("/exports/", 1)
                try:
                    self._send_file(manager.export_path(run_id, export_type))
                except KeyError:
                    self._send(HTTPStatus.NOT_FOUND, {"error": "Run export not found."})
                return
            if self.path.startswith(prefix):
                try:
                    self._send(HTTPStatus.OK, manager.status(self.path.removeprefix(prefix)))
                except KeyError:
                    self._send(HTTPStatus.NOT_FOUND, {"error": "Agent run not found."})
                return
            self._send(HTTPStatus.NOT_FOUND, {"error": "Route not found."})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/agent-runs":
                self._send(HTTPStatus.NOT_FOUND, {"error": "Route not found."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 4096:
                    raise ValueError("Request body must be between 1 and 4096 bytes.")
                request = AgentRunRequest.model_validate_json(self.rfile.read(length))
                self._send(HTTPStatus.ACCEPTED, manager.start(request))
            except (ValidationError, ValueError) as error:
                self._send(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except (RuntimeError, TimeoutError) as error:
                self._send(HTTPStatus.BAD_GATEWAY, {"error": str(error)})

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local AI providers for the Embodied Worlds observer.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("AGENT_SERVICE_PORT", "8090")))
    parser.add_argument("--server-url", default=os.environ.get("SIM_SERVER_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "runs")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(AgentRunManager(args.server_url, args.output)))
    print(f"Agent control service listening on http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
