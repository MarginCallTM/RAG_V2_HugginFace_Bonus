"""Serve the local RAG API with Python's standard HTTP server."""

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

from pydantic import ValidationError

from src.api_service import ApiConfig, ApiError, QueryRequest, RagService


MAX_BODY_BYTES = 65536
LOGGER = logging.getLogger(__name__)


class RagHTTPServer(ThreadingHTTPServer):
    """Share one service between HTTP handlers."""

    daemon_threads = True

    def __init__(
        self, address: tuple[str, int], service: RagService,
    ) -> None:
        """Bind the listening socket and retain the service."""
        self.service = service
        super().__init__(address, RagHandler)


class RagHandler(BaseHTTPRequestHandler):
    """Translate local HTTP requests into validated RAG operations."""

    def setup(self) -> None:
        """Bound how long a connection may wait for request bytes."""
        self.request.settimeout(15)
        super().setup()

    def _send_json(self, status: int, payload: object) -> None:
        """Send a complete UTF-8 JSON response and close the connection."""
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        if status == 503:
            self.send_header("Retry-After", "1")
        if status == 405:
            route = urlsplit(self.path).path
            self.send_header("Allow", "GET" if route == "/health" else "POST")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(body)

    def send_error(
        self, code: int, message: str | None = None,
        explain: str | None = None,
    ) -> None:
        """Use JSON for protocol errors rather than the default HTML page."""
        self._send_json(code, {"error": message or "HTTP request rejected"})

    def do_GET(self) -> None:
        """Report readiness without loading the generation model."""
        route = urlsplit(self.path).path
        if route == "/health":
            service = cast(RagHTTPServer, self.server).service
            self._send_json(200, {
                "status": "ok",
                "method": service.config.method,
                "device": "cpu",
                "generator_loaded": service.generator is not None,
            })
        elif route in ("/search", "/answer"):
            self._send_json(405, {"error": "Use POST with a JSON body"})
        else:
            self._send_json(404, {"error": "Route not found"})

    def _read_query(self) -> QueryRequest:
        """Validate framing, body size, UTF-8 JSON and request fields."""
        if self.headers.get("Transfer-Encoding") is not None:
            raise ApiError(400, "Transfer-Encoding is not supported")
        lengths = self.headers.get_all("Content-Length", [])
        if not lengths:
            raise ApiError(411, "Content-Length is required")
        if len(lengths) != 1:
            raise ApiError(400, "Use one Content-Length header")
        try:
            length = int(lengths[0])
        except ValueError as error:
            raise ApiError(400, "Invalid Content-Length") from error
        if length < 0:
            raise ApiError(400, "Invalid Content-Length")
        if length > MAX_BODY_BYTES:
            raise ApiError(413, "JSON body exceeds 65536 bytes")
        content_type = self.headers.get("Content-Type", "")
        if content_type.split(";", 1)[0].strip().lower() != "application/json":
            raise ApiError(415, "Content-Type must be application/json")
        body = self.rfile.read(length)
        if len(body) != length:
            raise ApiError(400, "Incomplete request body")
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            raise ApiError(400, "Body must be valid UTF-8 JSON") from error
        try:
            return QueryRequest.model_validate(payload)
        except ValidationError as error:
            raise ApiError(
                422, "Expected a nonblank question (<=16000 characters) "
                "and integer k between 0 and 20; no extra fields",
            ) from error

    def do_POST(self) -> None:
        """Search the loaded index or generate a sourced answer."""
        route = urlsplit(self.path).path
        if route not in ("/search", "/answer"):
            status = 405 if route == "/health" else 404
            self._send_json(status, {"error": "Route does not accept POST"})
            return
        try:
            query = self._read_query()
            service = cast(RagHTTPServer, self.server).service
            response = service.execute(route, query)
            self._send_json(200, response.model_dump(mode="json"))
        except ApiError as error:
            self._send_json(error.status, {"error": str(error)})
        except TimeoutError:
            self._send_json(408, {"error": "Request body timed out"})
        except (BrokenPipeError, ConnectionResetError):
            LOGGER.info("Client disconnected before the response completed")
        except Exception:
            LOGGER.exception("RAG request failed")
            self._send_json(500, {"error": "RAG processing failed"})


def main() -> None:
    """Load an index snapshot and start the local CPU HTTP API."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--method", choices=("bm25", "semantic", "hybrid"), default="bm25",
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=Path("data/processed"),
    )
    parser.add_argument(
        "--semantic-index-path", type=Path,
        default=Path("data/processed/semantic_index.pt"),
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--context-budget", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")
    try:
        config = ApiConfig(
            method=args.method, processed_dir=args.processed_dir,
            semantic_index_path=args.semantic_index_path,
            model_name=args.model_name, context_budget=args.context_budget,
            max_new_tokens=args.max_new_tokens,
        )
        service = RagService(config)
        server = RagHTTPServer((args.host, args.port), service)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f"API startup failed: {error}\n")
    with server:
        print(
            f"RAG API: http://{args.host}:{server.server_port} "
            f"({config.method}, CPU); Ctrl+C to stop",
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping RAG API", flush=True)


if __name__ == "__main__":
    main()
