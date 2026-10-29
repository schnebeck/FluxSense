# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_wol.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Optional WOL-Server device requests against a fake WOL
                server (stdlib http.server): acquire, heartbeat, expiry
                re-request, 502 release, auth errors, idle release, and
                zero overhead when not configured.
------------------------------------------------------------------------------
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterator, List

import pytest

from fluxsense.config import HookSettings
from fluxsense.wol import WolError, WolLease, lease_for


class FakeWol:
    """In-memory WOL-Server: one device, open requests by id."""

    def __init__(self) -> None:
        self.requests: Dict[str, Dict[str, Any]] = {}
        self.log: List[str] = []
        self.post_status = 201
        self.token = "secret-token"
        self.counter = 0


@pytest.fixture
def wol() -> Iterator[Any]:
    """A running fake WOL server; yields (state, base url)."""
    state = FakeWol()

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, body: Dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _auth(self) -> bool:
            if self.headers.get("Authorization") != f"Bearer {state.token}":
                self._reply(401, {"error": "unauthorized"})
                return False
            return True

        def do_POST(self) -> None:  # noqa: N802 - http.server API
            state.log.append(f"POST {self.path}")
            if not self._auth():
                return
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            state.counter += 1
            rid = f"r{state.counter}"
            state.requests[rid] = {"id": rid, "client": body.get("client"), "ttl": body.get("ttl")}
            self._reply(state.post_status, {"id": rid, "client": body.get("client"), "online": state.post_status == 201})

        def do_PUT(self) -> None:  # noqa: N802
            state.log.append(f"PUT {self.path}")
            if not self._auth():
                return
            rid = self.path.rsplit("/", 1)[-1]
            self._reply(200 if rid in state.requests else 404, {"id": rid})

        def do_DELETE(self) -> None:  # noqa: N802
            state.log.append(f"DELETE {self.path}")
            if not self._auth():
                return
            state.requests.pop(self.path.rsplit("/", 1)[-1].split("?")[0], None)
            self._reply(200, {"sleep": None})

        def do_GET(self) -> None:  # noqa: N802
            if not self._auth():
                return
            self._reply(200, {"requests": list(state.requests.values())})

        def log_message(self, *args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield state, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _lease(url: str, **kw: Any) -> WolLease:
    return WolLease(url, "ai-server", token_command="echo secret-token", client="fluxsense@test", **kw)


def test_acquire_heartbeat_release(wol: Any) -> None:
    """A request is opened with client and ttl, renewed, re-requested after expiry and released (never then=sleep)."""
    state, url = wol
    lease = _lease(url, ttl_s=900)
    assert lease.acquire() and lease.request_id == "r1"
    assert state.requests["r1"] == {"id": "r1", "client": "fluxsense@test", "ttl": 900}
    assert any(e.startswith("POST /api/devices/ai-server/requests?wait=300") for e in state.log)
    assert lease.heartbeat() and lease.request_id == "r1"
    state.requests.clear()  # expired on the server
    assert lease.heartbeat() and lease.request_id == "r2"
    assert lease.status()["requests"][0]["id"] == "r2"
    lease.release()
    assert state.requests == {} and lease.request_id is None
    assert not any("then=sleep" in e for e in state.log)


def test_failed_wake_releases_and_auth_errors_are_clear(wol: Any) -> None:
    """502/504 leave the request open on the server: it is released; a wrong token gives a clear error."""
    state, url = wol
    state.post_status = 502
    with pytest.raises(WolError, match="could not bring"):
        _lease(url).acquire()
    assert state.requests == {} and any(e.startswith("DELETE") for e in state.log)
    state.post_status, state.token = 201, "other"
    with pytest.raises(WolError, match="rejected the API token"):
        _lease(url).acquire()
    with pytest.raises(WolError, match="unreachable"):
        WolLease("http://127.0.0.1:9", "ai-server").acquire(wait_s=1)


def test_idle_release_after_the_last_job(wol: Any) -> None:
    """The request stays while jobs run and is released after the idle period."""
    state, url = wol
    lease = _lease(url, release_after_idle_s=0.2)
    with lease.job():
        lease.acquire()
        with lease.job():
            pass
        assert state.requests  # still a job running
    time.sleep(0.6)
    assert state.requests == {} and lease.request_id is None


def test_disabled_without_url() -> None:
    """No WOL-Server configured: no lease object, nothing runs."""
    assert lease_for(HookSettings()) is None
    threads = threading.active_count()
    assert lease_for(HookSettings(wol_url="")) is None and threading.active_count() == threads


def test_first_llm_call_requests_the_device(wol: Any) -> None:
    """With a WOL-Server configured, the first LLM call opens the device request before the health check."""
    import httpx
    from fluxsense.config import LlmSettings
    from fluxsense.llm.base import LLMRequest
    from fluxsense.llm.openai_client import OpenAICompatClient
    state, url = wol

    def llm(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "{\"ok\": true}"}, "finish_reason": "stop"}]})

    hooks = HookSettings(wol_url=url, wol_device="ai-server", wol_token_command="echo secret-token", wol_client="fluxsense@t")
    client = OpenAICompatClient(LlmSettings(base_url="http://llm.invalid/v1"), hooks, transport=httpx.MockTransport(llm))
    assert client.generate_json(LLMRequest(prompt="json", label="t", model="m")) == {"ok": True}
    assert len(state.requests) == 1 and next(iter(state.requests.values()))["client"] == "fluxsense@t"
    lease_for(hooks).release()
    assert state.requests == {}
