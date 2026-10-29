# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/wol.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Optional energy management of a suspended AI server through a
                WOL-Server 'device requests' API. Disabled unless
                hooks.wol_url is set (then nothing here runs: no requests,
                threads or log lines). A thread-safe lease per (server,
                device) is shared by the library, the CLI and the REST
                server: acquire on the first LLM call (wakes the device),
                heartbeat every ttl/3 while jobs run, release after an idle
                period and at process exit. The device's own idle logic
                decides about sleep (no 'then=sleep').
------------------------------------------------------------------------------
"""

import atexit
import shlex
import socket
import subprocess
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Union

import httpx

from fluxsense.log import get_logger

logger = get_logger("wol")


class WolError(Exception):
    """The WOL-Server could not be used (unreachable, authentication, no answer)."""


class WolLease:
    """One open request on a WOL-Server device, kept alive while FluxSense works."""

    def __init__(self, url: str, device: str, token_command: str = "", ttl_s: int = 900, release_after_idle_s: float = 600.0,
                 verify: Union[bool, str] = True, client: str = "", transport: Optional[httpx.BaseTransport] = None) -> None:
        """
        Args:
            url: WOL-Server base URL (e.g. https://wol.example.lan).
            device: Device id or name.
            token_command: Command printing the API token (the token is never logged).
            ttl_s: Request lifetime without heartbeat.
            release_after_idle_s: Release the request after this long without jobs.
            verify: TLS verification (bool, or a CA bundle path for a self-signed server).
            client: Client name shown on the WOL-Server (default fluxsense@<hostname>).
            transport: httpx transport (tests).
        """
        self.url, self.device = url.rstrip("/"), device
        self.token_command, self.ttl_s, self.idle_s = token_command, max(60, int(ttl_s)), float(release_after_idle_s)
        self.client = client or f"fluxsense@{socket.gethostname()}"
        self._verify, self._transport = verify, transport
        self._lock = threading.RLock()
        self._id: Optional[str] = None
        self._jobs = 0
        self._stop = threading.Event()
        self._heartbeat: Optional[threading.Thread] = None
        self._idle_timer: Optional[threading.Timer] = None
        self._token: Optional[str] = None

    # ----------------------------------------------------------------- http

    def _http(self) -> httpx.Client:
        """HTTP client with the bearer token."""
        if self._token is None:
            self._token = self._read_token()
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        return httpx.Client(base_url=f"{self.url}/api/devices/{self.device}", headers=headers, verify=self._verify,
                            timeout=httpx.Timeout(330.0, connect=10.0), transport=self._transport)

    def _read_token(self) -> str:
        """Runs the token command (output is kept in memory only)."""
        if not self.token_command.strip():
            return ""
        try:
            done = subprocess.run(shlex.split(self.token_command), check=False, timeout=30, capture_output=True, text=True)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise WolError(f"WOL token command failed: {e.__class__.__name__}") from e
        if done.returncode != 0 or not done.stdout.strip():
            raise WolError("WOL token command returned no token")
        return done.stdout.strip()

    # ---------------------------------------------------------------- lease

    @property
    def request_id(self) -> Optional[str]:
        """Id of the open request (None if none)."""
        return self._id

    def acquire(self, wait_s: int = 300) -> bool:
        """
        Opens a request (wakes the device if needed) unless one is open.

        Returns:
            True if the device is reported online.

        Raises:
            WolError: The WOL-Server is unreachable, rejects the token or the device did not come up.
        """
        with self._lock:
            if self._id:
                return True
            try:
                with self._http() as http:
                    resp = http.post(f"/requests?wait={int(wait_s)}", json={"client": self.client, "ttl": self.ttl_s})
            except httpx.HTTPError as e:
                raise WolError(f"WOL-Server {self.url} unreachable: {e.__class__.__name__}") from e
            if resp.status_code in (401, 403):
                raise WolError(f"WOL-Server {self.url} rejected the API token (HTTP {resp.status_code})")
            body = _json(resp)
            if resp.status_code in (502, 504):
                if body.get("id"):  # the request stays open on 502/504: release it
                    self._delete(str(body["id"]))
                raise WolError(f"WOL-Server could not bring '{self.device}' online (HTTP {resp.status_code})")
            if resp.status_code != 201 or not body.get("id"):
                raise WolError(f"WOL-Server answered HTTP {resp.status_code} for '{self.device}'")
            self._id = str(body["id"])
            self._start_heartbeat()
            logger.info(f"WOL: requested '{self.device}' (online={body.get('online')})")
            return bool(body.get("online"))

    def heartbeat(self) -> bool:
        """Renews the request; a 404 (expired) opens a new one. Returns True if a request is open afterwards."""
        with self._lock:
            if not self._id:
                return False
            try:
                with self._http() as http:
                    resp = http.put(f"/requests/{self._id}")
            except httpx.HTTPError as e:
                logger.warning(f"WOL heartbeat failed: {e.__class__.__name__}")
                return True
            if resp.status_code == 404:
                self._id = None
                try:
                    self.acquire()
                except WolError as e:
                    logger.warning(f"WOL re-request failed: {e}")
                    return False
            return self._id is not None

    def release(self) -> None:
        """Ends the open request (never asks for sleep)."""
        with self._lock:
            self._stop.set()
            if self._idle_timer is not None:
                self._idle_timer.cancel()
                self._idle_timer = None
            if self._id:
                self._delete(self._id)
                logger.info(f"WOL: released '{self.device}'")
                self._id = None

    def reacquire(self) -> bool:
        """Backend gone mid-run: drop the request and request the device again (once per call site)."""
        with self._lock:
            if self._id:
                self._delete(self._id)
                self._id = None
            return self.acquire()

    def _delete(self, request_id: str) -> None:
        """DELETE one request (best effort)."""
        try:
            with self._http() as http:
                http.delete(f"/requests/{request_id}")
        except (httpx.HTTPError, WolError) as e:
            logger.warning(f"WOL release failed: {e.__class__.__name__}")

    def status(self) -> Dict[str, Any]:
        """The device's open requests as the WOL-Server lists them."""
        with self._http() as http:
            resp = http.get("/requests")
        resp.raise_for_status()
        return _json(resp)

    # ----------------------------------------------------------------- jobs

    @contextmanager
    def job(self) -> Iterator[None]:
        """Marks a running analysis: keeps the request alive; the release timer starts when the last job ends."""
        with self._lock:
            self._jobs += 1
            if self._idle_timer is not None:
                self._idle_timer.cancel()
                self._idle_timer = None
        try:
            yield
        finally:
            with self._lock:
                self._jobs -= 1
                if self._jobs == 0 and self._id:
                    self._idle_timer = threading.Timer(self.idle_s, self._release_if_idle)
                    self._idle_timer.daemon = True
                    self._idle_timer.start()

    def _release_if_idle(self) -> None:
        """Idle timer callback."""
        with self._lock:
            if self._jobs == 0:
                self.release()

    def _start_heartbeat(self) -> None:
        """Heartbeat thread (every ttl/3)."""
        self._stop.clear()
        if self._heartbeat is not None and self._heartbeat.is_alive():
            return
        interval = self.ttl_s / 3.0

        def loop() -> None:
            while not self._stop.wait(interval):
                if not self.heartbeat():
                    break

        self._heartbeat = threading.Thread(target=loop, name="fluxsense-wol-heartbeat", daemon=True)
        self._heartbeat.start()


def _json(resp: httpx.Response) -> Dict[str, Any]:
    """JSON body or {}."""
    try:
        data = resp.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


_LEASES: Dict[tuple, WolLease] = {}
_REGISTRY_LOCK = threading.Lock()


def lease_for(hooks: Any) -> Optional[WolLease]:
    """
    The shared lease of the configured WOL-Server device, or None when WOL
    is not configured (hooks.wol_url empty: zero overhead).
    """
    url = str(getattr(hooks, "wol_url", "") or "").strip()
    if not url:
        return None
    key = (url.rstrip("/"), hooks.wol_device)
    with _REGISTRY_LOCK:
        lease = _LEASES.get(key)
        if lease is None:
            lease = WolLease(url, hooks.wol_device, hooks.wol_token_command, hooks.wol_ttl_s, hooks.wol_release_after_idle_s,
                             hooks.wol_verify_tls, hooks.wol_client)
            _LEASES[key] = lease
            atexit.register(lease.release)
        return lease


def wait_for(probe: Any, timeout_s: float, interval_s: float = 5.0) -> bool:
    """Polls a probe until it returns True or the timeout passes."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if probe():
            return True
        time.sleep(interval_s)
    return False
