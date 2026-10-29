# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/deferred.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Deferred verify phase (pipeline.verify_mode = "deferred"): a
                caching backend records verify-role requests (zoom / verify /
                IBAN choice with models.vision) into a persistent JSONL queue
                instead of calling them; all other answers are cached. After
                the batch, queued requests run grouped by model (one model
                load each, bounded concurrency) and pending documents are
                replayed through the unchanged pipeline with cached answers -
                the same decision code as inline mode. New cache misses are
                queued for the next round until a fixpoint is reached.
------------------------------------------------------------------------------
"""

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fluxsense.config import Settings
from fluxsense.llm.base import CallCollector, LLMBackend, LLMRequest
from fluxsense.llm.images import ImagePart
from fluxsense.log import get_logger
from fluxsense.models.result import CallRecord

logger = get_logger("deferred")

RECORD: str = "record"
REPLAY: str = "replay"
VERIFY_LABEL_PREFIXES: Tuple[str, ...] = ("Zoom", "Review (vision)")


def request_to_json(request: LLMRequest) -> Dict[str, Any]:
    """Serializable form of a request (images as data URIs)."""
    data = asdict(request)
    data["images"] = [asdict(img) for img in request.images]
    return data


def request_from_json(data: Dict[str, Any]) -> LLMRequest:
    """Rebuilds a request from its serialized form."""
    data = dict(data)
    data["images"] = [ImagePart(**img) for img in data.get("images") or []]
    return LLMRequest(**data)


def fingerprint(request: LLMRequest) -> str:
    """Stable hash of everything that determines an answer (not the label's attempt counters)."""
    digest = hashlib.sha256()
    payload = {"model": request.model, "system": request.system, "prompt": request.prompt, "followups": request.followups,
               "schema": request.schema, "max_tokens": request.max_tokens, "verbatim": request.verbatim,
               "sampling": request.sampling, "images": [hashlib.sha256(i.data_uri.encode()).hexdigest() for i in request.images]}
    digest.update(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


class DeferredStore:
    """Persistent answer cache and request queue (JSONL files)."""

    def __init__(self, directory: Optional[Path] = None) -> None:
        """
        Args:
            directory: Folder for answers.jsonl / queue.jsonl (None = in memory only).
        """
        self.directory = directory
        self.answers: Dict[str, Dict[str, Any]] = {}
        self.queue: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            for name, target in (("answers.jsonl", self.answers), ("queue.jsonl", self.queue)):
                path = directory / name
                if path.exists():
                    for line in path.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            entry = json.loads(line)
                            target[entry["fp"]] = entry

    def _append(self, name: str, entry: Dict[str, Any]) -> None:
        """Appends one JSONL entry (if persistent)."""
        if self.directory is not None:
            with (self.directory / name).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def answer(self, fp: str) -> Optional[Dict[str, Any]]:
        """Cached answer entry for a fingerprint."""
        with self._lock:
            return self.answers.get(fp)

    def store_answer(self, fp: str, answer: Any, record: Optional[CallRecord]) -> None:
        """Caches an answer (with the call statistics)."""
        entry = {"fp": fp, "answer": answer, "record": record.model_dump() if record else None}
        with self._lock:
            self.answers[fp] = entry
            self._append("answers.jsonl", entry)

    def enqueue(self, fp: str, request: LLMRequest, doc: Optional[str]) -> None:
        """Queues a request (once per fingerprint)."""
        with self._lock:
            if fp in self.queue or fp in self.answers:
                return
            entry = {"fp": fp, "doc": doc, "model": request.model, "label": request.label, "request": request_to_json(request)}
            self.queue[fp] = entry
            self._append("queue.jsonl", entry)

    def open_requests(self) -> List[Dict[str, Any]]:
        """Queued requests without an answer yet."""
        with self._lock:
            return [e for fp, e in self.queue.items() if fp not in self.answers]


class CachingBackend(CallCollector):
    """
    Backend wrapper for deferred verification. RECORD: verify-role requests are
    queued (answer None), everything else is called and cached. REPLAY: every
    request is served from the cache; misses are queued.
    """

    def __init__(self, inner: LLMBackend, store: DeferredStore, settings: Settings, mode: str) -> None:
        """
        Args:
            inner: Real backend.
            store: Answer cache / queue.
            settings: Settings (model roles).
            mode: RECORD or REPLAY.
        """
        super().__init__()
        self.inner = inner
        self.store = store
        self.settings = settings
        self.mode = mode
        self._doc = threading.local()
        self.fingerprints: Dict[str, str] = getattr(inner, "fingerprints", {})

    def set_doc(self, doc: Optional[str]) -> None:
        """Sets the document the current thread works on (for queue entries)."""
        self._doc.name = doc
        self._doc.pending = False

    @property
    def pending(self) -> bool:
        """True if the current thread queued a request since set_doc()."""
        return bool(getattr(self._doc, "pending", False))

    def is_verify(self, request: LLMRequest) -> bool:
        """Verify-role request (vision model, zoom/verify/choice/vision review)."""
        return request.model == self.settings.models.vision and request.label.startswith(VERIFY_LABEL_PREFIXES)

    def list_models(self) -> List[str]:
        """Delegates to the real backend."""
        return self.inner.list_models()

    def generate_json(self, request: LLMRequest) -> Optional[Any]:
        """Cache / queue / call according to the mode."""
        fp = fingerprint(request)
        cached = self.store.answer(fp)
        if cached is not None:
            if cached.get("record"):
                self.record(CallRecord(**{**cached["record"], "label": f"{cached['record']['label']} (cached)", "seconds": 0.0}))
            return cached["answer"]
        if self.mode == REPLAY or self.is_verify(request):
            self.store.enqueue(fp, request, getattr(self._doc, "name", None))
            self._doc.pending = True
            return None
        sink: List[CallRecord] = []
        collect = getattr(self.inner, "collect", None)
        if collect is not None:
            with collect(sink):
                answer = self.inner.generate_json(request)
        else:
            answer = self.inner.generate_json(request)
        record = sink[-1] if sink else None
        if record is not None:
            self.record(record)
        self.store.store_answer(fp, answer, record)
        return answer


def run_queue(store: DeferredStore, backend: LLMBackend, concurrency: int) -> Tuple[int, float]:
    """
    Runs all open queued requests grouped by model (each model loaded once).

    Args:
        store: Queue / cache.
        backend: Real backend.
        concurrency: Parallel requests.

    Returns:
        Tuple (requests run, wall seconds).
    """
    start = time.monotonic()
    open_requests = store.open_requests()
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for entry in open_requests:
        by_model.setdefault(entry["model"], []).append(entry)
    for model, entries in by_model.items():
        logger.info(f"deferred phase: {len(entries)} requests for {model}")

        def run(entry: Dict[str, Any]) -> None:
            request = request_from_json(entry["request"])
            sink: List[CallRecord] = []
            collect = getattr(backend, "collect", None)
            if collect is not None:
                with collect(sink):
                    answer = backend.generate_json(request)
            else:
                answer = backend.generate_json(request)
            store.store_answer(entry["fp"], answer, sink[-1] if sink else None)

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            list(pool.map(run, entries))
    return len(open_requests), round(time.monotonic() - start, 3)
