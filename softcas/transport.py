"""Length-framed TCP adapter exchange and deployment RPC, Windows/Linux.

Every accepted/sent application byte, including frame prefixes, controls, partial
payloads and retries, is charged. Requests restart at zero. One outgoing adapter
payload is active per sender, ordered by observed arrival time/requester ID.
"""
from __future__ import annotations
import hashlib
import hmac
import json
import os
import socket
import socketserver
import struct
import threading
import time
import uuid
from pathlib import Path
from .cache import Advertisement, ContactExchange
from .state import Package, PortSpec, canonical
from .resources import Cost, ExecutionStopped, TransferJournal


HEADER = struct.Struct("!Q")
CHUNK = 64 * 1024


class RemoteOperationFailed(RuntimeError):
    def __init__(self, response):
        super().__init__(response.get("error", "Remote operation failed"))
        self.response = response


class Wire:
    def __init__(self, sock, ledger, node, peer, request_id, scope="search", log=None):
        self.sock, self.ledger, self.node, self.peer = sock, ledger, node, peer
        self.request_id, self.scope, self.log = request_id, scope, log
        self.totals = {}
        self.operation = None

    def _charge(self, direction, size, label, elapsed):
        key = self.request_id + ":" + label + ":" + direction
        self.totals[key] = self.totals.get(key, 0) + size
        self.ledger.charge("tcp_" + label, Cost(B=size, W=elapsed), self.scope,
                           transfer_id=self.request_id + ":" + label, direction=direction, peer=self.peer)

    def send(self, data, label):
        with self.ledger.lock:
            self._send(data, label)

    def _send(self, data, label):
        view = memoryview(data)
        while view:
            self.ledger.guard(scope=self.scope)
            remaining = max(0, int(self.ledger.remaining(self.scope).B))
            if remaining <= 0:
                raise ExecutionStopped("Byte allowance reached during transmission")
            start = time.perf_counter()
            sent = self.sock.send(view[:min(CHUNK, remaining)])
            if sent <= 0:
                raise ConnectionError("Socket closed while transmitting")
            self._charge("tx", sent, label, time.perf_counter() - start)
            view = view[sent:]

    def receive(self, size, label):
        with self.ledger.lock:
            return self._receive(size, label)

    def _receive(self, size, label):
        chunks, count = [], 0
        while count < size:
            self.ledger.guard(scope=self.scope)
            remaining = max(0, int(self.ledger.remaining(self.scope).B))
            if remaining <= 0:
                raise ExecutionStopped("Byte allowance reached during receipt")
            start = time.perf_counter()
            block = self.sock.recv(min(CHUNK, size - count, remaining))
            if not block:
                raise ConnectionError("Incomplete receipt; the next request must restart")
            self._charge("rx", len(block), label, time.perf_counter() - start)
            chunks.append(block)
            count += len(block)
        return b"".join(chunks)

    def send_frame(self, value, label="control"):
        payload = canonical(value)
        self.send(HEADER.pack(len(payload)) + payload, label)

    def frame(self, label="control", maximum=4 * 1024 * 1024):
        size = HEADER.unpack(self.receive(HEADER.size, label))[0]
        if size > maximum:
            raise ValueError("Control frame exceeds the configured bound")
        return json.loads(self.receive(size, label))

    def finish(self):
        if self.log is not None:
            for key, size in self.totals.items():
                self.log.append({"request": self.request_id, "leg": key, "node": self.node,
                                 "peer": self.peer, "bytes": size, "time": time.time(), "operation": self.operation})

    def reidentify(self, request_id, peer):
        old = self.request_id
        if self.log is not None:
            self.log.append({"kind": "request_identity", "receipt_id": old, "request": request_id,
                             "peer": peer, "node": self.node, "time": time.time()})
        self.request_id, self.peer = request_id, peer
        self.totals = {request_id + k[len(old):]: v for k, v in self.totals.items()}
        with self.ledger.lock:
            for event in self.ledger.events:
                identifier = event.get("transfer_id", "")
                if identifier.startswith(old + ":"):
                    event.update(transfer_id=request_id + identifier[len(old):], peer=peer)


class TcpClient:
    def __init__(self, node, host, port, ledger, token_env, timeout=30, records=None):
        self.node, self.host, self.port = node, host, port
        self.ledger, self.token_env, self.timeout = ledger, token_env, timeout
        self.records = records if records is not None else []

    def call(self, kind, payload=None, upload=None, scope="search", maximum_payload=1 << 30):
        start, before = time.perf_counter(), self.ledger.used[scope].W
        try:
            return self._call(kind, payload, upload, scope, maximum_payload)
        finally:
            self.ledger.charge("tcp_unbilled_wait_or_failure", Cost(W=max(0, time.perf_counter() - start -
                               (self.ledger.used[scope].W - before))), scope)

    def _call(self, kind, payload=None, upload=None, scope="search", maximum_payload=1 << 30):
        request_id = uuid.uuid4().hex
        token = os.environ.get(self.token_env, "")
        if not token:
            raise ValueError("Set transport token environment variable: " + self.token_env)
        start = time.perf_counter()
        self.ledger.guard(scope=scope)
        timeout = min(self.timeout, max(0.001, self.ledger.remaining(scope).W))
        with socket.create_connection((self.host, self.port), timeout=timeout) as sock:
            remaining = self.ledger.remaining(scope).W
            if remaining <= 0:
                raise ExecutionStopped("No remaining wall allowance for this connection")
            sock.settimeout(min(self.timeout, remaining))
            self.ledger.charge("tcp_connect", Cost(W=time.perf_counter() - start), scope, peer=self.host)
            wire = Wire(sock, self.ledger, self.node, self.host, request_id, scope, self.records)
            wire.operation = kind
            try:
                wire.send_frame({"id": request_id, "kind": kind, "node": self.node, "token": token,
                                 "payload": payload or {}, "upload_bytes": len(upload) if upload is not None else 0}, "request")
                ready = wire.frame("response")
                if not ready.get("ok"):
                    raise RemoteOperationFailed(ready)
                if upload is not None:
                    if not ready.get("ready_for_upload"):
                        raise RuntimeError("Peer did not acknowledge bounded upload")
                    wire.send(upload, "upload")
                    ready = wire.frame("result")
                    if not ready.get("ok"):
                        raise RemoteOperationFailed(ready)
                size = ready.get("download_bytes", 0)
                if size < 0 or size > maximum_payload:
                    raise ValueError("Peer payload exceeds storage admission")
                download = wire.receive(size, "payload") if size else None
                return ready.get("result", {}), download
            finally:
                wire.finish()


class SenderQueue:
    def __init__(self):
        self.condition = threading.Condition()
        self.pending = []
        self.active = False

    def acquire(self, requester, timeout):
        ticket = (time.time_ns(), requester, uuid.uuid4().hex)
        deadline = time.monotonic() + timeout
        with self.condition:
            self.pending.append(ticket)
            while self.active or ticket != min(self.pending):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.pending.remove(ticket)
                    raise TimeoutError("Sender queue deadline")
                self.condition.wait(min(remaining, 0.5))
            self.pending.remove(ticket)
            self.active = True
        return ticket

    def release(self):
        with self.condition:
            self.active = False
            self.condition.notify_all()


class PeerService:
    def __init__(self, owner, ledger, token_env, publication_root, quota, maximum_upload=1 << 30, jobs=None,
                 interventions=None):
        self.owner, self.ledger, self.token_env = owner, ledger, token_env
        self.root = Path(publication_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.quota, self.maximum_upload, self.jobs = quota, maximum_upload, jobs
        self.queue, self.lock = SenderQueue(), threading.RLock()
        self.records = []
        self.remaining_quota = quota
        self.interventions = interventions

    def publish(self, package):
        from .storage import atomic_write
        if package.owner != self.owner:
            raise ValueError("A peer may publish only its own immutable local adapter versions")
        target = self.root / (package.hash + ".json")
        with self.lock:
            if target.exists() and target.read_bytes() != package.wire:
                raise ValueError("Immutable publication conflict")
            if not target.exists():
                atomic_write(target, package.wire)

    def restore_quota(self):
        legs = {r["leg"]: r["bytes"] for r in self.records if r.get("leg", "").endswith(":payload:tx")}
        self.remaining_quota = max(0, self.remaining_quota - sum(legs.values()))

    def catalogue(self, limit, allowlist=None):
        now, versions = time.time(), []
        if allowlist is not None:
            from .interventions import is_hash
            if not isinstance(allowlist, list) or len(allowlist) > 4096 or not all(map(is_hash, allowlist)):
                raise ValueError('Invalid exact donor allowlist')
            allowlist = frozenset(allowlist)
        with self.lock:
            files = [p for p in sorted(self.root.glob('*.json'))
                     if (allowlist is None or p.stem in allowlist) and
                     (self.interventions is None or self.interventions.allows(p.stem))]
            for file in files[:limit]:
                package = Package.read(file.read_bytes())
                if package.owner == self.owner and package.available <= now and package.created <= now:
                    versions.append({"hash": package.hash, "owner": self.owner, "observed": now,
                                     "available": package.available, "size": file.stat().st_size,
                                     "ratio": package.ratio,
                                     "interface": __import__("dataclasses").asdict(package.interface)})
        return {"owner": self.owner, "versions": versions, "payload_quota": self.remaining_quota}

    def handle(self, sock):
        started, prior_W = time.perf_counter(), self.ledger.used["search"].W
        wire = Wire(sock, self.ledger, self.owner, "unknown", uuid.uuid4().hex, "search", self.records)
        payload_started = False
        try:
            request = wire.frame("request")
            wire.reidentify(str(request["id"]), str(request["node"]))
            token = os.environ.get(self.token_env, "")
            if not token or not hmac.compare_digest(str(request.get("token", "")), token):
                raise ValueError("Authentication failed")
            kind, payload = request["kind"], request.get("payload", {})
            wire.operation = kind
            if kind == "catalogue":
                limit = min(32, max(0, int(payload.get("limit", 32))))
                wire.send_frame({"ok": True, "result": self.catalogue(limit, payload.get('allowlist'))}, "response")
            elif kind == "get":
                key = payload["hash"]
                if len(key) != 64 or any(c not in "0123456789abcdef" for c in key) or payload.get("offset", 0) != 0:
                    raise ValueError("Exact hash and zero restart offset required")
                if self.interventions is not None and not self.interventions.allows(key):
                    raise ValueError('External package excluded by donor policy')
                self.queue.acquire(wire.peer, 30)
                try:
                    data = (self.root / (key + ".json")).read_bytes()
                    package = Package.read(data)
                    if package.hash != key or package.owner != self.owner or package.available > time.time():
                        raise ValueError("Unavailable immutable package")
                    wire.send_frame({"ok": True, "download_bytes": len(data), "result": {"hash": key}}, "response")
                    payload_started = True
                    with self.lock:
                        allowance = min(len(data), self.remaining_quota)
                    cut = self.interventions.payload_limit(wire.peer, self.owner, payload.get('stage'), len(data), wire.request_id) if self.interventions else None
                    if cut is not None:
                        allowance = min(allowance, cut)
                    try:
                        wire.send(data[:allowance], "payload")
                    finally:
                        actual = wire.totals.get(wire.request_id + ":payload:tx", 0)
                        with self.lock:
                            self.remaining_quota -= actual
                        if cut is not None:
                            self.records.append({'kind': 'registered_half_payload',
                                                 **self.interventions.finish_payload(wire.request_id, actual)})
                    if cut is not None:
                        # No error frame may be appended to a truncated payload.
                        sock.shutdown(socket.SHUT_WR)
                        return
                finally:
                    self.queue.release()
            elif kind == "accounting":
                start = max(0, int(payload.get("cursor", 0)))
                stop = min(len(self.records), start + 1000)
                wire.send_frame({"ok": True, "result": {"records": self.records[start:stop], "cursor": stop,
                                "more": stop < len(self.records), "owner": self.owner}}, "response")
            elif self.jobs is not None and kind in ("audit", "evaluate", "test", "measure", "provision", "commit", "status"):
                upload_size = int(request.get("upload_bytes", 0))
                if upload_size < 0 or upload_size > self.maximum_upload:
                    raise ValueError("Agent upload exceeds transient admission")
                if kind in ("audit", "evaluate", "test", "measure", "provision"):
                    wire.send_frame({"ok": True, "ready_for_upload": True}, "response")
                    data = wire.receive(upload_size, "upload")
                    result = self.jobs.provision(payload, data) if kind == "provision" else self.jobs.execute(kind, payload, data)
                    wire.send_frame({"ok": True, "result": result}, "result")
                else:
                    result = getattr(self.jobs, kind)(payload)
                    wire.send_frame({"ok": True, "result": result}, "response")
            else:
                raise ValueError("Unknown peer operation")
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            self.records.append({"time": time.time(), "request": wire.request_id, "status": "failed", "error": str(exc)})
            try:
                if payload_started:
                    sock.shutdown(socket.SHUT_WR)
                else:
                    wire.send_frame({"ok": False, "error": str(exc), "partial_cost": getattr(exc, "partial_cost", None),
                                     "remote_job_id": getattr(exc, "remote_job_id", None),
                                     "cost_complete": getattr(exc, "cost_complete", True)}, "result")
            except (OSError, RuntimeError):
                pass
        finally:
            wire.finish()
            self.ledger.charge("source_control_or_failed_wait", Cost(W=max(0, time.perf_counter() - started -
                               (self.ledger.used["search"].W - prior_W))))


class PeerServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, service, timeout=30):
        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                self.request.settimeout(timeout)
                service.handle(self.request)
        super().__init__(address, Handler)
        self.service = service


class LiveExchange(ContactExchange):
    def __init__(self, receiver, cache, ledger, peers, token_env, contact_gate=None, timeout=30,
                 interventions=None, stage=None):
        super().__init__(receiver, cache, ledger)
        self.peers, self.token_env, self.contact_gate, self.timeout = peers, token_env, contact_gate, timeout
        self.interventions, self.stage = interventions, stage

    def _client(self, owner):
        peer = self.peers[owner]
        return TcpClient(self.receiver, peer["host"], peer["port"], self.ledger, self.token_env, self.timeout, self.records)

    def refresh(self):
        now = time.time()
        if self.interventions is not None and self.interventions.mode == 'empty':
            self.catalogue.clear()
            self.connected.clear()
            return
        for owner in sorted(self.peers):
            if self.contact_gate is not None and not self.contact_gate(owner, now):
                self.connected.discard(owner)
                continue
            try:
                request = {'limit': self.catalogue_limit}
                if self.interventions is not None and self.interventions.mode == 'allowlist':
                    request['allowlist'] = sorted(self.interventions.allowed)
                info, _ = self._client(owner).call("catalogue", request)
                if info["owner"] != owner:
                    raise ValueError("Peer owner mismatch")
                self.connected.add(owner)
                for row in info["versions"]:
                    interface = row["interface"]
                    interface["labels"] = tuple(interface["labels"])
                    row["interface"] = PortSpec(**interface)
                    row["observed"] = now
                    if (row["owner"] == owner and row["available"] <= now and row["interface"] in self.cache.interfaces and
                            (self.interventions is None or self.interventions.allows(row['hash']))):
                        self.catalogue[row["hash"]] = Advertisement(**row)
            except (OSError, RuntimeError, ValueError, KeyError) as exc:
                self.connected.discard(owner)
                self.records.append({"time": now, "owner": owner, "kind": "contact_failure", "error": str(exc)})

    def acquire(self, key, now):
        if self.interventions is not None and not self.interventions.allows(key):
            raise ExecutionStopped('External package excluded by donor policy')
        if key in self.cache:
            return self.cache.get(key)
        ad = self.catalogue.get(key)
        if ad is None or ad.owner not in self.connected or ad.available > now or ad.observed > now:
            raise ExecutionStopped("No observed live advertisement for the exact requested version")
        if ad.size > self.cache.capacity:
            raise ExecutionStopped("Incoming package cannot fit transient storage")
        result, payload = self._client(ad.owner).call("get", {"hash": key, "offset": 0, 'stage': self.stage}, maximum_payload=self.cache.capacity)
        if result["hash"] != key or payload is None:
            raise ValueError("Wrong immutable package response")
        self.cache.put(key, payload, time.time())
        return self.cache.get(key)
