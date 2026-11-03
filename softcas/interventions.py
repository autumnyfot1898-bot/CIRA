"""Trajectory-bound donor visibility and durable, once-only payload interruption."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import threading
import time
from pathlib import Path
from .state import canonical, digest
from .storage import atomic_write
from .resources import ExecutionStopped


def is_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_interventions(spec):
    spec = copy.deepcopy(spec or {})
    if not isinstance(spec, dict) or set(spec) - {'donors', 'contacts'}:
        raise ValueError('Unknown runtime intervention fields')
    donor, contact = spec.get('donors') or {}, spec.get('contacts') or {}
    mode = donor.get('mode', 'normal')
    if mode not in ('normal', 'empty', 'allowlist'):
        raise ValueError('Unknown donor policy')
    if mode == 'allowlist':
        allowed = donor.get('allowlist')
        if not isinstance(allowed, list) or not allowed or len(set(allowed)) != len(allowed) or not all(map(is_hash, allowed)):
            raise ValueError('Weak donor policy needs unique exact package hashes')
        binding = donor.get('development_lock', {})
        if donor.get('role') != 'DEV' or not is_hash(binding.get('sha256')) or not binding.get('path'):
            raise ValueError('Weak donors require a DEV lock binding')
        path = Path(binding['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != binding['sha256']:
            raise ValueError('Weak donor DEV lock changed')
        lock = json.loads(path.read_bytes())
        if (lock.get('role') != 'DEV' or lock.get('schema') != 'cira.weak_donor_lock.v1' or
                sorted(lock.get('allowlist', [])) != sorted(allowed)):
            raise ValueError('Weak donor policy differs from DEV lock')
    if contact:
        if (contact.get('kind') != 'single_half_payload' or contact.get('stage') != 2 or
                contact.get('payload_fraction') != .5 or not is_hash(contact.get('event_id')) or
                not isinstance(contact.get('receiver'), str) or not isinstance(contact.get('owner'), str) or
                contact['receiver'] == contact['owner'] or
                isinstance(contact.get('contact_offset_seconds'), bool) or
                not isinstance(contact.get('contact_offset_seconds'), (int, float)) or
                not math.isfinite(contact['contact_offset_seconds']) or contact['contact_offset_seconds'] < 0):
            raise ValueError('Bind one explicit stage-two half-payload event')
    return {'donors': donor, 'contacts': contact}


class RuntimeInterventions:
    def __init__(self, spec=None, state_path=None, trace=(), origin=0.):
        self.spec = validate_interventions(spec)
        self.mode = self.spec['donors'].get('mode', 'normal')
        self.allowed = frozenset(self.spec['donors'].get('allowlist', []))
        self.trace, self.origin = tuple(copy.deepcopy(trace)), origin
        event = self.spec['contacts']
        if event and not any(str(r['receiver'])==event['receiver'] and str(r['owner'])==event['owner'] and
                             r['offset_seconds']==event['contact_offset_seconds'] and r['online'] for r in self.trace):
            raise ValueError('Registered half-payload event has no matching online contact in this trace')
        self.path = Path(state_path) if state_path is not None else None
        self.lock = threading.RLock()
        self.state = {'schema': 'cira.runtime_interventions.v1', 'identity': digest(self.spec), 'claims': {}}
        if self.path is not None and self.path.exists():
            saved = json.loads(self.path.read_bytes())
            if saved.get('identity') != self.state['identity'] or saved.get('schema') != self.state['schema']:
                raise ValueError('Cannot reuse intervention state from another condition')
            self.state = saved

    def allows(self, key):
        return self.mode == 'normal' or (self.mode == 'allowlist' and key in self.allowed)

    def require_state(self, state):
        if any(not self.allows(k) for k in state.C):
            raise ValueError('Inherited external members violate the trajectory donor policy')

    def _save(self):
        if self.path is not None:
            atomic_write(self.path, canonical(self.state))

    def payload_limit(self, receiver, owner, stage, total, request_id, now=None):
        """Claim only the registered contact; persist before any payload is sent.

        A process crash after the claim remains consumed/uncertain, not a second
        injected interruption. The trajectory's interrupted-stage gate still
        requires reconciliation before search may resume.
        """
        event = self.spec['contacts']
        if not event or (str(receiver), str(owner), stage) != (event['receiver'], event['owner'], event['stage']):
            return None
        elapsed = (time.time() if now is None else now) - self.origin
        current = None
        for row in sorted(self.trace, key=lambda r: r['offset_seconds']):
            if row['offset_seconds'] > elapsed:
                break
            if (str(row['receiver']), str(row['owner'])) == (str(receiver), str(owner)):
                current = row
        if current is None or not current['online'] or current['offset_seconds'] != event['contact_offset_seconds']:
            return None
        with self.lock:
            key = event['event_id']
            if key in self.state['claims']:
                return None
            cut = total // 2
            self.state['claims'][key] = {'request': request_id, 'payload_bytes': total, 'cut_after_bytes': cut,
                                        'receiver': str(receiver), 'owner': str(owner), 'stage': stage,
                                        'status': 'claimed', 'actual_payload_bytes': None}
            self._save()
            return cut

    def finish_payload(self, request_id, actual):
        with self.lock:
            for claim in self.state['claims'].values():
                if claim['request'] == request_id:
                    claim.update(actual_payload_bytes=actual,
                                 status='interrupted_at_half' if actual == claim['cut_after_bytes'] else 'interrupted_early')
                    self._save()
                    return copy.deepcopy(claim)
        raise ValueError('Unknown payload intervention request')

    def summary(self):
        with self.lock:
            return {'policy': copy.deepcopy(self.spec), 'events': copy.deepcopy(self.state['claims'])}
