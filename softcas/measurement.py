"""Within-trajectory summaries of independently identified device sessions."""
import math
from .state import digest


def summarize_sessions(reports):
    if not reports:
        raise ValueError("Missing device measurement sessions")
    identities = {r['candidate'] for r in reports}
    if len(identities) != 1 or len({r['job_id'] for r in reports}) != len(reports):
        raise ValueError("Measurement sessions must be distinct jobs for one deployed checkpoint")
    if len({digest(r.get('runtime', {})) for r in reports}) != 1:
        raise ValueError("Measurement sessions must share one bound runtime")
    devices = [r['device'] for r in reports]
    for field in ('physical_device_id', 'executor'):
        if len({d.get(field) for d in devices}) != 1:
            raise ValueError("Cannot combine device or executor identities")
    for d in devices:
        if any(not math.isfinite(d[k]) or d[k] < 0 for k in ('p95_ms', 'p50_ms', 'maximum_ms')):
            raise ValueError("Invalid device latency")
    summary = dict(devices[0])
    for field in ('p95_ms', 'p50_ms'):
        summary[field] = sum(d[field] for d in devices) / len(devices)
    for field in ('maximum_ms', 'peak_process_ram_bytes', 'peak_system_used_bytes',
                  'cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes'):
        values = [d[field] for d in devices if field in d]
        if values:
            summary[field] = max(values)
    # Keep raw timing arrays per session rather than creating a pooled quantile.
    summary.pop('latency_samples_ms', None)
    summary.update(measurement_sessions=len(reports),
                   timed_requests=sum(d['repeats'] for d in devices),
                   warmup_requests=sum(d['warmup'] for d in devices),
                   p95_aggregation='mean of per-session P95 within this trajectory; no pooled quantile',
                   session_job_ids=[r['job_id'] for r in reports])
    return summary
