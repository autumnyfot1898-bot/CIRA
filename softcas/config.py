"""Configuration is explicit; fixture choices never become manuscript defaults."""
import json
import math
from pathlib import Path
from .controller import FeatureEncoder
from .resources import Cost

METHODS = ("local", "local_nas_hpo", "fixed_collaboration", "serial", "round_robin", "softcas")
ABLATIONS = ("none", "nas_only", "hpo_only", "no_handoff", "no_refine", "discrete", "no_gain")
PENDING = ("feature_encoding", "cold_start_order", "archive_tournament_policy", "cost_calibration",
           "fitting_blocks", "serial_development_rule", "scene_registry", "base_eight_class_checkpoint",
           "six_port_interfaces", "device_limits")


def merge_dict(base, patch):
    value = __import__("copy").deepcopy(base)
    for key, item in patch.items():
        value[key] = merge_dict(value[key], item) if isinstance(item, dict) and isinstance(value.get(key), dict) else item
    return value


def effective_formal(config, formal=False):
    """Resolve the execution context once; an explicit false cannot downgrade FORMAL.

    Flags are JSON booleans, not truthy strings/integers. Callers must pass this
    result to every data, cost and deployment validation for the same job.
    """
    if not isinstance(formal, bool):
        raise ValueError("formal must be an explicit boolean")
    return formal or config.get("purpose") == "FORMAL"


def _read_config(path, parents=()):
    path = Path(path).resolve()
    if path in parents:
        raise ValueError("Configuration inheritance cycle")
    d = json.loads(path.read_text(encoding="utf-8"))
    if "extends" in d:
        parent = path.parent / d.pop("extends")
        d = merge_dict(_read_config(parent, (*parents, path)), d)
    return d


def load_config(path, formal=False, overrides=None):
    # Validate the final merged configuration. A child/job may supply protocol
    # bindings missing from its parent; validating intermediate parents would
    # reject a complete job before those bindings are applied.
    d = merge_dict(_read_config(path), overrides or {})
    return validate_config(d, formal)


def validate_config(d, formal=False):
    """Validate the effective configuration, including job-level overrides."""
    if d["method"] not in METHODS or d["ablation"] not in ABLATIONS:
        raise ValueError("Unknown matched method/intervention")
    formal = effective_formal(d, formal)
    if formal:
        missing = [k for k in PENDING if not d.get("confirmed", {}).get(k)]
        if d.get("purpose") != "FORMAL" or missing:
            raise ValueError("Formal execution blocked until development/protocol confirmations: " + ", ".join(missing))
        if d.get("proposal_protocol") != "native_r1000" or d.get("fitting", {}).get("unit") != "epochs":
            raise ValueError("Formal CIRA execution requires explicit native_r1000 epoch semantics")
        if d.get("flop_meter") != "scalar_ops_mac2_v1" or not d.get("meter", {}).get("strict"):
            raise ValueError("Formal total computation requires calibrated complete cost meter")
        if not d.get("cost_profile"):
            raise ValueError("Formal run needs an actual hardware/precision-bound cost-profile artifact")
        if not d.get("storage") or not d.get("deployment"):
            raise ValueError("Formal storage and target deployment profiles are required")
        if not d.get("device_limits") or any(v is None or not math.isfinite(v) or v <= 0 for v in d["device_limits"].values()):
            raise ValueError("Freeze numeric limits on an identified target device before formal execution")
    FeatureEncoder(d["features"]["fields"], d["features"]["scales"])
    for name in ("search", "release"):
        if any(v < 0 for v in Cost(**d["budgets"][name]).values()):
            raise ValueError("Negative allowance")
    for name in ("phi_steps", "beta_steps", "attempts", "proposals_per_type"):
        if isinstance(d[name], bool) or not isinstance(d[name], int) or d[name] < 0:
            raise ValueError("Invalid block/attempt length")
    if any(not math.isfinite(d[k]) or d[k] <= 0 for k in ("beta_lr", "residual_epsilon", "ridge_lambda", "priority_epsilon")):
        raise ValueError("Invalid fitting constants")
    if not math.isfinite(d["kappa"]) or d["kappa"] < 0 or not math.isfinite(d["audit_epsilon"]):
        raise ValueError("Invalid controller or audit constant")
    if isinstance(d["archive_capacity"], bool) or not isinstance(d["archive_capacity"], int) or d["archive_capacity"] < 2:
        raise ValueError("Archive must have room for entry and champion")
    if d.get("proposal_protocol", "legacy") not in ("legacy", "native_r1000"):
        raise ValueError("Unknown proposal protocol")
    fitting = d.get("fitting")
    if fitting is not None:
        if fitting.get("unit") != "epochs":
            raise ValueError("Explicit fitting unit must be epochs; omit it for legacy batch steps")
        for name in ("stage_epochs", "local_epochs", "receiving_epochs"):
            v = fitting.get(name)
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError("Positive integer epoch limits required")
        if d.get("proposal_protocol") != "native_r1000":
            raise ValueError("Epoch fitting requires bound native_r1000 proposal fidelity")
    sessions = d.get("report_measurement_sessions", 1)
    if isinstance(sessions, bool) or not isinstance(sessions, int) or sessions < 1:
        raise ValueError("Positive integer reporting measurement sessions required")
    if d["method"] == "serial":
        s = d["serial"]
        if set(s["order"]) != {"configuration", "collaboration"} or not 0 < s["fraction"] < 1 or not 0 < s["attempt_fraction"] < 1:
            raise ValueError("Explicit Serial order/resource/attempt split required")
    return d
