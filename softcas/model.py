"""eq:adapter/weights/forward; freeze weights and buffers, never input gradients."""
from __future__ import annotations
import copy
from dataclasses import replace
import torch
from torch import nn
from .state import State, freeze_tensors, thaw_tensors, digest


class Adapter(nn.Module):
    def __init__(self, channels, ratio):
        super().__init__()
        hidden = max(1, int(channels * ratio))
        self.down = nn.Conv2d(channels, hidden, 1, bias=False)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.up = nn.Conv2d(hidden, channels, 1, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.gate.sigmoid() * self.up(torch.nn.functional.silu(
            self.dw(torch.nn.functional.silu(self.down(x)))))


class PortComposition(nn.Module):
    def __init__(self, spec, local, packages, logits):
        super().__init__()
        self.local = local
        self.external = nn.ModuleDict()
        self.beta = nn.ParameterDict()
        for package in packages:
            key = package.hash
            branch = Adapter(spec.channels, package.ratio)
            branch.load_state_dict(thaw_tensors(package.tensors), strict=True)
            branch.requires_grad_(False)
            branch.eval()
            self.external[key] = branch
            self.beta[key] = nn.Parameter(torch.tensor(float(logits[key])))
        self.last_penalty = None
        self.path_calls = 0
        self.epsilon = 1e-8
        self.discrete_choice = None
        self.is_discrete = False
        self.capture_penalty = True

    def weights(self):
        keys = tuple(self.external.keys())
        if not keys:
            return keys, torch.ones(1)
        values = [self.beta[key] for key in keys]
        weights = torch.softmax(torch.stack([values[0].new_zeros(())] + values), dim=0)
        return keys, weights

    def forward(self, x):
        y = x
        self.last_penalty = None
        if self.local is not None:
            residual = self.local(x)
            dims = tuple(range(1, x.ndim))
            if self.capture_penalty and torch.is_grad_enabled():
                self.last_penalty = (residual.float().square().sum(dims) /
                                     (x.float().square().sum(dims) + self.epsilon)).mean()
            y = y + residual
        keys, weights = (tuple(self.external.keys()), None) if self.is_discrete else self.weights()
        for i, key in enumerate(keys):
            if self.is_discrete:
                if key != self.discrete_choice:
                    continue
                coefficient = 1.0
            else:
                coefficient = weights[i + 1]
            # Even tiny coefficients execute and incur path cost. Both use x.
            y = y + coefficient * self.external[key](x)
            self.path_calls += 1
        return y


class Assembly(nn.Module):
    def __init__(self, base, port_modules, specs, state, cache, seed, mode):
        super().__init__()
        self.base = copy.deepcopy(base).float()
        self.base.requires_grad_(False)
        self.base.eval()
        self.ports = nn.ModuleDict()
        self.handles = []
        modules = dict(self.base.named_modules())
        lineage = __import__("json").loads(state.nu)
        signature = digest([__import__("dataclasses").asdict(spec) for spec in specs])
        if lineage.get("base") != specs[0].base or lineage.get("ports_signature") != signature:
            raise ValueError("Complete checkpoint base/label/preprocessing/port identity mismatch")
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            for i, (module_name, spec) in enumerate(zip(port_modules, specs)):
                # External-module construction must not shift later local draws.
                # Shared ports get identical initial phi across diagnostic contexts.
                import hashlib
                port_seed = int.from_bytes(hashlib.sha256(f"{seed}:{i}".encode()).digest()[:8], "big") % (2**63 - 1)
                with torch.random.fork_rng():
                    torch.manual_seed(port_seed)
                    local = Adapter(spec.channels, state.z.ratio) if state.z.mask[i] else None
                packages = [cache.get(key) for key in state.C if cache.get(key).interface.port == i]
                self.ports[str(i)] = PortComposition(spec, local, packages, dict(state.beta))
                port = self.ports[str(i)]
                port.is_discrete = lineage.get("receiving_rule") == "discrete"
                port.discrete_choice = lineage.get("discrete_choices", {}).get(str(i))
                self.handles.append(modules[module_name].register_forward_hook(
                    lambda module, args, output, p=port: p(output)))
        if mode != "fresh":
            self.load_local(state.phi)
        self.set_mode(mode)

    def load_local(self, phi):
        for i, port in self.ports.items():
            if port.local is not None:
                values = {k[len(i) + 1:]: v for k, v in thaw_tensors(phi).items() if k.startswith(i + ".")}
                port.local.load_state_dict(values, strict=True)

    def set_mode(self, mode):
        if mode not in ("fresh", "refine", "collaboration", "import"):
            raise ValueError(mode)
        self.base.requires_grad_(False)
        for port in self.ports.values():
            port.capture_penalty = mode != "import"
            if port.local is not None:
                port.local.requires_grad_(mode in ("fresh", "refine"))
            port.external.requires_grad_(False)
            port.beta.requires_grad_(mode != "import" and not port.is_discrete)
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        # Eval freezes BN running statistics, dropout, and all base buffers.
        # Raw detection training outputs are extracted explicitly by the backend.
        self.base.eval()
        for port in self.ports.values():
            port.external.eval()
        return self

    def forward(self, image):
        return self.base(image)

    def penalty(self):
        values = [p.last_penalty for p in self.ports.values() if p.last_penalty is not None]
        return sum(values) if values else next(self.base.parameters()).new_zeros(())

    def snapshot(self, state):
        local = {}
        beta = {}
        for i, port in self.ports.items():
            if port.local is not None:
                local.update({i + "." + k: v for k, v in port.local.state_dict().items()})
            beta.update({k: float(v.detach()) for k, v in port.beta.items()})
        return replace(state, phi=freeze_tensors(local), beta=tuple(sorted(beta.items())))

    def optimizer(self, lr, beta_lr, settings=None):
        groups = []
        local = [v for p in self.ports.values() if p.local is not None
                 for v in p.local.parameters() if v.requires_grad]
        beta = [v for p in self.ports.values() for v in p.beta.parameters() if v.requires_grad]
        if local:
            groups.append({"params": local, "lr": lr})
        if beta:
            groups.append({"params": beta, "lr": beta_lr})
        # A new optimizer object is created for every fitting block.
        if not groups:
            return None
        settings = dict(settings or {"name": "SGD", "momentum": 0.0, "weight_decay": 0.0})
        name = settings.pop("name")
        if name not in ("SGD", "Adam", "AdamW"):
            raise ValueError("Unsupported fitting optimizer: " + name)
        return getattr(torch.optim, name)(groups, **settings)
