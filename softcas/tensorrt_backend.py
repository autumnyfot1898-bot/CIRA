"""TensorRT 8/10 adapters built ON the receiving Jetson, never copied from RTX.

The primary TensorRT mode compiles each residual function separately. Python's
composition loop still invokes every selected external member, even at a zero
floating-point receiving coefficient. Whole-model ONNX is an optional export,
not silently substituted for this membership-preserving executor.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .state import digest, canonical
from .storage import atomic_write


def export_onnx(module, example, path, inference_output_only=False):
    class Wrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            value = self.model(x)
            return value[0] if inference_output_only and isinstance(value, (tuple, list)) else value
    module.eval()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        import inspect
        compatibility = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
        torch.onnx.export(Wrapper(module), example, str(path), input_names=["images"], output_names=["predictions"],
                          opset_version=17, do_constant_folding=False, dynamic_axes=None, **compatibility)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_engine(onnx, output, runtime, fp16=False, workspace_bytes=1 << 30):
    import tensorrt as trt
    runtime.validate()
    from .metering import CURRENT_COUNTER
    counter = CURRENT_COUNTER.get()
    if counter is not None:
        counter.unknown["TensorRT_builder_autotuning_requires_separate_target_cost_calibration"] += 1
        if counter.strict:
            raise RuntimeError("Build and cost-profile target plans during an explicit reserved calibration phase before strict execution")
    if runtime.family != "jetson_orin":
        raise ValueError("This deployment profile builds its plans on the receiving Orin")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx).read_bytes()):
        raise RuntimeError("TensorRT ONNX parse failed: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes
    if fp16:
        if not getattr(builder, "platform_has_fast_fp16", True):
            raise RuntimeError("Requested FP16 builder is unsupported")
        config.set_flag(trt.BuilderFlag.FP16)
    if hasattr(trt.BuilderFlag, "TF32"):
        config.clear_flag(trt.BuilderFlag.TF32)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")
    output = Path(output)
    atomic_write(output, bytes(plan))
    metadata = {"onnx_sha256": hashlib.sha256(Path(onnx).read_bytes()).hexdigest(),
                "plan_sha256": hashlib.sha256(bytes(plan)).hexdigest(), "runtime": runtime.identity(),
                "tensorrt": trt.__version__, "fp16": fp16, "workspace_bytes": workspace_bytes,
                "portable": False, "membership_execution": "one_call_per_selected_adapter"}
    atomic_write(output.with_suffix(output.suffix + ".json"), canonical(metadata))
    return metadata


class TensorRTRunner:
    def __init__(self, plan, runtime):
        import tensorrt as trt
        self.trt = trt
        self.runtime_spec = runtime
        meta = json.loads(Path(str(plan) + ".json").read_bytes())
        if meta["runtime"] != runtime.identity() or meta["tensorrt"] != trt.__version__:
            raise ValueError("Engine hardware/software/precision identity mismatch")
        data = Path(plan).read_bytes()
        if hashlib.sha256(data).hexdigest() != meta["plan_sha256"]:
            raise ValueError("Engine checksum mismatch")
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(data)
        if self.engine is None:
            raise RuntimeError("TensorRT plan deserialization failed")
        self.context = self.engine.create_execution_context()
        self.calls = 0
        cost_file = Path(str(plan) + ".cost.json")
        self.cost_profile = json.loads(cost_file.read_bytes()) if cost_file.is_file() else None
        if self.cost_profile and (self.cost_profile["plan_sha256"] != meta["plan_sha256"] or
                                  self.cost_profile["runtime"] != runtime.identity() or self.cost_profile["status"] != "validated"):
            raise ValueError("TensorRT execution cost profile is not valid for this exact target plan")

    def __call__(self, x):
        trt = self.trt
        x = x.contiguous()
        stream = torch.cuda.current_stream(x.device)
        buffers = {}
        if hasattr(self.engine, "num_io_tensors") and hasattr(self.context, "execute_async_v3"):
            names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
            if len(inputs) != 1:
                raise ValueError("Residual engine must have exactly one tensor input")
            self.context.set_input_shape(inputs[0], tuple(x.shape))
            for name in names:
                dtype = torch.from_numpy(np.empty((), dtype=trt.nptype(self.engine.get_tensor_dtype(name)))).dtype
                if name == inputs[0]:
                    buffers[name] = x.to(dtype=dtype)
                else:
                    shape = tuple(self.context.get_tensor_shape(name))
                    if any(d < 0 for d in shape):
                        raise RuntimeError("Unresolved TensorRT output dimensions")
                    buffers[name] = torch.empty(shape, dtype=dtype, device=x.device)
                self.context.set_tensor_address(name, buffers[name].data_ptr())
            ok = self.context.execute_async_v3(stream.cuda_stream)
            outputs = [buffers[n] for n in names if n not in inputs]
        else:
            bindings, outputs = [], []
            for i in range(self.engine.num_bindings):
                dtype = torch.from_numpy(np.empty((), dtype=trt.nptype(self.engine.get_binding_dtype(i)))).dtype
                if self.engine.binding_is_input(i):
                    self.context.set_binding_shape(i, tuple(x.shape))
                    value = x.to(dtype=dtype)
                else:
                    value = torch.empty(tuple(self.context.get_binding_shape(i)), dtype=dtype, device=x.device)
                    outputs.append(value)
                buffers[i] = value
                bindings.append(value.data_ptr())
            ok = self.context.execute_async_v2(bindings, stream.cuda_stream)
        if not ok or len(outputs) != 1:
            raise RuntimeError("TensorRT execution failed or output contract changed")
        # Keep bindings alive until all asynchronous use completes.
        stream.synchronize()
        self.calls += 1
        return outputs[0].to(dtype=x.dtype)


class TensorRTAdapter(nn.Module):
    def __init__(self, runner):
        super().__init__()
        self.runner = runner
    def forward(self, x):
        if torch.is_grad_enabled() and x.requires_grad:
            raise RuntimeError("TensorRT adapters are deployment-only; fitting uses native PyTorch")
        from .metering import CURRENT_COUNTER
        counter = CURRENT_COUNTER.get()
        if counter is not None:
            if not self.runner.cost_profile:
                counter.unknown["TensorRT_opaque_plan_requires_target_kernel_profile"] += 1
                if counter.strict:
                    raise RuntimeError("Provide a validated target TensorRT cost profile or use the native target executor for strict F accounting")
        output = self.runner(x)
        if counter is not None and self.runner.cost_profile:
            counter.add_external("TensorRT_exact_plan", self.runner.cost_profile["scalar_flops_per_call"])
        return output


def register_plan_cost(plan, runtime, reviewed_layer_costs):
    """Bind reviewed operation counts to every actual optimized layer of a plan.

    This requires target engine metadata; no nominal RTX FLOPs are transferred.
    Native-versus-optimized layer counts are intentionally not assumed equal.
    """
    runner = TensorRTRunner(plan, runtime)
    trt = runner.trt
    inspector = runner.engine.create_engine_inspector()
    layers = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    layers = layers.get("Layers", layers) if isinstance(layers, dict) else layers
    names = {row["Name"] for row in layers}
    if set(reviewed_layer_costs["layers"]) != names or not reviewed_layer_costs.get("reviewed_by"):
        raise ValueError("Every actual optimized TensorRT layer needs a reviewed operation count")
    if any(not isinstance(v, (int, float)) or v < 0 for v in reviewed_layer_costs["layers"].values()):
        raise ValueError("Invalid optimized layer operation count")
    value = {"status": "validated", "plan_sha256": hashlib.sha256(Path(plan).read_bytes()).hexdigest(),
             "runtime": runtime.identity(), "counting_convention": "scalar_ops_mac2_v1",
             "scalar_flops_per_call": sum(reviewed_layer_costs["layers"].values()),
             "layers": reviewed_layer_costs["layers"], "reviewed_by": reviewed_layer_costs["reviewed_by"],
             "inspector_sha256": digest(layers)}
    atomic_write(str(plan) + ".cost.json", canonical(value))
    return value


def install_adapter_engines(model, example, runtime, directory, fp16=False, workspace_bytes=1 << 30):
    directory = Path(directory)
    collected, handles = {}, []
    def capture(port_id):
        def hook(module, args):
            collected[port_id] = args[0].detach().clone()
        return hook
    for port_id, port in model.ports.items():
        handles.append(port.register_forward_pre_hook(capture(port_id)))
    with torch.no_grad(), runtime.autocast():
        model(example)
    for h in handles:
        h.remove()
    plans = []
    for port_id, port in model.ports.items():
        branches = ([("local", port.local)] if port.local is not None else []) + list(port.external.items())
        for name, branch in branches:
            from .state import freeze_tensors
            signature = digest({"weights": [(k, v.__dict__) for k, v in freeze_tensors(branch.state_dict())],
                                "shape": list(collected[port_id].shape), "runtime": runtime.identity(), "fp16": fp16})
            onnx, plan = directory / (signature + ".onnx"), directory / (signature + ".plan")
            if not plan.is_file():
                export_onnx(branch, collected[port_id].float(), onnx)
                build_engine(onnx, plan, runtime, fp16, workspace_bytes)
            replacement = TensorRTAdapter(TensorRTRunner(plan, runtime))
            if name == "local":
                port.local = replacement
            else:
                port.external[name] = replacement
            plans.append({"port": port_id, "branch": name, "plan": str(plan), "bytes": plan.stat().st_size})
    return plans
