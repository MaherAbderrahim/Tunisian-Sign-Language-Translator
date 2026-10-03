"""ONNX export, latency measurement and quantization helpers."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .paths import MODELS_DIR


def export_onnx(model: torch.nn.Module, sample: torch.Tensor, name: str,
                opset: int = 18) -> Path:
    """Export a model to ONNX (fp32) and verify it with onnxruntime."""
    import onnx
    import onnxruntime as ort

    path = MODELS_DIR / f"{name}.onnx"
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model, sample, str(path),
            input_names=["input"], output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=opset, dynamo=False,
        )
    onnx.checker.check_model(str(path))
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": sample.numpy()})[0]
    with torch.no_grad():
        ref = model(sample).detach().numpy()
    assert np.allclose(out, ref, atol=1e-4), "ONNX output mismatch"
    return path


def quantize_fp16(onnx_path: Path, name: str) -> Path:
    """Weight-only FP16 conversion: float initializers are stored in
    half precision and cast back to fp32 before use, so the compute graph
    stays fp32. This halves the model size without any type mismatch
    (onnxconverter_common's full-graph FP16 pass emits invalid graphs for
    some architectures, e.g. GCN/GAT)."""
    import onnx
    from onnx import TensorProto, numpy_helper, helper

    out = MODELS_DIR / f"{name}_fp16.onnx"
    model = onnx.load(str(onnx_path))
    graph = model.graph
    keep = []
    new_inits = []
    for init in list(graph.initializer):
        if init.data_type != TensorProto.FLOAT:
            keep.append(init)
            continue
        arr = numpy_helper.to_array(init).astype(np.float16)
        half_name = init.name + "_fp16w"
        new_inits.append(numpy_helper.from_array(arr, half_name))
        cast = helper.make_node("Cast", [half_name], [init.name],
                                to=TensorProto.FLOAT)
        graph.node.insert(0, cast)
    del graph.initializer[:]
    graph.initializer.extend(keep + new_inits)
    onnx.save(model, str(out))
    return out


def quantize_int8(onnx_path: Path, name: str, calib_data: np.ndarray = None) -> Path:
    """Dynamic INT8 quantization (no calibration data required)."""
    from onnxruntime.quantization import quantize_dynamic, QuantType
    out = MODELS_DIR / f"{name}_int8.onnx"
    quantize_dynamic(str(onnx_path), str(out), weight_type=QuantType.QInt8)
    return out


def ort_latency(onnx_path: Path, sample: np.ndarray, iters: int = 300) -> float:
    """Mean per-sample latency (ms) with onnxruntime, CPU, batch=1."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feed = {"input": sample}
    for _ in range(20):
        sess.run(None, feed)
    t0 = time.perf_counter()
    for _ in range(iters):
        sess.run(None, feed)
    return float((time.perf_counter() - t0) / iters * 1000)


def ort_predict(onnx_path: Path, features: np.ndarray,
                batch: int = 256) -> np.ndarray:
    """Argmax predictions of an ONNX model on a feature matrix."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outs = []
    for i in range(0, len(features), batch):
        out = sess.run(None, {"input": features[i:i + batch].astype(np.float32)})[0]
        outs.append(out.argmax(1))
    return np.concatenate(outs)


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def save_torch(model: torch.nn.Module, name: str):
    path = MODELS_DIR / f"{name}.pth"
    torch.save({"state_dict": model.state_dict()}, path)
    return path