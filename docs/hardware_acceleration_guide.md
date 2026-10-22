# EdgeLoop Hardware Acceleration & Device Fallback Guide

EdgeLoop is engineered for heterogeneous edge and cloud environments. It automatically detects and binds to the optimal compute acceleration backend while guaranteeing graceful degradation to standard CPU execution.

---

## 1. Supported Acceleration Backends

| Backend | Platform | Target Hardware | Precision | Expected Latency (MobileNetV2) |
| :--- | :--- | :--- | :--- | :--- |
| **MPS** | macOS | Apple Silicon (M1/M2/M3/M4) | FP32 / FP16 | ~4 - 8 ms |
| **CUDA** | Linux / Windows | NVIDIA GPUs (Jetson, RTX, Tesla) | FP32 / FP16 | ~2 - 6 ms |
| **CPU** | All Platforms | x86_64, ARM64 (Raspberry Pi, Intel) | FP32 | ~12 - 25 ms |

---

## 2. Dynamic Device Resolution Logic

In `src/edgeloop/inference/engine.py`, device binding follows this hierarchy:

```python
def _resolve_device(self, requested_device: Optional[str]) -> torch.device:
    if requested_device and requested_device != "auto":
        return torch.device(requested_device)

    env_dev = os.getenv("EDGELOOP_DEVICE", "auto").lower()
    if env_dev == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif env_dev == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif env_dev == "cpu":
        return torch.device("cpu")

    # Automatic detection
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
```

---

## 3. Explicit Override Flags

You can force a specific compute target using the `EDGELOOP_DEVICE` environment variable or the `--device` CLI flag:

### Force CPU execution:
```bash
export EDGELOOP_DEVICE=cpu
python scripts/run_inference.py --image path/to/image.jpg --device cpu
```

### Force MPS on Apple Silicon:
```bash
export EDGELOOP_DEVICE=mps
python scripts/run_inference.py --image path/to/image.jpg --device mps
```

### Force CUDA on NVIDIA machines:
```bash
export EDGELOOP_DEVICE=cuda
python scripts/run_inference.py --image path/to/image.jpg --device cuda
```

---

## 4. Fallback Handling
If an accelerator fails to initialize or encounters out-of-memory (OOM) conditions during batch inference, the system catches the hardware exception and falls back to CPU execution without terminating the inference pipeline or corrupting the SQLite audit logs.
