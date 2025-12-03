  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
 46
 47
 48
 49
 50
 51
 52
 53
 54
 55
 56
 57
 58
 59
 60
 61
 62
 63
 64
 65
 66
 67
 68
 69
 70
 71
 72
 73
 74
 75
 76
 77
 78
 79
 80
 81
 82
 83
 84
 85
 86
 87
 88
 89
 90
 91
 92
 93
 94
 95
 96
 97
 98
 99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
"""
ModelRegistry replacement with batching, concurrency control, warmup, and basic OOM protection.

Provides:
 - ModelRegistry class with load_model / unload_model / list_models
 - Per-model AsyncBatcher and concurrency Semaphore
 - predict_sync(model, version, request_payload) -> blocking call
 - predict_async(...) -> awaitable

Tuning knobs per model:
 - max_concurrency: number of parallel in-flight requests
 - max_batch_size, batch_latency_ms
 - warmup sample & iters

This implementation is deliberately opinionated but small; extend it to:
 - integrate with Prometheus metrics and OpenTelemetry spans
 - add model eviction / LRU caching of loaded models
 - advanced OOM handling and preflight memory checks
"""
import asyncio
import logging
import time
from typing import Dict, Any, Optional

from .model_loader import TorchModelWrapper, ONNXModelWrapper, get_preferred_device, BaseModelWrapper
from .batcher import AsyncBatcher

logger = logging.getLogger("aegis.model_runner")


class ModelConfig:
    def __init__(self, model_path: str, runtime: str = "torch", device: Optional[str] = None,
                 max_concurrency: int = 4, max_batch_size: int = 8, batch_latency_ms: int = 50,
                 warmup_sample: Optional[dict] = None, warmup_iters: int = 1):
        self.model_path = model_path
        self.runtime = runtime
        self.device = device
        self.max_concurrency = max_concurrency
        self.max_batch_size = max_batch_size
        self.batch_latency_ms = batch_latency_ms
        self.warmup_sample = warmup_sample
        self.warmup_iters = warmup_iters


class ModelRegistry:
    def __init__(self):
        # key = (model_name, version)
        self._configs: Dict[str, ModelConfig] = {}
        self._models: Dict[str, BaseModelWrapper] = {}
        self._batchers: Dict[str, AsyncBatcher] = {}
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        self._loop = asyncio.get_event_loop()

    def _key(self, model_name: str, version: str) -> str:
        return f"{model_name}:{version}"

    def register(self, model_name: str, version: str, config: ModelConfig):
        k = self._key(model_name, version)
        self._configs[k] = config

    def load(self, model_name: str, version: str):
        k = self._key(model_name, version)
        if k in self._models:
            return self._models[k]
        cfg = self._configs.get(k)
        if not cfg:
            raise KeyError("model config not registered")
        # choose wrapper
        if cfg.runtime.lower().startswith("torch"):
            wrapper = TorchModelWrapper(cfg.model_path, model_name, version, device=cfg.device or get_preferred_device())
        elif cfg.runtime.lower().startswith("onnx"):
            wrapper = ONNXModelWrapper(cfg.model_path, model_name, version, use_gpu=(cfg.device == "cuda"))
        else:
            raise ValueError("unsupported runtime")
        # load & warmup
        try:
            wrapper.load()
        except Exception:
            # torch wrapper loads lazily in warmup too; ignore here
            logger.exception("model load failed for %s", k)
        if cfg.warmup_sample:
            try:
                wrapper.warmup(cfg.warmup_sample, iters=cfg.warmup_iters)
            except Exception:
                logger.exception("warmup failed for %s", k)
        # create batcher and sem
        sem = asyncio.Semaphore(cfg.max_concurrency)
        batcher = AsyncBatcher(process_batch=wrapper.predict_batch, max_batch_size=cfg.max_batch_size, max_latency_ms=cfg.batch_latency_ms, loop=self._loop)
        self._models[k] = wrapper
        self._batchers[k] = batcher
        self._semaphores[k] = sem
        logger.info("Model %s loaded with concurrency=%d batch_size=%d", k, cfg.max_concurrency, cfg.max_batch_size)
        return wrapper

    async def predict_async(self, model_name: str, version: str, input_payload: Any, timeout_s: float = 30.0):
        k = self._key(model_name, version)
        if k not in self._models:
            # try to load on demand
            try:
                self.load(model_name, version)
            except Exception:
                raise KeyError("model not found")
        sem = self._semaphores[k]
        batcher = self._batchers[k]

        # Acquire concurrency permit (async)
        try:
            await asyncio.wait_for(sem.acquire(), timeout=timeout_s)
        except asyncio.TimeoutError:
            raise TimeoutError("concurrency limit acquire timeout")
        try:
            # Submit to batcher; batcher returns result sync per item
            fut = await batcher.submit(input_payload)
            return fut
        finally:
            try:
                sem.release()
            except Exception:
                logger.exception("semaphore release failed for %s", k)

    def predict_sync(self, model_name: str, version: str, input_payload: Any, timeout_s: float = 30.0):
        """
        Blocking wrapper for sync callers (e.g., FastAPI handlers).
        Runs the asyncio predict_async in the registry loop with timeout.
        """
        coro = self.predict_async(model_name, version, input_payload, timeout_s=timeout_s)
        return asyncio.get_event_loop().run_until_complete(coro)

    def unload(self, model_name: str, version: str):
        k = self._key(model_name, version)
        if k in self._batchers:
            # stop batcher gracefully
            try:
                coro = self._batchers[k].stop()
                asyncio.get_event_loop().run_until_complete(coro)
            except Exception:
                logger.exception("failed to stop batcher for %s", k)
            del self._batchers[k]
        if k in self._models:
            try:
                self._models[k].cpu_offload()
            except Exception:
                pass
            del self._models[k]
        if k in self._semaphores:
            del self._semaphores[k]
        if k in self._configs:
            del self._configs[k]
        logger.info("Unloaded model %s", k)

    def list_models(self):
        return list(self._configs.keys())


# Global registry instance for easy import
registry = ModelRegistry()
