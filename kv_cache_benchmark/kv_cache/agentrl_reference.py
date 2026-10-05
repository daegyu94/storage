"""Dense veRL/vLLM TP geometry; no GPU performance or connector emulation."""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class ReferenceSettings:
    gpu_name: str
    tensor_parallel_size: int = 1
    framework: str = "verl"
    orchestrator: str = "ray"
    engine: str = "vllm"
    engine_version: str = "0.29.0"
    storage_payload_scope: str = "replica_aggregate_unpadded"
    gpu_memory_bytes: int | None = None
    kv_budget_bytes_per_gpu: int | None = None

    @classmethod
    def parse(cls, values):
        if not isinstance(values, dict) or set(values) - {f.name for f in fields(cls)}:
            raise ValueError("rollout_reference requires known reference fields")
        try:
            result = cls(**values)
        except TypeError as exc:
            raise ValueError("rollout_reference requires gpu_name") from exc
        if not isinstance(result.gpu_name, str) or not result.gpu_name.strip():
            raise ValueError("reference gpu_name must be a nonempty label")
        for name, expected in (
            ("framework", "verl"),
            ("orchestrator", "ray"),
            ("engine", "vllm"),
            ("engine_version", "0.29.0"),
            ("storage_payload_scope", "replica_aggregate_unpadded"),
        ):
            if getattr(result, name) != expected:
                raise ValueError(f"reference {name} must be {expected}")
        for name in ("tensor_parallel_size", "gpu_memory_bytes", "kv_budget_bytes_per_gpu"):
            value = getattr(result, name)
            if value is None and name != "tensor_parallel_size":
                continue
            if type(value) is not int or value <= 0:
                raise ValueError(f"reference {name} must be a positive integer")
        if (
            result.gpu_memory_bytes is not None
            and result.kv_budget_bytes_per_gpu is not None
            and result.kv_budget_bytes_per_gpu > result.gpu_memory_bytes
        ):
            raise ValueError("reference KV budget exceeds declared gpu memory")
        return result

    def geometry(self, model):
        if model.attention_type not in ("mha", "gqa"):
            raise ValueError("reference supports uniform dense MHA/GQA only")
        if model.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("reference KV dtype requires uncompressed float storage")
        for name in ("num_layers", "hidden_dim", "num_heads", "kv_heads"):
            if type(getattr(model, name)) is not int or getattr(model, name) <= 0:
                raise ValueError(f"invalid reference model {name}")
        if type(model._kv_dim_override) is not int or model._kv_dim_override < 0:
            raise ValueError("invalid reference model head dimension")
        if (not model._kv_dim_override and model.hidden_dim % model.num_heads) or model.num_heads % model.kv_heads:
            raise ValueError("invalid reference model head divisibility")
        tp = self.tensor_parallel_size
        if model.num_heads % tp or max(tp, model.kv_heads) % min(tp, model.kv_heads):
            raise ValueError("reference TP must divide query heads and shard/replicate KV heads exactly")
        worker_heads = max(1, model.kv_heads // tp)
        worker_bytes = model.num_layers * worker_heads * model.kv_dim_per_head * 2 * model.bytes_per_element
        return {
            "unique_bytes_per_token": model.kv_cache_size_per_token,
            "worker_kv_heads": worker_heads,
            "head_dim": model.kv_dim_per_head,
            "worker_bytes_per_token": worker_bytes,
            "replica_storage_bytes_per_token": tp * worker_bytes,
            "kv_replication_factor": max(1, tp // model.kv_heads),
        }

    def budget(self, geometry, block_tokens):
        if self.kv_budget_bytes_per_gpu is None:
            return None
        if type(block_tokens) is not int or block_tokens <= 0:
            raise ValueError("reference capacity block_tokens must be a positive integer")
        page_bytes = block_tokens * geometry["worker_bytes_per_token"]
        pages = self.kv_budget_bytes_per_gpu // page_bytes
        if not pages:
            raise ValueError("reference GPU KV budget cannot fit one block")
        return {
            "declared_bytes_per_gpu": self.kv_budget_bytes_per_gpu,
            "pages_per_worker": pages,
            "tokens_per_worker": pages * block_tokens,
            "unused_bytes_per_gpu": self.kv_budget_bytes_per_gpu - pages * page_bytes,
            "replica_capacity_bytes": pages * page_bytes * self.tensor_parallel_size,
        }

    def summary(self, model, cache):
        geometry = self.geometry(model)
        return {
            "framework": self.framework,
            "orchestrator": self.orchestrator,
            "engine": self.engine,
            "engine_version": self.engine_version,
            "gpu_name": self.gpu_name,
            "gpu_memory_bytes": self.gpu_memory_bytes,
            "tensor_parallel_size": self.tensor_parallel_size,
            "storage_payload_scope": self.storage_payload_scope,
            "kv_geometry": geometry,
            "gpu_kv_budget": self.budget(geometry, cache.get("block_tokens", 16) if cache else 16),
            "service_time_fidelity": "uncalibrated",
            "connector_behavior_fidelity": "uncalibrated",
            "execution_adapter": "synthetic_lifecycle",
        }
