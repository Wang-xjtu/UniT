"""Shared cleanup for partially initialized attention caches."""

from torch import nn


def release_caches(model: nn.Module) -> None:
    """Release initialized queues without allocating or invoking model hooks."""
    for module in model.modules():
        if getattr(module, "kv_cache", None) is not None:
            module.kv_cache = None
