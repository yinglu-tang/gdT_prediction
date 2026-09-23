"""Gamma-delta TCR embedding and prediction, with optional model backends."""

from importlib import import_module

__version__ = "0.1.0"
__all__ = ["EmbeddingPipeline", "TCRPipeline", "ESMCPipeline"]

_EXPORTS = {
    "EmbeddingPipeline": (".pipeline_prediction", "EmbeddingPipeline"),
    "TCRPipeline": (".pipeline_TCR", "TCRPipeline"),
    "ESMCPipeline": (".pipeline_esmc", "ESMCPipeline"),
}


def __getattr__(name):
    """Import model dependencies only when their public classes are requested."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

