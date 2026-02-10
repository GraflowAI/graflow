"""Graflow - An executable task graph engine."""

try:
    from importlib.metadata import version
    __version__ = version("graflow")
except Exception:
    pass
