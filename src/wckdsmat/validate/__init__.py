"""Validation against motion capture: pair, align, trace, score, summarize.

See :mod:`wckdsmat.validate.run` for the per-segment flow and
:mod:`wckdsmat.validate.metrics` for metric definitions.
"""
from .run import summarize, validate_many, validate_segment, write_outputs

__all__ = ["validate_segment", "validate_many", "summarize", "write_outputs"]
