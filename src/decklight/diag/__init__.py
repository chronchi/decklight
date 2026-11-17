"""Reusable policy diagnosis (windows, behaviour, probes, attention, interventions, report).

Architecture-agnostic: a `diag.adapter.ModelAdapter` turns one observation into option
probabilities/features/attention; everything else works from that interface.
"""

from decklight.diag.adapter import AdapterOutput, BCPolicyAdapter, HookedAdapter, ModelAdapter
from decklight.diag.windows import Window, WindowFrame

__all__ = [
    "AdapterOutput",
    "BCPolicyAdapter",
    "HookedAdapter",
    "ModelAdapter",
    "Window",
    "WindowFrame",
]
