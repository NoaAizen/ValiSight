"""Display-channel definitions for the live viewer.

A display channel answers "what should the operator see?".  It is deliberately
separate from the thermal/radar/fusion *student* switches, which answer "which
AI evidence should be computed and diagnosed?".  Keeping the two concepts apart
prevents a top-level AI view from becoming a fourth meaning of ``ai_fusion``.

The visible source is currently the PAG7936 luma stream, not true colour RGB.
The public id keeps ``rgb`` because it is the requested product channel and will
remain stable when the board-side RGB565 transport lands; the label is honest
about what this build actually supplies.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ChannelSpec:
    id: str
    label: str
    base: str
    raw_radar: bool = False
    ai_overlay: bool = False


CHANNEL_SPECS = (
    # AI overlays belong on the thermal-bearing operator products as well as
    # the dedicated AI page: in darkness the visible base is black, while the
    # thermal image is the surface on which a body outline is actually useful.
    ChannelSpec("thermal", "thermal", "thermal", ai_overlay=True),
    ChannelSpec("rgb_radar", "visible + radar", "visible", raw_radar=True),
    ChannelSpec("thermal_radar", "thermal + radar", "thermal",
                raw_radar=True, ai_overlay=True),
    ChannelSpec("fusion", "thermal + visible + radar", "fusion",
                raw_radar=True, ai_overlay=True),
    ChannelSpec("ai", "ai", "visible", ai_overlay=True),
)

CHANNELS = tuple(spec.id for spec in CHANNEL_SPECS)
_BY_ID = {spec.id: spec for spec in CHANNEL_SPECS}


def get(channel):
    """Return a validated channel, defaulting to the operational fusion view."""
    return _BY_ID.get(channel, _BY_ID["fusion"])


def public_specs():
    """Small JSON-safe description used to build the browser control."""
    return [{"id": s.id, "label": s.label} for s in CHANNEL_SPECS]
