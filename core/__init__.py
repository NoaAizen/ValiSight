"""core — pure layer. Zero hardware imports, zero I/O, zero network.

Everything here must be 100% testable without a radar, a camera, a UART port
or a network connection. If a module in this package needs to touch the outside
world, it belongs in `adapters`, not here. See CLAUDE.md ("ארכיטקטורה").
"""
