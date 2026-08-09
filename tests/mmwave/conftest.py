# Make the on-device library importable under host pytest with the SAME
# import name it has on the N6 (flat module on the board's filesystem).
import os
import sys

_LIB = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                    '..', '..', 'board', 'mmwave'))
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)
