"""utils package for shared helpers (material, timing, parallel training).

This package exposes modules as submodules; imports should use
`from utils.time_manager import Clock` or `from utils.utils import material_balance`.
"""

__all__ = ["utils", "time_manager", "parallel_training"]
