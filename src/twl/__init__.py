"""Resource-budgeted think-while-listening for on-device cascaded voice agents.

Invariants that hold across the whole package:
- All timestamps within a turn share one ``time.perf_counter_ns()`` origin.
- No module fabricates a number: every reported quantity traces to a log line
  under ``results/raw/``.
- Configuration is data (YAML under ``src/configs/``), validated at load time.
"""

__version__ = "0.0.1"
