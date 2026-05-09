"""Coloured, structured logging for the ptp package.

One call to ``configure_logging()`` at process start installs a coloured
console handler on the root logger.  Every module then calls
``get_structured_logger(__name__)`` to get a named child — no per-module
handler setup needed.

Format (console):
    HH:MM:SS.mmm  LEVEL     module.name  » message

Colours (ANSI, disabled when stdout is not a TTY):
    DEBUG    — dim white
    INFO     — bright cyan
    WARNING  — bright yellow
    ERROR    — bright red
    CRITICAL — bright red on white background
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# ANSI colour codes
# ---------------------------------------------------------------------------

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"

_CYAN   = "\033[36m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_WHITE  = "\033[37m"
_BRIGHT_CYAN   = "\033[96m"
_BRIGHT_YELLOW = "\033[93m"
_BRIGHT_RED    = "\033[91m"
_BRIGHT_WHITE  = "\033[97m"
_RED_BG        = "\033[41m"

_LEVEL_COLOURS = {
    logging.DEBUG:    _DIM + _WHITE,
    logging.INFO:     _BRIGHT_CYAN,
    logging.WARNING:  _BRIGHT_YELLOW,
    logging.ERROR:    _BRIGHT_RED,
    logging.CRITICAL: _RED_BG + _BRIGHT_WHITE + _BOLD,
}

_LEVEL_LABELS = {
    logging.DEBUG:    "DEBUG   ",
    logging.INFO:     "INFO    ",
    logging.WARNING:  "WARNING ",
    logging.ERROR:    "ERROR   ",
    logging.CRITICAL: "CRITICAL",
}


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

class _ColouredFormatter(logging.Formatter):
    """Single-line coloured formatter with fixed-width columns.

    Console output:
        15:22:48.130  INFO      ptp.camera.realsense  » Starting pipeline
        ^^^^^^^^^^^^  ^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^
        timestamp     level     logger name            message
    """

    _NAME_WIDTH = 32

    def __init__(self, use_colour: bool = True) -> None:
        super().__init__()
        self._use_colour = use_colour

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:  # noqa: N802
        import datetime
        dt = datetime.datetime.fromtimestamp(record.created)
        return dt.strftime("%H:%M:%S") + f".{dt.microsecond // 1000:03d}"

    def format(self, record: logging.LogRecord) -> str:
        ts    = self.formatTime(record)
        level = _LEVEL_LABELS.get(record.levelno, record.levelname.ljust(8))
        name  = record.name
        msg   = record.getMessage()

        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)

        if self._use_colour:
            lvl_colour = _LEVEL_COLOURS.get(record.levelno, "")
            ts_str    = _DIM + ts + _RESET
            lvl_str   = lvl_colour + level + _RESET
            name_str  = _CYAN + name.ljust(self._NAME_WIDTH) + _RESET
            arrow_str = _DIM + "»" + _RESET
            msg_str   = (_BOLD if record.levelno >= logging.ERROR else "") + msg + _RESET
        else:
            ts_str   = ts
            lvl_str  = level
            name_str = name.ljust(self._NAME_WIDTH)
            arrow_str = "»"
            msg_str  = msg

        return f"{ts_str}  {lvl_str}  {name_str}  {arrow_str} {msg_str}"


# ---------------------------------------------------------------------------
# File formatter (no colour, with date)
# ---------------------------------------------------------------------------

_FILE_FORMATTER = logging.Formatter(
    fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    callback: Optional[Callable[[str, logging.LogRecord], None]] = None,
    force: bool = False,
) -> None:
    """Install a coloured console handler on the root logger.

    Call once at process start (e.g. in ``main()``).  Subsequent calls are
    no-ops unless ``force=True``.

    Args:
        level:    Minimum log level (default INFO).
        log_file: Optional path to write plain-text log alongside the console.
        callback: Optional ``(formatted_str, record)`` callable for UI layers.
        force:    Re-install handlers even if already configured.
    """
    root = logging.getLogger()

    # Suppress noisy third-party loggers that we don't own.
    for noisy in ("transformers", "accelerate", "torch", "PIL", "urllib3",
                  "httpx", "httpcore", "openai", "google", "bitsandbytes"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if not force and root.handlers:
        root.setLevel(min(root.level, level) if root.level else level)
        return

    root.setLevel(level)

    use_colour = sys.stdout.isatty()
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(_ColouredFormatter(use_colour=use_colour))
    root.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(_FILE_FORMATTER)
        root.addHandler(fh)

    if callback is not None:
        class _CallbackHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                try:
                    callback(self.format(record), record)
                except Exception:
                    self.handleError(record)

        ch = _CallbackHandler(level)
        ch.setFormatter(_ColouredFormatter(use_colour=False))
        root.addHandler(ch)


class RunTimer:
    """Lightweight wall-clock timer for named stages.

    Usage::

        t = RunTimer()
        with t.measure("model_load.molmo"):
            load_molmo()
        t.log_summary(logger)
        timings = t.to_dict()   # JSON-serialisable
    """

    def __init__(self) -> None:
        import time as _time
        self._time = _time
        self._records: list = []   # list of (name, elapsed_s)
        self._stack:  list = []    # (name, t0) for nested contexts

    class _Context:
        def __init__(self, timer: "RunTimer", name: str) -> None:
            self._timer = timer
            self._name  = name
        def __enter__(self) -> "RunTimer._Context":
            self._timer._stack.append((self._name, self._timer._time.perf_counter()))
            return self
        def __exit__(self, *_: object) -> None:
            name, t0 = self._timer._stack.pop()
            self._timer._records.append((name, self._timer._time.perf_counter() - t0))

    def measure(self, name: str) -> "_Context":
        """Context manager: ``with timer.measure("stage"): ...``"""
        return self._Context(self, name)

    def record(self, name: str, elapsed_s: float) -> None:
        """Manually record a pre-measured duration."""
        self._records.append((name, elapsed_s))

    def elapsed(self, name: str) -> float:
        """Sum of all records matching name (0.0 if none)."""
        return sum(e for n, e in self._records if n == name)

    def to_dict(self) -> dict:
        """Return {name: elapsed_s} with summed duplicates, JSON-serialisable."""
        out: dict = {}
        for name, elapsed in self._records:
            out[name] = round(out.get(name, 0.0) + elapsed, 4)
        return out

    def log_summary(self, logger: logging.Logger, level: int = logging.INFO) -> None:
        """Log all timings as a formatted table."""
        if not self._records:
            return
        totals = self.to_dict()
        width = max(len(k) for k in totals)
        lines = ["Timing summary:"]
        for name, elapsed in totals.items():
            lines.append(f"  {name:<{width}}  {elapsed:7.3f}s")
        logger.log(level, "\n".join(lines))


def get_structured_logger(name: str) -> logging.Logger:
    """Return a named child logger that propagates to the root handler.

    Args:
        name: Typically ``__name__`` or a short descriptive string.

    Example::

        logger = get_structured_logger(__name__)
        logger.info("Pipeline started")
    """
    return logging.getLogger(name)
