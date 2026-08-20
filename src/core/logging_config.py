"""
Logging configuration for the Intelli-Script OCR pipeline.

Provides colored console logging, rotating file logs, and a dedicated
performance logger.
"""

import logging
import logging.handlers
import sys

from src.config.settings import Settings


class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"


class ColoredFormatter(logging.Formatter):
    """Logging formatter that adds ANSI colors for console output."""

    LEVEL_COLORS = {
        logging.DEBUG: Colors.CYAN,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.BOLD + Colors.RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        original_levelname = record.levelname
        original_name = record.name

        try:
            level_color = self.LEVEL_COLORS.get(record.levelno, Colors.WHITE)
            record.levelname = f"{level_color}{original_levelname}{Colors.RESET}"
            record.name = f"{Colors.BLUE}{original_name}{Colors.RESET}"
            return super().format(record)
        finally:
            record.levelname = original_levelname
            record.name = original_name


def _clear_handlers(logger: logging.Logger) -> None:
    """Remove and close all handlers attached to a logger."""
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()


def setup_logging(settings: Settings) -> None:
    """Configure root logging with console and rotating file handlers."""
    log_level = getattr(logging, settings.logging.level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    _clear_handlers(root_logger)

    if settings.logging.log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(
            ColoredFormatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(console_handler)

    if settings.logging.log_to_file and settings.logging.log_file:
        log_file = settings.storage.logs_dir / settings.logging.log_file
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=settings.logging.max_file_size_mb * 1024 * 1024,
            backupCount=settings.logging.backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(file_handler)


def setup_performance_logging(settings: Settings) -> None:
    """Configure a dedicated rotating logger for performance metrics."""
    perf_logger = logging.getLogger("performance")
    perf_logger.setLevel(logging.INFO)
    perf_logger.propagate = False
    _clear_handlers(perf_logger)

    if settings.logging.log_to_file:
        perf_log_file = settings.storage.logs_dir / settings.logging.perf_log_file
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_log_file,
            maxBytes=settings.logging.perf_max_file_size_mb * 1024 * 1024,
            backupCount=settings.logging.perf_backup_count,
            encoding="utf-8",
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        perf_logger.addHandler(perf_handler)


def log_performance(stage_name: str, duration_ms: float) -> None:
    """Log execution time for a pipeline stage."""
    perf_logger = logging.getLogger("performance")
    perf_logger.info("%s: %.2fms", stage_name, duration_ms)