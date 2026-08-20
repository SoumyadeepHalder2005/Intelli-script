"""
Static, non-configurable constants used across the Intelli-Script project.

Configuration values such as paths, feature flags, and defaults are managed
in src.config.settings.
Domain-specific types and enums are managed in src.core.models.
"""

from typing import Final

# File encoding and JSON formatting
ENCODING: Final[str] = "utf-8"
OUTPUT_INDENT: Final[int] = 2
JSON_ENSURE_ASCII: Final[bool] = False