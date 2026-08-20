"""
Authoritative storage service for file I/O.

Provides async-first read and write helpers, atomic writes for data integrity,
and cleanup utilities for temporary files.
"""

import asyncio
import csv
import json
import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from time import time
from typing import Any, Dict, Iterator, List

from src.config.settings import Settings
from src.core.constants import ENCODING, JSON_ENSURE_ASCII
from src.core.exceptions import FileProcessingError


class StorageService:
    """Manage file I/O operations with async wrappers and atomic writes."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self._json_cache: Dict[Path, Dict[str, Any]] = {}
        self.logger.info("StorageService initialized")

    @contextmanager
    def _atomic_writer(
        self,
        file_path: Path,
        mode: str = "w",
        encoding: str = ENCODING,
    ) -> Iterator[Any]:
        """
        Write to a temporary file in the target directory and replace on success.
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode=mode,
                encoding=encoding if "b" not in mode else None,
                dir=file_path.parent,
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                yield temp_file
                temp_file.flush()
                os.fsync(temp_file.fileno())

            os.replace(temp_path, file_path)

        except Exception as exc:
            self.logger.error("Atomic write to %s failed: %s", file_path, exc)
            if temp_path and temp_path.exists():
                temp_path.unlink()
            raise FileProcessingError(
                f"Failed to write file: {exc}",
                file_path=str(file_path),
            ) from exc

    async def save_json(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Save a dictionary to JSON asynchronously."""
        await asyncio.to_thread(self._save_json_sync, file_path, data)
        self.logger.debug("Saved JSON to %s", file_path)

    async def save_csv(self, file_path: Path, data: List[Dict[str, Any]]) -> None:
        """Save a list of records to CSV asynchronously."""
        await asyncio.to_thread(self._save_csv_sync, file_path, data)
        self.logger.debug("Saved CSV to %s", file_path)

    async def save_text(self, file_path: Path, content: str) -> None:
        """Save text content asynchronously."""
        await asyncio.to_thread(self._save_text_sync, file_path, content)
        self.logger.debug("Saved text to %s", file_path)

    async def load_json(self, file_path: Path) -> Dict[str, Any]:
        """Load a JSON file asynchronously, using a per-instance cache."""
        return await asyncio.to_thread(self._load_json_sync, file_path)

    async def cleanup_old_files(
        self,
        directory: Path,
        max_age_hours: int = 24,
    ) -> int:
        """Remove stale temporary files asynchronously."""
        try:
            return await asyncio.to_thread(
                self._cleanup_old_files_sync,
                directory,
                max_age_hours,
            )
        except Exception as exc:
            self.logger.warning("Cleanup failed in %s: %s", directory, exc)
            return 0

    def clear_cache(self) -> None:
        """Clear the in-memory JSON cache."""
        self._json_cache.clear()
        self.logger.debug("StorageService JSON cache cleared")

    def _load_json_sync(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON from disk or return a cached copy."""
        if file_path in self._json_cache:
            self.logger.debug("Loaded %s from JSON cache", file_path)
            return self._json_cache[file_path]

        self.logger.debug("Loading %s from disk", file_path)

        try:
            with file_path.open("r", encoding=ENCODING) as file_obj:
                data = json.load(file_obj)
                self._json_cache[file_path] = data
                return data

        except json.JSONDecodeError as exc:
            raise FileProcessingError(
                f"Corrupt JSON file: {exc}",
                file_path=str(file_path),
            ) from exc
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise FileProcessingError(
                f"Failed to read JSON: {exc}",
                file_path=str(file_path),
            ) from exc

    def _save_json_sync(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Write JSON atomically."""
        with self._atomic_writer(file_path, "w", encoding=ENCODING) as file_obj:
            json.dump(
                data,
                file_obj,
                indent=self.settings.pipeline.output_indent,
                ensure_ascii=JSON_ENSURE_ASCII,
            )

        self._json_cache[file_path] = data

    def _save_csv_sync(self, file_path: Path, data: List[Dict[str, Any]]) -> None:
        """Write CSV atomically."""
        if not data:
            self.logger.warning("No data provided to save CSV: %s", file_path)
            return

        fieldnames = self._collect_csv_fieldnames(data)

        with self._atomic_writer(file_path, "w", encoding=ENCODING) as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(data)

    def _save_text_sync(self, file_path: Path, content: str) -> None:
        """Write text atomically."""
        with self._atomic_writer(file_path, "w", encoding=ENCODING) as file_obj:
            file_obj.write(content)

    @staticmethod
    def _collect_csv_fieldnames(rows: List[Dict[str, Any]]) -> List[str]:
        """Collect stable CSV headers from all rows."""
        fieldnames: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        return fieldnames

    @staticmethod
    def _cleanup_old_files_sync(directory: Path, max_age_hours: int) -> int:
        """Delete stale temporary files from a directory."""
        deleted = 0
        current_time = time()
        max_age_seconds = max_age_hours * 3600

        if not directory.exists():
            return 0

        try:
            for file_path in directory.glob("*.tmp"):
                if current_time - file_path.stat().st_mtime > max_age_seconds:
                    file_path.unlink()
                    deleted += 1
        except Exception as exc:
            logging.warning("Error during cleanup in %s: %s", directory, exc)

        return deleted