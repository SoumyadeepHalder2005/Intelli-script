"""
Dataset service for loading reference datasets from a manifest.

Provides centralized access to JSON-based datasets used across the pipeline
and caches loaded content in memory for repeated reads.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

from src.config.settings import Settings
from src.core.exceptions import DatasetNotFoundError, DatasetServiceError


class DatasetService:
    """Manages access to reference datasets through a manifest."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self._cache: Dict[str, Dict[str, Any]] = {}

        base_path = self.settings.storage.datasets_dir
        self.dataset_manifest: Dict[str, Path] = {
            "lab_ranges": base_path / "medical" / "lab_reference_ranges.json",
            "medical_entities": base_path / "medical" / "medical_entities.json",
            "document_keywords": (
                base_path / "document_types" / "document_keywords.json"
            ),
            "layman_terms": base_path / "templates" / "layman_phases.json",
        }

        self.logger.info(
            "Initialized dataset service with %d datasets",
            len(self.dataset_manifest),
        )

    def get_dataset(self, name: str) -> Dict[str, Any]:
        """
        Return a dataset by its registered manifest name.

        Raises:
            DatasetNotFoundError: If the dataset name is unknown or file is missing.
            DatasetServiceError: If the dataset cannot be read or parsed.
        """
        if name not in self.dataset_manifest:
            self.logger.error("Dataset '%s' not found in manifest", name)
            raise DatasetNotFoundError(dataset_name=name)

        if name in self._cache:
            self.logger.debug("Returning cached dataset '%s'", name)
            return self._cache[name]

        file_path = self.dataset_manifest[name]
        dataset = self._load_json(file_path, dataset_name=name)
        self._cache[name] = dataset
        return dataset

    def clear_cache(self) -> None:
        """Clear all cached datasets."""
        self._cache.clear()
        self.logger.info("Dataset cache cleared")

    def _load_json(self, path: Path, dataset_name: str) -> Dict[str, Any]:
        """Load and validate a JSON dataset from disk."""
        self.logger.info("Loading dataset '%s' from %s", dataset_name, path)

        if not path.exists():
            self.logger.error("Dataset file not found: %s", path)
            raise DatasetNotFoundError(dataset_name=dataset_name, key=str(path))

        try:
            with path.open("r", encoding="utf-8") as file_obj:
                return json.load(file_obj)

        except json.JSONDecodeError as exc:
            self.logger.error(
                "Failed to parse JSON for dataset %s: %s",
                path,
                exc,
            )
            raise DatasetServiceError(
                f"Corrupt dataset file: {path}"
            ) from exc

        except OSError as exc:
            self.logger.error("Failed to read dataset %s: %s", path, exc)
            raise DatasetServiceError(
                f"I/O error reading dataset: {path}"
            ) from exc

        except Exception as exc:
            self.logger.error(
                "Unexpected error loading dataset %s: %s",
                path,
                exc,
            )
            raise DatasetServiceError(
                f"Unexpected error for dataset {path}: {exc}"
            ) from exc