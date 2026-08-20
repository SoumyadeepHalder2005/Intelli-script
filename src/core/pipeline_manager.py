"""
Pipeline orchestrator for the Intelli-Script document processing workflow.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Type

from src.config.settings import Settings
from src.core.exceptions import (
    IntelliScriptError,
    ProcessingTimeoutError,
    StageDependencyError,
    UnsupportedFileTypeError,
)
from src.core.models import PipelineData, PipelineStage, ProcessingStatus
from src.services.dataset_service import DatasetService
from src.services.pdf_table_extractor import PDFTableExtractor
from src.services.storage_service import StorageService
from src.stages.base import PipelineStage as BaseStage
from src.stages.classification import ClassificationStage
from src.stages.extraction import ExtractionStage
from src.stages.ocr import OCRStage
from src.stages.packaging import PackagingStage
from src.stages.phi_scrubbing import PHIScrubbingStage
from src.stages.preprocessing import PreprocessingStage
from src.stages.summarization import SummarizationStage
from src.stages.validation import ValidationStage


STAGE_REGISTRY: Dict[str, Type[BaseStage]] = {
    "preprocessing": PreprocessingStage,
    "ocr": OCRStage,
    "phi_scrubbing": PHIScrubbingStage,
    "classification": ClassificationStage,
    "extraction": ExtractionStage,
    "validation": ValidationStage,
    "summarization": SummarizationStage,
}


class PipelineManager:
    """Orchestrates the OCR and NLP pipeline."""

    def __init__(
        self,
        settings: Settings,
        dataset_service: DatasetService,
        storage_service: StorageService,
        pdf_table_extractor: PDFTableExtractor,
    ) -> None:
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.stages: List[BaseStage] = []

        injectable_services = {
            "dataset_service": dataset_service,
            "storage_service": storage_service,
            "pdf_table_extractor": pdf_table_extractor,
        }

        self.logger.info("Initializing pipeline stages")

        for stage_name in self.settings.pipeline.active_stages:
            if stage_name == "packaging":
                continue

            if stage_name not in STAGE_REGISTRY:
                self.logger.warning(
                    "Unknown stage '%s' in settings. Skipping.",
                    stage_name,
                )
                continue

            stage_class = STAGE_REGISTRY[stage_name]

            try:
                stage_instance = stage_class(
                    settings=settings,
                    stage_key=stage_name,
                    **injectable_services,
                )
                self.stages.append(stage_instance)
                self.logger.info("Loaded stage: %s", stage_instance.name)
            except Exception as exc:
                self.logger.error(
                    "Failed to instantiate stage '%s'. Check constructor arguments: %s",
                    stage_name,
                    exc,
                )
                raise

        self.logger.info(
            "Pipeline initialized with %d active stages",
            len(self.stages),
        )

        try:
            self.logger.info("Loading final packaging stage")
            self.packaging_stage = PackagingStage(
                settings=settings,
                stage_key="packaging",
                **injectable_services,
            )
            self.logger.info("Packaging stage loaded")
        except Exception as exc:
            self.logger.error(
                "Failed to instantiate PackagingStage: %s",
                exc,
            )
            raise

    async def process_document(self, input_path: str) -> PipelineData:
        """Process a document through the pipeline."""
        start_time = time.perf_counter()
        data = self._create_pipeline_data(input_path)

        try:
            self.logger.info(
                "Processing file: %s (%d bytes)",
                data.file_name,
                data.file_size_bytes,
            )

            for stage in self.stages:
                stage_start = time.perf_counter()

                try:
                    self._validate_dependencies(stage, data)
                    self.logger.info("Executing stage: %s", stage.name)

                    data = await asyncio.wait_for(
                        stage.process(data),
                        timeout=self.settings.pipeline.timeout_seconds,
                    )

                    stage_time_ms = (time.perf_counter() - stage_start) * 1000
                    data.stage_timings[PipelineStage(stage.stage_key)] = stage_time_ms
                    self.logger.info(
                        "%s completed in %.2f ms",
                        stage.name,
                        stage_time_ms,
                    )

                except (IntelliScriptError, asyncio.TimeoutError) as exc:
                    stage_time_ms = (time.perf_counter() - stage_start) * 1000
                    data.stage_timings[PipelineStage(stage.stage_key)] = stage_time_ms
                    self._handle_stage_exception(exc, stage.name, data)
                    raise

            data.status = ProcessingStatus.SUCCESS
            self.logger.info("Pipeline processing completed successfully")

        except IntelliScriptError as exc:
            self.logger.warning("Pipeline failed with known error: %s", exc)

            if data.status not in {ProcessingStatus.ERROR, ProcessingStatus.FAILED}:
                data.status = ProcessingStatus.FAILED

        except Exception as exc:
            self.logger.exception(
                "Pipeline failed with unhandled exception: %s",
                exc,
            )
            data.status = ProcessingStatus.ERROR
            data.errors.append(
                {
                    "error_code": "PIPELINE_UNHANDLED_ERROR",
                    "stage": "PipelineManager",
                    "message": f"An unexpected error occurred: {exc}",
                }
            )

        finally:
            total_time_ms = (time.perf_counter() - start_time) * 1000
            data.processing_time_ms = total_time_ms
            self.logger.info(
                "Pipeline finished in %.2f ms with status %s",
                total_time_ms,
                data.status.value,
            )

            try:
                self.logger.info("Handing off to PackagingStage")
                data = await self.packaging_stage.process(data)
            except Exception as exc:
                self.logger.error("PackagingStage failed: %s", exc)
                data.errors.append(
                    {
                        "error_code": "PACKAGING_FAILURE",
                        "stage": "PackagingStage",
                        "message": f"Failed to save final report: {exc}",
                    }
                )

        return data

    def _create_pipeline_data(self, input_path_str: str) -> PipelineData:
        """Create the initial pipeline data object from an input file path."""
        file_path = Path(input_path_str)

        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path_str}")

        file_size = file_path.stat().st_size
        file_type = file_path.suffix.lower()

        if file_type not in self.settings.pipeline.supported_file_types:
            raise UnsupportedFileTypeError(str(file_path), file_type)

        return PipelineData(
            input_file_path=file_path,
            file_name=file_path.name,
            file_type=file_type,
            file_size_bytes=file_size,
            status=ProcessingStatus.PROCESSING,
        )

    def _validate_dependencies(self, stage: BaseStage, data: PipelineData) -> None:
        """Validate that required stage inputs exist and are usable."""
        for required_field in stage.REQUIRED_INPUTS:
            value = getattr(data, required_field, None)

            if value is None:
                raise StageDependencyError(
                    stage.name,
                    f"Required field '{required_field}' is missing",
                )

            if isinstance(value, (list, dict, str)) and len(value) == 0:
                if (
                    stage.stage_key in {"validation", "summarization"}
                    and required_field == "extracted_entities"
                ):
                    continue

                if stage.stage_key == "phi_scrubbing" and required_field == "full_text":
                    continue

                raise StageDependencyError(
                    stage.name,
                    f"Required field '{required_field}' is empty",
                )

    def _handle_stage_exception(
        self,
        exc: Exception,
        stage_name: str,
        data: PipelineData,
    ) -> None:
        """Normalize and record stage exceptions."""
        if isinstance(exc, asyncio.TimeoutError):
            exc = ProcessingTimeoutError(
                f"{stage_name} timed out after {self.settings.pipeline.timeout_seconds}s",
                timeout_seconds=self.settings.pipeline.timeout_seconds,
            )

        if isinstance(exc, IntelliScriptError):
            self.logger.error("%s failed: %s", stage_name, exc.message)
            payload = exc.to_dict()
            payload.setdefault("stage", stage_name)
            data.errors.append(payload)
            return

        self.logger.exception(
            "%s failed with unhandled exception: %s",
            stage_name,
            exc,
        )
        data.errors.append(
            {
                "error_code": "STAGE_UNHANDLED_ERROR",
                "stage": stage_name,
                "message": f"An unexpected error occurred in stage: {exc}",
            }
        )