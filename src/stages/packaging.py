"""
Final output packaging stage for assembling and saving pipeline results.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List

from src.config.settings import Settings
from src.core.exceptions import PackagingError
from src.core.models import (
    MedicationFinding,
    Finding,
    PipelineData,
    ProcessingSummary,
    ValidationReport,
)
from src.services.storage_service import StorageService
from src.stages.base import PipelineStage


class PackagingStage(PipelineStage):
    """Assemble and persist final processing results."""

    REQUIRED_INPUTS = ["validation_report", "summary"]

    def __init__(
        self,
        settings: Settings,
        storage_service: StorageService,
        stage_key: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(settings=settings, stage_key=stage_key, **kwargs)
        self.storage_service = storage_service

    async def process(self, data: PipelineData) -> PipelineData:
        """Assemble and save the final report."""
        try:
            self.logger.info("Packaging final report")

            output_dict = self._build_output_dict(data)

            output_dir = self.settings.storage.outputs_dir
            json_filename = f"{data.file_name}.result.json"
            json_path = output_dir / json_filename

            await self.storage_service.save_json(json_path, output_dict)
            output_dict["json_path"] = str(json_path)

            data.structured_output = output_dict

            self.logger.info("Report packaged and saved to %s", json_path)
            return data

        except Exception as exc:
            self.logger.exception("Packaging failed: %s", exc)
            raise PackagingError(f"Packaging failed: {exc}") from exc

    def _build_output_dict(self, data: PipelineData) -> Dict[str, Any]:
        """Build the final structured output payload."""
        summary: ProcessingSummary = data.summary or ProcessingSummary()
        report: ValidationReport = (
            data.validation_report or self._create_empty_report()
        )

        return {
            "file_name": data.file_name,
            "document_type": (
                data.classification.document_type.value
                if data.classification
                else "unknown"
            ),
            "classification_confidence": (
                data.classification.confidence if data.classification else 0.0
            ),
            "extracted_entities": self._build_entity_output(data),
            "validation_summary": {
                "total_validated": report.total_entities,
                "valid_count": report.valid_count,
                "invalid_count": report.invalid_count,
                "abnormal_count": len(report.abnormal_values),
                "critical_count": len(report.critical_values),
            },
            "summary": {
                "document_summary": summary.document_summary,
                "key_findings": summary.key_findings,
                "diagnoses": summary.diagnoses,
                "layman_explanation": summary.layman_explanation,
            },
            "processing_status": data.status.value,
            "processing_time_ms": data.processing_time_ms,
            "stage_timings": {
                stage.value: duration for stage, duration in data.stage_timings.items()
            },
            "errors": data.errors,
            "warnings": data.warnings,
        }

    def _create_empty_report(self) -> ValidationReport:
        """Create a blank validation report."""
        return ValidationReport(validated_at=datetime.now(timezone.utc))

    def _build_entity_output(
        self,
        data: PipelineData,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group extracted entities by entity type for output."""
        output: Dict[str, List[Dict[str, Any]]] = {}

        for entity in data.extracted_entities:
            entity_type_key = f"{entity.type.value}s"
            if entity_type_key not in output:
                output[entity_type_key] = []

            output[entity_type_key].append(
                {
                    "name": entity.name,
                    "value": entity.value,
                    "unit": entity.unit,
                    "confidence": entity.confidence,
                    "source": entity.metadata.get("source", "unknown"),
                }
            )

        return output