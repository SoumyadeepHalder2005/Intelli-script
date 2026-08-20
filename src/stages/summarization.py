"""
Report summarization stage.

Builds structured summaries from extracted entities and validation results.
"""

import asyncio
from typing import Any, List

from src.config.settings import Settings
from src.core.exceptions import SummarizationError
from src.core.models import (
    EntityType,
    Finding,
    MedicationFinding,
    NLPEntity,
    PipelineData,
    ProcessingSummary,
)
from src.services.dataset_service import DatasetService
from src.stages.base import PipelineStage


class SummarizationStage(PipelineStage):
    """Generate human-readable summaries from extracted and validated data."""

    REQUIRED_INPUTS = ["extracted_entities", "validation_report"]

    def __init__(
        self,
        settings: Settings,
        dataset_service: DatasetService,
        stage_key: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(settings=settings, stage_key=stage_key, **kwargs)
        self.dataset_service = dataset_service

        try:
            self.layman_terms = self.dataset_service.get_dataset("layman_terms")
            self.logger.info("Summarization datasets loaded")
        except Exception as exc:
            self.logger.error("Failed to load summarization datasets: %s", exc)
            raise SummarizationError(
                f"Missing summarization dataset: {exc}"
            ) from exc

    async def process(self, data: PipelineData) -> PipelineData:
        """Generate summary artifacts for the processed document."""
        try:
            self.logger.info("Generating summary report")
            data.summary = await asyncio.to_thread(self._build_summary_sync, data)
            self.logger.info("Summary generation complete")
            return data

        except Exception as exc:
            self.logger.error("Summarization failed: %s", exc, exc_info=True)
            data.summary = ProcessingSummary(
                document_summary=f"Summary generation failed: {exc}"
            )
            raise SummarizationError(f"Summarization failed: {exc}") from exc

    def _build_summary_sync(self, data: PipelineData) -> ProcessingSummary:
        """Build the complete summary object."""
        abnormal_findings = self._get_abnormal_findings(data)
        medications = self._extract_medications_summary(data)
        diagnoses = self._extract_diagnoses_summary(
            abnormal_findings,
            data.extracted_entities,
        )
        key_findings = self._extract_key_findings(data)
        overall_summary = self._generate_overall_summary(
            data,
            abnormal_findings,
            medications,
        )
        layman_summary = self._generate_layman_summary(
            data,
            abnormal_findings,
            medications,
            diagnoses,
        )

        return ProcessingSummary(
            document_summary=overall_summary,
            key_findings=key_findings,
            abnormal_values=abnormal_findings,
            medications=medications,
            diagnoses=diagnoses,
            layman_explanation=layman_summary,
        )

    def _get_abnormal_findings(self, data: PipelineData) -> List[Finding]:
        """Return abnormal and critical findings from the validation report."""
        if not data.validation_report:
            return []
        return (
            data.validation_report.critical_values
            + data.validation_report.abnormal_values
        )

    def _generate_overall_summary(
        self,
        data: PipelineData,
        abnormal_findings: List[Finding],
        medications: List[MedicationFinding],
    ) -> str:
        """Generate an overall structured summary."""
        test_count = sum(
            1 for entity in data.extracted_entities
            if entity.type == EntityType.TEST_RESULT
        )
        vital_count = sum(
            1 for entity in data.extracted_entities
            if entity.type == EntityType.VITAL_SIGN
        )
        med_count = len(medications)
        critical_count = (
            len(data.validation_report.critical_values)
            if data.validation_report
            else 0
        )
        abnormal_count = (
            len(data.validation_report.abnormal_values)
            if data.validation_report
            else 0
        )

        lines = [f"Report: {data.file_name}"]

        if data.classification:
            lines.append(
                "Type: "
                f"{data.classification.document_type.value} "
                f"(confidence: {data.classification.confidence:.0%})"
            )

        extracted_parts = []
        if test_count > 0:
            extracted_parts.append(f"{test_count} test result(s)")
        if vital_count > 0:
            extracted_parts.append(f"{vital_count} vital sign(s)")
        if med_count > 0:
            extracted_parts.append(f"{med_count} medication(s)")

        if extracted_parts:
            lines.append("Data extracted: " + ", ".join(extracted_parts))

        if critical_count > 0:
            lines.append(
                f"Critical findings: {critical_count} value(s) requiring immediate attention"
            )
        if abnormal_count > 0:
            lines.append(f"Abnormal findings: {abnormal_count} value(s)")
        if critical_count == 0 and abnormal_count == 0:
            lines.append("All validated findings are within normal ranges")

        return "\n".join(lines)

    def _extract_key_findings(self, data: PipelineData) -> List[str]:
        """Extract key findings, prioritizing critical and abnormal values."""
        if not data.validation_report:
            return ["No validation report available"]

        findings: List[str] = []

        for finding in data.validation_report.critical_values:
            findings.append(
                "CRITICAL: "
                f"{finding.test_name} = "
                f"{self._format_value_with_unit(finding.value, finding.unit)} "
                f"({finding.reason})"
            )

        for finding in data.validation_report.abnormal_values:
            findings.append(
                "ABNORMAL: "
                f"{finding.test_name} = "
                f"{self._format_value_with_unit(finding.value, finding.unit)} "
                f"({finding.reason})"
            )

        return findings or ["All findings within normal limits"]

    def _extract_medications_summary(
        self,
        data: PipelineData,
    ) -> List[MedicationFinding]:
        """Extract medication entities into a structured summary list."""
        medications: List[MedicationFinding] = []
        seen = set()

        for entity in data.extracted_entities:
            if entity.type != EntityType.MEDICATION:
                continue

            med_name_key = entity.name.strip().lower()
            if med_name_key in seen:
                continue

            seen.add(med_name_key)
            medications.append(
                MedicationFinding(
                    name=entity.name,
                    dosage=str(entity.value) if entity.value else "Not specified",
                    unit=entity.unit or None,
                    frequency=entity.metadata.get("frequency", "As prescribed"),
                    confidence=entity.confidence,
                    metadata=entity.metadata,
                )
            )

        return medications

    def _extract_diagnoses_summary(
        self,
        abnormal_findings: List[Finding],
        extracted_entities: List[NLPEntity],
    ) -> List[str]:
        """Derive possible diagnoses from findings and extracted condition entities."""
        diagnoses = set()

        for finding in abnormal_findings:
            name = finding.test_name.lower()
            reason = (finding.reason or "").lower()

            if "blood pressure" in name and "hypertensive crisis" in reason:
                diagnoses.add("Hypertensive Crisis")
            elif "blood pressure" in name and "hypertension" in reason:
                diagnoses.add("Hypertension (High Blood Pressure)")
            elif "glucose" in name and "high" in reason:
                diagnoses.add("Hyperglycemia (High Blood Sugar)")
            elif "hemoglobin" in name and "low" in reason:
                diagnoses.add("Anemia (Low Hemoglobin)")
            elif "heart rate" in name and "tachycardia" in reason:
                diagnoses.add("Tachycardia (Elevated Heart Rate)")
            elif "heart rate" in name and "bradycardia" in reason:
                diagnoses.add("Bradycardia (Low Heart Rate)")
            elif "oxygen" in name and "hypoxemia" in reason:
                diagnoses.add("Hypoxemia (Low Oxygen Saturation)")

        for entity in extracted_entities:
            if entity.type == EntityType.CONDITION:
                diagnoses.add(entity.name.title())

        return sorted(diagnoses)

    def _generate_layman_summary(
        self,
        data: PipelineData,
        abnormal_findings: List[Finding],
        medications: List[MedicationFinding],
        diagnoses: List[str],
    ) -> str:
        """Generate a patient-friendly explanation."""
        parts: List[str] = []
        templates = self.layman_terms.get("test_results", {})

        if data.validation_report and data.validation_report.critical_values:
            parts.append(
                templates.get("critical", {}).get(
                    "explanation",
                    "Critical findings were identified.",
                )
            )
        elif abnormal_findings:
            parts.append(
                templates.get("abnormal", {}).get(
                    "explanation",
                    "Abnormal findings were identified.",
                )
            )
        else:
            parts.append(
                templates.get("normal", {}).get(
                    "explanation",
                    "Findings are within expected ranges.",
                )
            )

        if medications:
            med_names = ", ".join(med.name for med in medications)
            parts.append(f"Current medications mentioned in the report: {med_names}.")

        if diagnoses:
            diag_explanations = []
            diag_templates = self.layman_terms.get("medical_to_layman", {})

            for diagnosis in diagnoses:
                key = diagnosis.split(" (")[0].lower().replace(" ", "_")
                template = diag_templates.get(key)

                if template:
                    diag_explanations.append(
                        f"{template['layman']}: {template['explanation']}"
                    )
                else:
                    diag_explanations.append(diagnosis)

            parts.append("Health findings: " + " ".join(diag_explanations))

        return "\n\n".join(parts)

    def _format_value_with_unit(self, value: Any, unit: Any) -> str:
        """Format a value and optional unit without stray spaces."""
        return " ".join(part for part in [str(value), unit or ""] if part).strip()