"""
Validation stage for checking extracted entities against dataset-driven rules.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List

from src.config.settings import Settings
from src.core.exceptions import ValidationError
from src.core.models import (
    EntityType,
    Finding,
    FindingStatus,
    NLPEntity,
    PipelineData,
    ValidationReport,
)
from src.services.dataset_service import DatasetService
from src.stages.base import PipelineStage


class ValidationStage(PipelineStage):
    """Validate extracted entities against external reference rules."""

    REQUIRED_INPUTS = ["extracted_entities"]

    def __init__(
        self,
        settings: Settings,
        dataset_service: DatasetService,
        stage_key: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(settings=settings, stage_key=stage_key, **kwargs)
        self.dataset_service = dataset_service

    async def process(self, data: PipelineData) -> PipelineData:
        """Validate extracted entities and build a structured report."""
        try:
            if not data.extracted_entities:
                self.logger.warning("No entities to validate; creating empty report")
                data.validation_report = self._create_empty_report()
                return data

            self.logger.info(
                "Validating %d entities",
                len(data.extracted_entities),
            )

            data.validation_report = await asyncio.to_thread(
                self._run_validation,
                data.extracted_entities,
            )

            report = data.validation_report
            self.logger.info(
                "Validation complete: %d valid, %d invalid, %d critical, %d abnormal",
                report.valid_count,
                report.invalid_count,
                len(report.critical_values),
                len(report.abnormal_values),
            )

            return data

        except Exception as exc:
            self.logger.error("Validation failed: %s", exc, exc_info=True)
            data.validation_report = self._create_empty_report(
                len(data.extracted_entities)
            )
            raise ValidationError(f"Validation failed: {exc}") from exc

    def _run_validation(self, entities: List[NLPEntity]) -> ValidationReport:
        """Run validation rules over extracted entities."""
        try:
            lab_ranges_data = self.dataset_service.get_dataset("lab_ranges")
        except Exception as exc:
            self.logger.error(
                "Could not load 'lab_ranges' dataset: %s",
                exc,
            )
            raise ValidationError(f"Missing lab_ranges dataset: {exc}") from exc

        all_findings: List[Finding] = []

        for entity in entities:
            if isinstance(entity.value, (int, float)):
                finding = self._validate_numeric_entity(entity, lab_ranges_data)
            else:
                finding = Finding(
                    test_name=entity.name,
                    value=entity.value,
                    unit=entity.unit,
                    status=FindingStatus.UNKNOWN,
                    reason="Non-numeric entity not validated against numeric rules",
                )

            all_findings.append(finding)

        critical_values = [
            finding for finding in all_findings
            if finding.status == FindingStatus.CRITICAL
        ]
        abnormal_values = [
            finding for finding in all_findings
            if finding.status == FindingStatus.ABNORMAL
        ]
        valid_count = sum(
            1
            for finding in all_findings
            if finding.status in {FindingStatus.NORMAL, FindingStatus.UNKNOWN}
        )

        return ValidationReport(
            validated_at=datetime.now(timezone.utc),
            total_entities=len(entities),
            valid_count=valid_count,
            invalid_count=len(critical_values) + len(abnormal_values),
            abnormal_values=abnormal_values,
            critical_values=critical_values,
        )

    def _validate_numeric_entity(
        self,
        entity: NLPEntity,
        rules_data: Dict[str, Any],
    ) -> Finding:
        """Validate a numeric entity against dataset-driven ranges."""
        ruleset = None
        if entity.type == EntityType.TEST_RESULT:
            ruleset = rules_data.get("blood_tests", {})
        elif entity.type == EntityType.VITAL_SIGN:
            ruleset = rules_data.get("vital_signs", {})

        if not ruleset:
            return self._create_unknown_finding(entity, "No ruleset for entity type")

        entity_key = self._normalize_key(entity.name)
        rule = ruleset.get(entity_key)

        if not rule:
            for key, rule_data in ruleset.items():
                aliases = {
                    self._normalize_key(alias)
                    for alias in rule_data.get("aliases", [])
                }
                if entity_key in aliases:
                    rule = rule_data
                    entity_key = key
                    break

        if not rule:
            return self._create_unknown_finding(
                entity,
                "Test not found in reference dataset",
            )

        value = float(entity.value)
        reference_range = self._format_range(rule.get("ranges", []))

        for target_status in (
            FindingStatus.CRITICAL,
            FindingStatus.ABNORMAL,
            FindingStatus.NORMAL,
        ):
            matched = self._match_range(rule.get("ranges", []), value, target_status.value)
            if matched:
                return Finding(
                    test_name=entity.name,
                    value=value,
                    unit=entity.unit,
                    status=target_status,
                    reason=matched["reason"],
                    reference_range=reference_range,
                )

        return self._create_unknown_finding(
            entity,
            "Value did not match any defined range",
        )

    def _match_range(
        self,
        ranges: List[Dict[str, Any]],
        value: float,
        target_status: str,
    ) -> Dict[str, Any] | None:
        """Return the first matching range for a target status."""
        for range_rule in ranges:
            if range_rule.get("status") != target_status:
                continue

            min_val, max_val = range_rule["range"]
            if (min_val is None or value >= min_val) and (
                max_val is None or value < max_val
            ):
                return range_rule

        return None

    def _create_unknown_finding(self, entity: NLPEntity, reason: str) -> Finding:
        """Create a default unknown finding."""
        return Finding(
            test_name=entity.name,
            value=entity.value,
            unit=entity.unit,
            status=FindingStatus.UNKNOWN,
            reason=reason,
        )

    def _format_range(self, ranges: List[Dict[str, Any]]) -> str:
        """Format the normal reference range for display."""
        for range_rule in ranges:
            if range_rule.get("status") == "normal":
                min_val, max_val = range_rule["range"]
                return f"{min_val} - {max_val}"
        return "N/A"

    def _create_empty_report(self, total_entities: int = 0) -> ValidationReport:
        """Create an empty validation report."""
        return ValidationReport(
            validated_at=datetime.now(timezone.utc),
            total_entities=total_entities,
            valid_count=0,
            invalid_count=total_entities,
            abnormal_values=[],
            critical_values=[],
        )

    def _normalize_key(self, value: str) -> str:
        """Normalize entity names and aliases for matching."""
        return value.strip().casefold().replace(" ", "_")