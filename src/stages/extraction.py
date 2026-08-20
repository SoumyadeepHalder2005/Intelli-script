"""
Entity extraction stage using spaCy EntityRuler and PDF table parsing.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import spacy
from spacy.tokens import Span

from src.config.settings import Settings
from src.core.exceptions import ExtractionError
from src.core.models import EntityType, NLPEntity, PipelineData
from src.services.dataset_service import DatasetService
from src.services.pdf_table_extractor import PDFTableExtractor
from src.stages.base import PipelineStage


VALUE_REGEX = re.compile(
    r"(?:\bis\b\s*[:\-]?\s*|[:\-]\s*|\bDosage\b\s*[:\-]?\s*)"
    r"(\d{1,3}/\d{1,3}|\d+\.\d+|\d+)"
    r"\s*([a-zA-Z/%°µ]+)?",
    re.IGNORECASE,
)

VALUE_CONTEXT_WINDOW = 50


class ExtractionStage(PipelineStage):
    """Extract medical entities from text and PDF tables."""

    REQUIRED_INPUTS = ["scrubbed_text"]

    def __init__(
        self,
        settings: Settings,
        dataset_service: DatasetService,
        pdf_table_extractor: PDFTableExtractor,
        stage_key: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(settings=settings, stage_key=stage_key, **kwargs)
        self.dataset_service = dataset_service
        self.pdf_table_extractor = pdf_table_extractor

        try:
            self.logger.info("Loading spaCy model: %s", self.settings.spacy.model)
            self.nlp = spacy.load(self.settings.spacy.model, exclude=["ner"])
            self._build_entity_ruler()
            self.logger.info("spaCy model and EntityRuler loaded successfully")
        except Exception as exc:
            self.logger.exception(
                "Failed to load spaCy model or build EntityRuler: %s",
                exc,
            )
            raise ExtractionError(f"Failed to initialize spaCy: {exc}") from exc

    def _build_entity_ruler(self) -> None:
        """Load datasets and build a case-insensitive EntityRuler."""
        self.logger.info("Building case-insensitive EntityRuler from datasets")

        if "entity_ruler" in self.nlp.pipe_names:
            self.nlp.remove_pipe("entity_ruler")

        if "parser" in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="parser")
        else:
            ruler = self.nlp.add_pipe("entity_ruler")

        patterns: List[Dict[str, Any]] = []

        def create_pattern(term: str, label: EntityType) -> Dict[str, Any]:
            tokens = term.split()
            token_patterns = [{"LOWER": token.lower()} for token in tokens]
            return {
                "label": label.value,
                "pattern": token_patterns,
                "id": term,
            }

        try:
            lab_ranges = self.dataset_service.get_dataset("lab_ranges")
            for key in lab_ranges.get("blood_tests", {}).keys():
                patterns.append(create_pattern(key, EntityType.TEST_RESULT))
            for key in lab_ranges.get("vital_signs", {}).keys():
                patterns.append(create_pattern(key, EntityType.VITAL_SIGN))
        except Exception as exc:
            self.logger.warning(
                "Could not load 'lab_ranges' for EntityRuler: %s",
                exc,
            )

        try:
            medical_entities = self.dataset_service.get_dataset("medical_entities")
            for _, terms in medical_entities.get("medications", {}).items():
                for term in terms:
                    patterns.append(create_pattern(term, EntityType.MEDICATION))
            for term in medical_entities.get("conditions", []):
                patterns.append(create_pattern(term, EntityType.CONDITION))
            for term in medical_entities.get("symptoms", []):
                patterns.append(create_pattern(term, EntityType.SYMPTOM))
        except Exception as exc:
            self.logger.warning(
                "Could not load 'medical_entities' for EntityRuler: %s",
                exc,
            )

        ruler.add_patterns(patterns)
        self.logger.info("EntityRuler built with %d patterns", len(patterns))

    async def process(self, data: PipelineData) -> PipelineData:
        """Extract entities from text and PDF tables."""
        try:
            text_entities = await asyncio.to_thread(
                self._extract_from_text,
                data.scrubbed_text,
            )

            table_entities: List[NLPEntity] = []
            if data.file_type == ".pdf":
                table_entities = await self._extract_from_tables(data)

            combined_entities: Dict[tuple[str, str], NLPEntity] = {}

            for entity in table_entities:
                combined_entities[(entity.type.value, entity.name.lower())] = entity

            for entity in text_entities:
                key = (entity.type.value, entity.name.lower())
                if key not in combined_entities:
                    combined_entities[key] = entity

            data.extracted_entities = list(combined_entities.values())
            self.logger.info(
                "Extracted %d total unique entities",
                len(data.extracted_entities),
            )
            return data

        except ExtractionError:
            raise
        except Exception as exc:
            self.logger.exception("Extraction failed: %s", exc)
            raise ExtractionError(f"Extraction failed: {exc}") from exc

    def _extract_from_text(self, text: str) -> List[NLPEntity]:
        """Run spaCy and extract ruler-based entities from text."""
        entities: List[NLPEntity] = []

        if not text:
            return entities

        doc = self.nlp(text)

        for ent in doc.ents:
            try:
                entity_type = EntityType(ent.label_)
            except ValueError:
                self.logger.warning("Unknown entity label '%s' from spaCy", ent.label_)
                continue

            entity = NLPEntity(
                type=entity_type,
                name=ent.text,
                confidence=0.9,
                metadata={
                    "source": "text_ruler",
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                },
            )

            if entity_type in {
                EntityType.TEST_RESULT,
                EntityType.VITAL_SIGN,
                EntityType.MEDICATION,
            }:
                value, unit = self._find_value_for_entity(ent)
                entity.value = value
                entity.unit = unit

            entities.append(entity)

        return entities

    def _find_value_for_entity(self, ent: Span) -> Tuple[Optional[str], Optional[str]]:
        """Look for a nearby value immediately after an extracted entity."""
        context_window = ent.doc.text[ent.end_char : ent.end_char + VALUE_CONTEXT_WINDOW]
        match = VALUE_REGEX.search(context_window)

        if not match:
            return None, None

        value = match.group(1)
        unit = match.group(2) if match.group(2) else None

        try:
            numeric_value = float(value)
            if 1900 < numeric_value < 2100:
                return None, None
        except ValueError:
            pass

        return value, unit

    async def _extract_from_tables(self, data: PipelineData) -> List[NLPEntity]:
        """Extract structured entities from PDF tables."""
        try:
            tables = await self.pdf_table_extractor.extract_tables(str(data.input_file_path))
            if not tables:
                return []
        except Exception as exc:
            self.logger.warning("PDF table extraction failed: %s", exc)
            return []

        try:
            lab_ranges = self.dataset_service.get_dataset("lab_ranges").get(
                "blood_tests",
                {},
            )
        except Exception as exc:
            self.logger.warning("Failed to load lab_ranges for table extraction: %s", exc)
            return []

        known_tests = {key.lower() for key in lab_ranges.keys()}
        entities: List[NLPEntity] = []

        for table in tables:
            for row in table.rows:
                if not row or len(row) < 2:
                    continue

                test_name = str(row[0]).strip() if row[0] else ""
                if test_name.lower() not in known_tests:
                    continue

                value_str = str(row[1]).strip() if row[1] else ""
                try:
                    value: Any = float(value_str)
                except ValueError:
                    value = value_str

                unit = str(row[2]).strip() if len(row) > 2 and row[2] else None

                entities.append(
                    NLPEntity(
                        type=EntityType.TEST_RESULT,
                        name=test_name,
                        value=value,
                        unit=unit,
                        confidence=0.98,
                        metadata={
                            "source": "pdf_table",
                            "page": table.page_number,
                        },
                    )
                )

        self.logger.info("Extracted %d entities from PDF tables", len(entities))
        return entities