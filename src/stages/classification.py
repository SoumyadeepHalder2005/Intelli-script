"""
Document classification stage using spaCy PhraseMatcher.
"""

import asyncio
from typing import Any, Dict, List, Tuple

import spacy
from spacy.matcher import PhraseMatcher

from src.config.settings import Settings
from src.core.exceptions import ClassificationError
from src.core.models import (
    DocumentClassification,
    DocumentType,
    PipelineData,
)
from src.services.dataset_service import DatasetService
from src.stages.base import PipelineStage


class ClassificationStage(PipelineStage):
    """Classify medical documents using spaCy phrase matching."""

    REQUIRED_INPUTS = ["scrubbed_text"]

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
            self.logger.info("Loading spaCy model: %s", self.settings.spacy.model)
            self.nlp = spacy.load(self.settings.spacy.model)
            self.logger.info("spaCy model loaded successfully")
        except OSError as exc:
            self.logger.exception(
                "Failed to load spaCy model '%s'",
                self.settings.spacy.model,
            )
            raise ClassificationError(
                f"Missing spaCy model: {self.settings.spacy.model}"
            ) from exc

        self.matcher, self.pattern_weights, self.total_possible_scores = (
            self._build_matcher()
        )

    async def process(self, data: PipelineData) -> PipelineData:
        """Classify the document type from scrubbed text."""
        try:
            if not data.scrubbed_text or not data.scrubbed_text.strip():
                self.logger.warning("No text available for classification")
                data.classification = DocumentClassification(
                    document_type=DocumentType.UNKNOWN,
                    confidence=0.0,
                    keywords_found=[],
                )
                return data

            doc_type, confidence, keywords = await asyncio.to_thread(
                self._classify_document,
                data.scrubbed_text,
            )

            data.classification = DocumentClassification(
                document_type=doc_type,
                confidence=confidence,
                keywords_found=keywords,
            )

            self.logger.info(
                "Classified as %s with confidence %.2f%% using %d keyword matches",
                doc_type.value,
                confidence * 100,
                len(keywords),
            )
            return data

        except ClassificationError:
            raise
        except Exception as exc:
            self.logger.exception("Classification failed: %s", exc)
            raise ClassificationError(f"Classification failed: {exc}") from exc

    def _build_matcher(
        self,
    ) -> tuple[PhraseMatcher, Dict[DocumentType, Dict[str, float]], Dict[DocumentType, float]]:
        """Build and cache the PhraseMatcher and per-pattern weights."""
        try:
            keywords_db = self.dataset_service.get_dataset("document_keywords")
        except Exception as exc:
            self.logger.error("Failed to load document_keywords dataset: %s", exc)
            raise ClassificationError(
                f"Missing document_keywords dataset: {exc}"
            ) from exc

        matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        valid_doc_types = {enum_member.value for enum_member in DocumentType}
        valid_doc_types.discard(DocumentType.UNKNOWN.value)

        pattern_weights: Dict[DocumentType, Dict[str, float]] = {}
        total_possible_scores: Dict[DocumentType, float] = {}

        for doc_type_str, info in keywords_db.items():
            if doc_type_str not in valid_doc_types:
                continue

            doc_type_enum = DocumentType(doc_type_str)
            keywords = info.get("keywords", [])
            indicators = info.get("indicators", [])

            docs = []
            weights_for_type: Dict[str, float] = {}

            for keyword in keywords:
                normalized = keyword.strip()
                if not normalized:
                    continue
                docs.append(self.nlp.make_doc(normalized))
                weights_for_type[normalized.lower()] = 1.0

            for indicator in indicators:
                normalized = indicator.strip()
                if not normalized:
                    continue
                docs.append(self.nlp.make_doc(normalized))
                weights_for_type[normalized.lower()] = 2.0

            if docs:
                matcher.add(doc_type_enum.value, docs)
                pattern_weights[doc_type_enum] = weights_for_type
                total_possible_scores[doc_type_enum] = sum(weights_for_type.values())

        return matcher, pattern_weights, total_possible_scores

    def _classify_document(self, text: str) -> Tuple[DocumentType, float, List[str]]:
        """Run phrase matching and score document type candidates."""
        if not self.total_possible_scores:
            return DocumentType.UNKNOWN, 0.0, []

        doc = self.nlp(text)
        matches = self.matcher(doc)

        scores: Dict[DocumentType, float] = {
            doc_type: 0.0 for doc_type in self.total_possible_scores
        }
        found_keywords: Dict[DocumentType, set[str]] = {
            doc_type: set() for doc_type in self.total_possible_scores
        }

        for match_id, start, end in matches:
            rule_id_str = self.nlp.vocab.strings[match_id]
            doc_type_enum = DocumentType(rule_id_str)
            span = doc[start:end]
            span_key = span.text.lower()

            weight = self.pattern_weights.get(doc_type_enum, {}).get(span_key, 1.0)
            scores[doc_type_enum] += weight
            found_keywords[doc_type_enum].add(span.text)

        normalized_scores = {
            doc_type: (
                score / self.total_possible_scores[doc_type]
                if self.total_possible_scores[doc_type] > 0
                else 0.0
            )
            for doc_type, score in scores.items()
        }

        if not normalized_scores:
            return DocumentType.UNKNOWN, 0.0, []

        best_type_enum = max(normalized_scores, key=normalized_scores.get)
        best_score = normalized_scores[best_type_enum]

        if best_score < self.settings.classification.min_confidence:
            return DocumentType.UNKNOWN, best_score, []

        return best_type_enum, best_score, sorted(found_keywords[best_type_enum])