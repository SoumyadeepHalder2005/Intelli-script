"""
Pydantic data models for the Intelli-Script OCR pipeline.

Provides type-safe data structures for pipeline stages, domain entities,
and processing outputs.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Entity classification types."""

    TEST_RESULT = "test_result"
    VITAL_SIGN = "vital_sign"
    MEDICATION = "medication"
    DIAGNOSIS = "diagnosis"
    SYMPTOM = "symptom"
    CONDITION = "condition"
    OTHER = "other"


class DocumentType(str, Enum):
    """Supported medical document types."""

    LAB_REPORT = "lab_report"
    PRESCRIPTION = "prescription"
    PATIENT_HISTORY = "patient_history"
    DIAGNOSIS_REPORT = "diagnosis_report"
    DISCHARGE_SUMMARY = "discharge_summary"
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    """Pipeline processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    ERROR = "error"


class PipelineStage(str, Enum):
    """Pipeline stage identifiers."""

    PREPROCESSING = "preprocessing"
    OCR = "ocr"
    PHI_SCRUBBING = "phi_scrubbing"
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    VALIDATION = "validation"
    SUMMARIZATION = "summarization"
    PACKAGING = "packaging"


class FindingStatus(str, Enum):
    """Clinical status of a finding."""

    NORMAL = "normal"
    ABNORMAL = "abnormal"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class PDFTable(BaseModel):
    """Structured data for a table extracted from a PDF."""

    page_number: int = Field(..., description="Page where the table was found")
    rows: List[List[Optional[str]]] = Field(
        ...,
        description="Table rows represented as lists of cell values",
    )
    row_count: int = Field(..., ge=0, description="Number of rows")
    col_count: int = Field(..., ge=0, description="Number of columns")


class Finding(BaseModel):
    """Structured representation of a clinical finding."""

    test_name: str = Field(..., description="Name of the test or vital sign")
    value: Any = Field(..., description="Extracted value")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    status: FindingStatus = Field(..., description="Finding classification")
    reason: Optional[str] = Field(
        None,
        description="Short explanation such as 'Above normal'",
    )
    reference_range: Optional[str] = Field(
        None,
        description="Reference range such as '60-100'",
    )


class MedicationFinding(BaseModel):
    """Structured representation of an extracted medication."""

    name: str = Field(..., description="Medication name")
    dosage: Optional[str] = Field(None, description="Dosage amount")
    unit: Optional[str] = Field(None, description="Dosage unit")
    frequency: Optional[str] = Field(None, description="Usage frequency")
    confidence: float = Field(..., ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NLPEntity(BaseModel):
    """Extracted medical entity such as a test result or medication."""

    type: EntityType = Field(..., description="Entity type")
    name: str = Field(..., description="Entity name")
    value: Optional[Any] = Field(None, description="Extracted value")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="Confidence score")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional entity metadata",
    )


class OCRPage(BaseModel):
    """OCR result for a single page."""

    page_number: int = Field(..., description="Page number")
    raw_text: str = Field(..., description="Raw extracted text")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Average confidence")
    num_lines: int = Field(0, ge=0, description="Number of extracted text lines")


class DocumentClassification(BaseModel):
    """Document classification result."""

    document_type: DocumentType = Field(..., description="Classified document type")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Classification score")
    keywords_found: List[str] = Field(
        default_factory=list,
        description="Keywords used during classification",
    )


class ValidationReport(BaseModel):
    """Validation stage summary."""

    validated_at: datetime = Field(..., description="Validation timestamp")
    total_entities: int = Field(0, ge=0, description="Total entities validated")
    valid_count: int = Field(0, ge=0, description="Number of valid entities")
    invalid_count: int = Field(0, ge=0, description="Number of invalid entities")
    abnormal_values: List[Finding] = Field(
        default_factory=list,
        description="Abnormal findings",
    )
    critical_values: List[Finding] = Field(
        default_factory=list,
        description="Critical findings",
    )


class ProcessingSummary(BaseModel):
    """High-level summary of processed document output."""

    document_summary: str = Field("", description="Overall summary")
    key_findings: List[str] = Field(
        default_factory=list,
        description="Key findings",
    )
    abnormal_values: List[Finding] = Field(
        default_factory=list,
        description="Abnormal findings summary",
    )
    medications: List[MedicationFinding] = Field(
        default_factory=list,
        description="Extracted medications",
    )
    diagnoses: List[str] = Field(
        default_factory=list,
        description="Potential diagnoses",
    )
    layman_explanation: str = Field(
        "",
        description="Patient-friendly explanation",
    )


class PipelineData(BaseModel):
    """
    Main data structure passed between pipeline stages.

    This model acts as the shared processing contract for file metadata,
    OCR output, extracted entities, validation results, summaries,
    structured output, and execution metadata.
    """

    input_file_path: Path = Field(..., description="Input file path")
    file_name: str = Field(..., description="Input file name")
    file_type: str = Field(..., description="Input file extension or type")
    file_size_bytes: int = Field(0, ge=0, description="Input file size in bytes")

    status: ProcessingStatus = Field(
        default=ProcessingStatus.PENDING,
        description="Current processing status",
    )
    processing_time_ms: float = Field(0.0, ge=0.0, description="Total processing time")

    processed_image_paths: List[Path] = Field(
        default_factory=list,
        description="Paths to preprocessed images used for OCR",
    )

    ocr_pages: List[OCRPage] = Field(
        default_factory=list,
        description="Per-page OCR results",
    )
    full_text: str = Field("", description="Combined extracted text")
    scrubbed_text: str = Field(
        "",
        description="PHI-scrubbed text for downstream stages",
    )

    classification: Optional[DocumentClassification] = Field(
        None,
        description="Document classification result",
    )
    extracted_entities: List[NLPEntity] = Field(
        default_factory=list,
        description="Extracted entities",
    )

    validation_report: Optional[ValidationReport] = Field(
        None,
        description="Validation report",
    )
    summary: Optional[ProcessingSummary] = Field(
        None,
        description="Final processing summary",
    )

    structured_output: Dict[str, Any] = Field(
        default_factory=dict,
        description="Final structured output payload",
    )

    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Structured processing errors",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Processing warnings",
    )

    stage_timings: Dict[PipelineStage, float] = Field(
        default_factory=dict,
        description="Execution time in milliseconds for each stage",
    )