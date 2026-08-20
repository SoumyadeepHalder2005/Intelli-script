"""
Custom exception hierarchy for the Intelli-Script OCR pipeline.

Provides specific exception types for structured error handling, logging,
and user-facing error messages.
"""

from typing import Any, Dict, Optional


class IntelliScriptError(Exception):
    """Base exception for all Intelli-Script errors."""

    error_code: str = "UNKNOWN_ERROR"
    recoverable: bool = False

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the exception for logging or API responses."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "recoverable": self.recoverable,
        }


class ConfigurationError(IntelliScriptError):
    """Raised when configuration is invalid or missing."""

    error_code = "CONFIG_ERROR"
    recoverable = False


class SettingsValidationError(ConfigurationError):
    """Raised when settings validation fails."""


class FileProcessingError(IntelliScriptError):
    """Raised when file operations fail."""

    error_code = "FILE_ERROR"
    recoverable = True

    def __init__(
        self,
        message: str,
        file_path: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = details.copy() if details else {}
        if file_path:
            payload["file_path"] = file_path
        super().__init__(message, details=payload)


class UnsupportedFileTypeError(FileProcessingError):
    """Raised when a file type is not supported."""

    def __init__(self, file_path: str, file_type: str) -> None:
        message = f"Unsupported file type: {file_type}"
        details = {"file_type": file_type}
        super().__init__(message, file_path=file_path, details=details)


class CorruptedFileError(FileProcessingError):
    """Raised when a file is corrupted or unreadable."""

    def __init__(self, file_path: str, reason: str = "") -> None:
        message = f"File is corrupted or unreadable: {reason}"
        details = {"reason": reason}
        super().__init__(message, file_path=file_path, details=details)


class OCRProcessingError(IntelliScriptError):
    """Raised when OCR extraction fails."""

    error_code = "OCR_ERROR"
    recoverable = True


class PreprocessingError(OCRProcessingError):
    """Raised when image preprocessing fails."""


class ImageExtractionError(OCRProcessingError):
    """Raised when image extraction from a PDF fails."""

    def __init__(self, message: str, page_number: Optional[int] = None) -> None:
        details = {"page_number": page_number} if page_number is not None else {}
        super().__init__(message, details=details)


class ClassificationError(IntelliScriptError):
    """Raised when document classification fails."""

    error_code = "CLASSIFICATION_ERROR"
    recoverable = True


class DocumentTypeUnknownError(ClassificationError):
    """Raised when the document type cannot be determined."""

    def __init__(self, confidence: float = 0.0) -> None:
        message = f"Unable to determine document type (confidence: {confidence})"
        details = {"confidence": confidence}
        super().__init__(message, details=details)


class ExtractionError(IntelliScriptError):
    """Raised when entity extraction fails."""

    error_code = "EXTRACTION_ERROR"
    recoverable = True


class EntityExtractionError(ExtractionError):
    """Raised when a specific entity extraction step fails."""

    def __init__(self, entity_type: str, message: str) -> None:
        details = {"entity_type": entity_type}
        super().__init__(message, details=details)


class ValidationError(IntelliScriptError):
    """Raised when data validation fails."""

    error_code = "VALIDATION_ERROR"
    recoverable = True


class DataRangeError(ValidationError):
    """Raised when an extracted value is outside the acceptable range."""

    def __init__(
        self,
        test_name: str,
        value: float,
        valid_range: tuple,
        unit: str = "",
    ) -> None:
        message = f"{test_name}: {value} {unit} is outside valid range {valid_range}"
        details = {
            "test_name": test_name,
            "value": value,
            "valid_range": valid_range,
            "unit": unit,
        }
        super().__init__(message, details=details)


class UnitMismatchError(ValidationError):
    """Raised when the detected unit does not match the expected unit."""

    def __init__(
        self,
        test_name: str,
        detected_unit: str,
        expected_unit: str,
    ) -> None:
        message = (
            f"{test_name}: detected unit '{detected_unit}' "
            f"doesn't match expected '{expected_unit}'"
        )
        details = {
            "test_name": test_name,
            "detected_unit": detected_unit,
            "expected_unit": expected_unit,
        }
        super().__init__(message, details=details)


class DatasetServiceError(IntelliScriptError):
    """Raised when dataset service operations fail."""

    error_code = "DATASET_ERROR"
    recoverable = True


class DatasetNotFoundError(DatasetServiceError):
    """Raised when a requested dataset or reference entry is missing."""

    def __init__(self, dataset_name: str, key: str = "") -> None:
        message = f"Dataset '{dataset_name}' not found or has no entry for '{key}'"
        details = {"dataset_name": dataset_name, "key": key}
        super().__init__(message, details=details)


class PipelineExecutionError(IntelliScriptError):
    """Raised when pipeline execution fails."""

    error_code = "PIPELINE_ERROR"
    recoverable = True

    def __init__(
        self,
        message: str,
        stage: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = details.copy() if details else {}
        if stage:
            payload["stage"] = stage
        super().__init__(message, details=payload)


class StageDependencyError(PipelineExecutionError):
    """Raised when stage dependencies are not met."""

    def __init__(self, stage: str, missing_dependency: str) -> None:
        message = f"Stage '{stage}' missing required input: {missing_dependency}"
        details = {"missing_dependency": missing_dependency}
        super().__init__(message, stage=stage, details=details)


class SummarizationError(IntelliScriptError):
    """Raised when summarization fails."""

    error_code = "SUMMARIZATION_ERROR"
    recoverable = True


class ProcessingTimeoutError(IntelliScriptError):
    """Raised when processing exceeds the configured timeout."""

    error_code = "TIMEOUT_ERROR"
    recoverable = True

    def __init__(self, message: str, timeout_seconds: float) -> None:
        details = {"timeout_seconds": timeout_seconds}
        super().__init__(message, details=details)


class PackagingError(IntelliScriptError):
    """Raised when final report packaging or saving fails."""

    error_code = "PACKAGING_ERROR"
    recoverable = True