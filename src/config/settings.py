"""
Centralized configuration management for Intelli-Script.

Loads configuration from environment variables and an optional .env file,
then exposes typed and validated settings for the OCR pipeline.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class OCRSettings(BaseSettings):
    """OCR engine configuration."""

    engine: str = Field(default="tesseract")
    language: str = Field(default="eng")
    psm: int = Field(default=6, ge=0, le=13)
    oem: int = Field(default=3, ge=0, le=3)
    confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    tesseract_path: Optional[str] = None
    lang_data_path: Optional[str] = None


class ImageEnhancementSettings(BaseSettings):
    """Image preprocessing configuration."""

    denoise: bool = True
    threshold: bool = True
    deskew: bool = True
    auto_rotate: bool = True
    target_resolution_dpi: int = Field(default=300, ge=100, le=600)
    resize_width: int = Field(default=2000, ge=100)
    contrast_enhancement: bool = True
    brightness_adjustment: bool = False
    denoise_h: int = Field(default=10, ge=0)
    clahe_clip_limit: float = Field(default=2.0, gt=0.0)
    clahe_tile_grid_size: int = Field(default=8, ge=1)
    adaptive_thresh_block_size: int = Field(default=11, ge=3)
    adaptive_thresh_c: int = 2

    @field_validator("adaptive_thresh_block_size")
    @classmethod
    def validate_block_size(cls, value: int) -> int:
        """Ensure adaptive threshold block size is a positive odd number."""
        if value % 2 == 0:
            raise ValueError("adaptive_thresh_block_size must be odd")
        return value


class PDFTableSettings(BaseSettings):
    """PDF table extraction configuration."""

    vertical_strategy: str = "text"
    horizontal_strategy: str = "text"
    snap_tolerance: int = Field(default=3, ge=0)


class SpaCySettings(BaseSettings):
    """spaCy NLP model configuration."""

    model: str = "en_core_web_sm"


class PHIScrubbingSettings(BaseSettings):
    """Protected health information scrubbing configuration."""

    enabled: bool = True


class ClassificationSettings(BaseSettings):
    """Document classification configuration."""

    enabled: bool = True
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    model_path: Optional[str] = None
    use_keyword_matching: bool = True
    keyword_weight: float = Field(default=0.3, ge=0.0, le=1.0)


class ExtractionSettings(BaseSettings):
    """NLP entity extraction configuration."""

    enabled: bool = True
    model_path: Optional[str] = None
    min_entity_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    extract_relations: bool = True
    context_window_chars: int = Field(default=200, ge=0)


class ValidationSettings(BaseSettings):
    """Data validation configuration."""

    enabled: bool = True
    strict_mode: bool = False
    validate_units: bool = True
    validate_ranges: bool = True
    flag_suspicious: bool = True
    confidence_penalty_on_error: float = Field(default=0.2, ge=0.0, le=1.0)


class SummarizationSettings(BaseSettings):
    """Report summarization configuration."""

    enabled: bool = True
    include_layman_terms: bool = True
    max_summary_length: int = Field(default=1000, ge=100)
    readability_level: str = "general_public"
    include_key_findings: bool = True
    include_recommendations: bool = True


class LoggingSettings(BaseSettings):
    """Application logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = "intelliscript.log"
    log_to_file: bool = True
    log_to_console: bool = True
    max_file_size_mb: int = Field(default=10, ge=1)
    backup_count: int = Field(default=5, ge=1)
    perf_log_file: str = "performance.log"
    perf_max_file_size_mb: int = Field(default=10, ge=1)
    perf_backup_count: int = Field(default=3, ge=1)

    @field_validator("level")
    @classmethod
    def normalize_level(cls, value: str) -> str:
        """Normalize and validate the configured logging level."""
        normalized = value.upper()
        allowed_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

        if normalized not in allowed_levels:
            raise ValueError(
                f"level must be one of: {', '.join(sorted(allowed_levels))}"
            )

        return normalized


class StorageSettings(BaseSettings):
    """Relative storage paths resolved against the application base directory."""

    inputs_dir: Path = Path("data/inputs")
    outputs_dir: Path = Path("data/outputs")
    temp_dir: Path = Path("data/temp")
    cache_dir: Path = Path("data/cache")
    datasets_dir: Path = Path("datasets")
    logs_dir: Path = Path("logs")
    models_dir: Path = Path("models")


class DatabaseSettings(BaseSettings):
    """Optional database configuration."""

    enabled: bool = False
    provider: str = "supabase"
    url: Optional[str] = None
    api_key: Optional[SecretStr] = None
    timeout_seconds: int = Field(default=10, ge=1)
    connection_pool_size: int = Field(default=5, ge=1)

    @model_validator(mode="after")
    def validate_database_configuration(self) -> "DatabaseSettings":
        """Require database connection details when the database is enabled."""
        if self.enabled:
            if not self.url:
                raise ValueError(
                    "DATABASE__URL is required when the database is enabled"
                )

            if not self.api_key:
                raise ValueError(
                    "DATABASE__API_KEY is required when the database is enabled"
                )

        return self


class PipelineSettings(BaseSettings):
    """Overall pipeline configuration."""

    name: str = "Intelli-Script"
    version: str = "1.0.0"
    debug: bool = False
    timeout_seconds: int = Field(default=300, ge=30)
    max_workers: int = Field(default=4, ge=1, le=32)
    enable_caching: bool = True
    save_intermediate: bool = False
    cleanup_temp_files: bool = True
    supported_file_types: List[str] = Field(
        default_factory=lambda: [
            ".pdf",
            ".jpg",
            ".jpeg",
            ".png",
            ".tiff",
            ".bmp",
            ".txt",
            ".docx",
        ]
    )
    output_indent: int = Field(default=2, ge=0)
    max_file_size_mb: int = Field(default=100, ge=1)
    validate_phi: bool = True
    active_stages: List[str] = Field(
        default_factory=lambda: [
            "preprocessing",
            "ocr",
            "phi_scrubbing",
            "classification",
            "extraction",
            "validation",
            "summarization",
            "packaging",
        ]
    )


class Settings(BaseSettings):
    """Master configuration for the Intelli-Script pipeline."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    base_dir: Path = Field(
        default=Path("."),
        alias="APP_BASE_DIR",
        description="Base project directory",
    )

    ocr: OCRSettings = Field(default_factory=OCRSettings)
    preprocessing: ImageEnhancementSettings = Field(
        default_factory=ImageEnhancementSettings
    )
    pdf_table: PDFTableSettings = Field(default_factory=PDFTableSettings)
    spacy: SpaCySettings = Field(default_factory=SpaCySettings)
    phi_scrubbing: PHIScrubbingSettings = Field(
        default_factory=PHIScrubbingSettings
    )
    classification: ClassificationSettings = Field(
        default_factory=ClassificationSettings
    )
    extraction: ExtractionSettings = Field(default_factory=ExtractionSettings)
    validation: ValidationSettings = Field(default_factory=ValidationSettings)
    summarization: SummarizationSettings = Field(
        default_factory=SummarizationSettings
    )
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)

    @model_validator(mode="after")
    def derive_absolute_paths(self) -> "Settings":
        """Resolve all storage paths against the application base directory."""
        self.base_dir = self.base_dir.expanduser().resolve()

        if not self.base_dir.exists():
            raise ValueError(
                f"APP_BASE_DIR does not exist: {self.base_dir}"
            )

        if not self.base_dir.is_dir():
            raise ValueError(
                f"APP_BASE_DIR is not a directory: {self.base_dir}"
            )

        self.storage.inputs_dir = (
            self.base_dir / self.storage.inputs_dir
        ).resolve()
        self.storage.outputs_dir = (
            self.base_dir / self.storage.outputs_dir
        ).resolve()
        self.storage.temp_dir = (
            self.base_dir / self.storage.temp_dir
        ).resolve()
        self.storage.cache_dir = (
            self.base_dir / self.storage.cache_dir
        ).resolve()
        self.storage.datasets_dir = (
            self.base_dir / self.storage.datasets_dir
        ).resolve()
        self.storage.logs_dir = (
            self.base_dir / self.storage.logs_dir
        ).resolve()
        self.storage.models_dir = (
            self.base_dir / self.storage.models_dir
        ).resolve()

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings instance."""
    return Settings()