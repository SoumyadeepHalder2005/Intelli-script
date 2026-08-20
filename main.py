"""
Main entry point for the Intelli-Script OCR pipeline.

Handles CLI parsing, application startup, dependency checks,
service wiring, and document processing.
"""

import asyncio
import logging
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional

import pytesseract

from src.config.settings import Settings, get_settings
from src.core.exceptions import (
    ConfigurationError,
    IntelliScriptError,
    UnsupportedFileTypeError,
)
from src.core.logging_config import setup_logging
from src.core.models import PipelineData, ProcessingStatus
from src.core.pipeline_manager import PipelineManager
from src.services.dataset_service import DatasetService
from src.services.pdf_table_extractor import PDFTableExtractor
from src.services.storage_service import StorageService


def check_dependencies() -> None:
    """Run pre-flight dependency checks."""
    logger = logging.getLogger(__name__)
    logger.info("Running dependency checks...")

    try:
        pytesseract.get_tesseract_version()
        logger.info("Tesseract OCR engine found.")
    except Exception as exc:
        logger.critical("Tesseract executable not found: %s", exc)
        raise ConfigurationError(
            "Tesseract not found. Please install it and ensure it is available in PATH."
        ) from exc


def create_directories(settings: Settings) -> None:
    """Create required runtime directories."""
    logger = logging.getLogger(__name__)
    logger.info("Ensuring required directories exist...")

    directories: List[Path] = [
        settings.storage.logs_dir,
        settings.storage.temp_dir,
        settings.storage.outputs_dir,
        settings.storage.cache_dir,
    ]

    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug("Ensured directory exists: %s", directory)
        except OSError as exc:
            logger.error("Failed to create directory %s: %s", directory, exc)
            raise ConfigurationError(
                f"Failed to create required directory: {directory}"
            ) from exc


def setup_application(settings: Settings) -> PipelineManager:
    """Initialize shared services and wire the pipeline manager."""
    logger = logging.getLogger(__name__)

    check_dependencies()
    create_directories(settings)

    logger.info("Instantiating shared services...")
    dataset_service = DatasetService(settings)
    storage_service = StorageService(settings)
    pdf_table_extractor = PDFTableExtractor(settings)

    logger.info("Instantiating pipeline manager...")
    manager = PipelineManager(
        settings=settings,
        dataset_service=dataset_service,
        storage_service=storage_service,
        pdf_table_extractor=pdf_table_extractor,
    )

    logger.info("Application setup complete.")
    return manager


async def process_file(manager: PipelineManager, input_path: str) -> PipelineData:
    """Process a single document through the pipeline."""
    logger = logging.getLogger(__name__)

    try:
        return await manager.process_document(input_path=input_path)
    except (IntelliScriptError, FileNotFoundError) as exc:
        logger.error("Pipeline error: %s", exc)
        raise
    except Exception as exc:
        logger.error("Unexpected error: %s", exc, exc_info=True)
        raise


def build_parser() -> ArgumentParser:
    """Create the CLI argument parser."""
    parser = ArgumentParser(description="Intelli-Script OCR Document Processor")
    parser.add_argument("-i", "--input", required=True, help="Path to input file")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output directory (overrides configured output directory)",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def main() -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        settings: Optional[Settings] = get_settings()

        if args.debug:
            settings.logging.level = "DEBUG"

        setup_logging(settings)
        logger = logging.getLogger(__name__)

        if args.output:
            settings.storage.outputs_dir = Path(args.output).resolve()

        manager = setup_application(settings)

        if args.output:
            create_directories(settings)

        logger.info("Starting Intelli-Script v%s", settings.pipeline.version)
        logger.info("Input: %s", args.input)
        logger.info("Output: %s", settings.storage.outputs_dir)

        result_data: PipelineData = asyncio.run(
            process_file(
                manager=manager,
                input_path=str(args.input),
            )
        )

        if result_data.status == ProcessingStatus.SUCCESS:
            output_path = result_data.structured_output.get("json_path", "N/A")
            logger.info("Processing completed successfully.")
            logger.info("Results saved to: %s", output_path)
            return 0

        logger.error("Processing failed. See errors below.")
        for error in result_data.errors:
            logger.error(
                "- [STAGE: %s] %s",
                error.get("stage", "N/A"),
                error.get("message", "Unknown error"),
            )
        return 1

    except (ConfigurationError, FileNotFoundError, UnsupportedFileTypeError) as exc:
        if logging.getLogger().hasHandlers():
            logging.critical("Startup or input error: %s", exc)
        else:
            print(f"FATAL ERROR: {exc}", file=sys.stderr)
        return 1

    except KeyboardInterrupt:
        if logging.getLogger().hasHandlers():
            logging.warning("Processing interrupted by user.")
        else:
            print("Processing interrupted by user.", file=sys.stderr)
        return 130

    except Exception as exc:
        if logging.getLogger().hasHandlers():
            logging.exception("Fatal unexpected error: %s", exc)
        else:
            print(f"FATAL ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())