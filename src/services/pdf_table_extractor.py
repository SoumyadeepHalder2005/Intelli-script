"""
Extract tables from PDF documents into a structured, type-safe format.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pdfplumber

from src.config.settings import Settings
from src.core.exceptions import CorruptedFileError, FileProcessingError
from src.core.models import PDFTable


class PDFTableExtractor:
    """Extract structured tables from PDF documents using pdfplumber."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.table_settings: Dict[str, Any] = {
            "vertical_strategy": self.settings.pdf_table.vertical_strategy,
            "horizontal_strategy": self.settings.pdf_table.horizontal_strategy,
            "snap_tolerance": self.settings.pdf_table.snap_tolerance,
        }

        self.logger.info(
            "PDFTableExtractor initialized with settings: %s",
            self.table_settings,
        )

    async def extract_tables(self, pdf_path: str) -> List[PDFTable]:
        """
        Extract tables from a PDF file asynchronously.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            A list of extracted PDFTable models.

        Raises:
            FileProcessingError: If the file does not exist or cannot be accessed.
            CorruptedFileError: If the PDF cannot be parsed.
        """
        return await asyncio.to_thread(
            self._extract_tables_sync,
            Path(pdf_path),
        )

    def _extract_tables_sync(self, pdf_path: Path) -> List[PDFTable]:
        """
        Perform synchronous PDF parsing and return extracted tables.
        """
        if not pdf_path.exists():
            raise FileProcessingError(
                f"PDF file not found: {pdf_path}",
                file_path=str(pdf_path),
            )

        extracted_tables: List[PDFTable] = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables(self.table_settings)

                    for table_data in tables:
                        if not table_data:
                            continue

                        rows: List[List[Optional[str]]] = [
                            [str(cell) if cell is not None else None for cell in row]
                            for row in table_data
                        ]

                        row_count = len(rows)
                        col_count = max((len(row) for row in rows), default=0)

                        extracted_tables.append(
                            PDFTable(
                                page_number=page.page_number,
                                rows=rows,
                                row_count=row_count,
                                col_count=col_count,
                            )
                        )

            self.logger.info(
                "Extracted %d tables from %s",
                len(extracted_tables),
                pdf_path.name,
            )
            return extracted_tables

        except FileProcessingError:
            raise
        except Exception as exc:
            self.logger.error("Failed to process PDF %s: %s", pdf_path, exc)
            raise CorruptedFileError(
                str(pdf_path),
                reason=str(exc),
            ) from exc