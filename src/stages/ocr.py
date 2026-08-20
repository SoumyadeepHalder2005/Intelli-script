"""
OCR stage for extracting text from preprocessed image files.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import pytesseract

from src.config.settings import Settings
from src.core.exceptions import OCRProcessingError
from src.core.models import OCRPage, PipelineData
from src.stages.base import PipelineStage


class OCRStage(PipelineStage):
    """Extract text from image files using a single Tesseract data pass."""

    REQUIRED_INPUTS = ["processed_image_paths"]

    def __init__(
        self,
        settings: Settings,
        stage_key: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(settings=settings, stage_key=stage_key, **kwargs)

    async def process(self, data: PipelineData) -> PipelineData:
        """Extract OCR text from processed image paths."""
        try:
            if not data.processed_image_paths:
                raise OCRProcessingError("No processed images to OCR")

            self.logger.info(
                "Processing %d image(s) with OCR",
                len(data.processed_image_paths),
            )

            config = self._build_tesseract_config()

            tasks = [
                self._extract_page(img_path, page_index, config)
                for page_index, img_path in enumerate(data.processed_image_paths)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            pages: List[OCRPage] = []

            for page_index, result in enumerate(results, start=1):
                if isinstance(result, Exception):
                    self.logger.error(
                        "OCR extraction failed on page %d: %s",
                        page_index,
                        result,
                    )
                    if isinstance(result, OCRProcessingError):
                        raise result
                    raise OCRProcessingError(f"Page {page_index}: {result}")

                pages.append(result)

            if not pages:
                raise OCRProcessingError("OCR extraction failed for all pages")

            data.ocr_pages = sorted(pages, key=lambda page: page.page_number)
            data.full_text = "\n\n--- Page Break ---\n\n".join(
                page.raw_text for page in data.ocr_pages
            )

            avg_confidence = (
                sum(page.confidence for page in pages) / len(pages)
                if pages
                else 0.0
            )

            self.logger.info(
                "OCR complete: %d pages, avg confidence %.1f%%, %d total lines",
                len(pages),
                avg_confidence * 100,
                sum(page.num_lines for page in pages),
            )

            return data

        except OCRProcessingError:
            raise
        except Exception as exc:
            self.logger.exception("OCR processing failed: %s", exc)
            raise OCRProcessingError(f"OCR processing failed: {exc}") from exc

    async def _extract_page(
        self,
        img_path: Path,
        page_index: int,
        config: str,
    ) -> OCRPage:
        """Extract OCR text from a single image asynchronously."""
        try:
            raw_text, avg_confidence, num_lines = await asyncio.to_thread(
                self._run_ocr_on_image,
                img_path,
                config,
            )

            page_num = page_index + 1

            self.logger.debug(
                "Page %d: confidence %.1f%%, lines %d, text length %d chars",
                page_num,
                avg_confidence * 100,
                num_lines,
                len(raw_text),
            )

            return OCRPage(
                page_number=page_num,
                raw_text=raw_text,
                confidence=avg_confidence,
                num_lines=num_lines,
            )

        except Exception as exc:
            self.logger.error(
                "Page %d (%s) extraction failed: %s",
                page_index + 1,
                img_path.name,
                exc,
                exc_info=True,
            )
            raise OCRProcessingError(f"Page {page_index + 1}: {exc}") from exc

    def _run_ocr_on_image(self, img_path: Path, config: str) -> Tuple[str, float, int]:
        """Load an image and run one Tesseract pass using image_to_data."""
        if not img_path.exists():
            raise OCRProcessingError(f"Image file not found: {img_path}")

        img = cv2.imread(str(img_path))
        if img is None or img.size == 0:
            raise OCRProcessingError(f"Invalid or unreadable image file: {img_path}")

        data_dict = pytesseract.image_to_data(
            img,
            config=config,
            output_type=pytesseract.Output.DICT,
        )

        raw_text = self._reconstruct_text_from_data(data_dict)

        if not raw_text.strip():
            self.logger.warning("Page (%s): no text extracted", img_path.name)
            return "", 0.0, 0

        confidences = []
        for conf in data_dict["conf"]:
            try:
                conf_value = float(conf)
            except (TypeError, ValueError):
                continue
            if conf_value >= 0:
                confidences.append(conf_value)

        num_lines = len(
            {
                (block, par, line)
                for block, par, line in zip(
                    data_dict["block_num"],
                    data_dict["par_num"],
                    data_dict["line_num"],
                )
                if int(block) > 0
            }
        )

        avg_confidence_normalized = 0.0
        valid_confs = [conf for conf in confidences if conf > 0]
        if valid_confs:
            avg_confidence = sum(valid_confs) / len(valid_confs)
            avg_confidence_normalized = min(1.0, avg_confidence / 100.0)

        return raw_text, avg_confidence_normalized, num_lines

    def _reconstruct_text_from_data(self, data: Dict[str, Any]) -> str:
        """Reconstruct readable text from Tesseract word-level output."""
        lines: List[str] = []
        current_line: List[str] = []
        last_block, last_par, last_line = -1, -1, -1

        for index, text in enumerate(data["text"]):
            text = text.strip()
            if not text:
                continue

            block = data["block_num"][index]
            par = data["par_num"][index]
            line = data["line_num"][index]

            if (block, par, line) != (last_block, last_par, last_line):
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [text]

                if (block, par) != (last_block, last_par) and lines and lines[-1] != "":
                    lines.append("")
            else:
                current_line.append(text)

            last_block, last_par, last_line = block, par, line

        if current_line:
            lines.append(" ".join(current_line))

        cleaned_lines: List[str] = []
        previous_blank = False
        for line in lines:
            is_blank = line == ""
            if is_blank and previous_blank:
                continue
            cleaned_lines.append(line)
            previous_blank = is_blank

        return "\n".join(cleaned_lines).strip()

    def _build_tesseract_config(self) -> str:
        """Build the Tesseract configuration string."""
        return " ".join(
            [
                f"--psm {self.settings.ocr.psm}",
                f"--oem {self.settings.ocr.oem}",
            ]
        )