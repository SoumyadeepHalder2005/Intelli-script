"""
Image preprocessing stage with multi-format input support.

Loads documents, converts them into OCR-friendly images, enhances them,
and stores processed page images on disk for downstream OCR.
"""

import asyncio
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np
from PIL import Image

from src.config.settings import Settings
from src.core.exceptions import PreprocessingError, UnsupportedFileTypeError
from src.core.models import PipelineData
from src.stages.base import PipelineStage


try:
    import pymupdf

    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


try:
    import pdf2image

    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False


class PreprocessingStage(PipelineStage):
    """Preprocess images for OCR with support for PDFs and common image formats."""

    REQUIRED_INPUTS = ["input_file_path"]

    def __init__(
        self,
        settings: Settings,
        stage_key: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(settings=settings, stage_key=stage_key, **kwargs)

        if HAS_PYMUPDF:
            self.logger.info("PyMuPDF available for PDF processing")
        elif HAS_PDF2IMAGE:
            self.logger.info("pdf2image available as PDF fallback")
        else:
            self.logger.warning(
                "No PDF backend available; PDF processing will fail"
            )

    async def process(self, data: PipelineData) -> PipelineData:
        """Load, enhance, and save page images for OCR."""
        try:
            images = await self._load_images(data.input_file_path, data.file_type)

            if not images:
                raise PreprocessingError(
                    f"No images could be extracted from {data.file_name}. "
                    f"File type: {data.file_type}"
                )

            self.logger.info(
                "Extracted %d image(s) from %s",
                len(images),
                data.file_name,
            )

            processed_paths: List[Path] = []
            temp_dir = self.settings.storage.temp_dir
            temp_dir.mkdir(parents=True, exist_ok=True)

            for index, img in enumerate(images, start=1):
                output_path = temp_dir / f"{data.file_name}_page_{index}.png"

                try:
                    enhanced = await asyncio.to_thread(self._enhance_image, img)

                    write_ok = await asyncio.to_thread(
                        cv2.imwrite,
                        str(output_path),
                        enhanced,
                    )
                    if not write_ok:
                        raise PreprocessingError(
                            f"Failed to write processed image: {output_path}"
                        )

                    processed_paths.append(output_path)

                    self.logger.debug(
                        "Processed image %d/%d and saved to %s",
                        index,
                        len(images),
                        output_path.name,
                    )

                except Exception as exc:
                    self.logger.warning(
                        "Failed to enhance image %d: %s",
                        index,
                        exc,
                    )

            if not processed_paths:
                raise PreprocessingError("All images failed to process")

            data.processed_image_paths = processed_paths
            return data

        except PreprocessingError:
            raise
        except Exception as exc:
            self.logger.exception("Preprocessing failed: %s", exc)
            raise PreprocessingError(f"Preprocessing failed: {exc}") from exc

    async def _load_images(self, file_path: Path, file_type: str) -> List[np.ndarray]:
        """Load images from supported document formats."""
        if not file_path.exists():
            raise PreprocessingError(f"File not found: {file_path}")

        normalized_type = file_type.lower().lstrip(".")

        try:
            if normalized_type == "pdf":
                return await self._load_pdf(file_path)
            if normalized_type in {"tiff", "tif"}:
                return await self._load_tiff(file_path)
            if normalized_type in {"jpg", "jpeg", "png", "bmp", "webp"}:
                return await self._load_image(file_path)

            raise UnsupportedFileTypeError(str(file_path), normalized_type)

        except (PreprocessingError, UnsupportedFileTypeError):
            raise
        except Exception as exc:
            raise PreprocessingError(
                f"Failed to load {normalized_type} file: {exc}"
            ) from exc

    async def _load_pdf(self, file_path: Path) -> List[np.ndarray]:
        """Load PDF pages using PyMuPDF or pdf2image."""
        if HAS_PYMUPDF:
            try:
                return await asyncio.to_thread(self._load_pdf_pymupdf, file_path)
            except Exception as exc:
                self.logger.warning(
                    "PyMuPDF failed for %s: %s; trying pdf2image",
                    file_path.name,
                    exc,
                )

        if HAS_PDF2IMAGE:
            try:
                return await asyncio.to_thread(self._load_pdf_pdf2image, file_path)
            except Exception as exc:
                self.logger.error("pdf2image failed for %s: %s", file_path.name, exc)
                raise PreprocessingError(
                    "PDF conversion failed with all backends"
                ) from exc

        raise PreprocessingError(
            "No PDF library available. Install PyMuPDF or pdf2image."
        )

    def _load_pdf_pymupdf(self, file_path: Path) -> List[np.ndarray]:
        """Load PDF pages using PyMuPDF."""
        images: List[np.ndarray] = []
        document = pymupdf.open(file_path)

        try:
            dpi = self.settings.preprocessing.target_resolution_dpi

            for page_index, page in enumerate(document, start=1):
                pix = page.get_pixmap(dpi=dpi, alpha=False)

                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height,
                    pix.width,
                    pix.n,
                )

                if pix.n == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

                images.append(img)
                self.logger.debug(
                    "Loaded PDF page %d/%d",
                    page_index,
                    len(document),
                )

        finally:
            document.close()

        return images

    def _load_pdf_pdf2image(self, file_path: Path) -> List[np.ndarray]:
        """Load PDF pages using pdf2image."""
        dpi = self.settings.preprocessing.target_resolution_dpi
        pil_images = pdf2image.convert_from_path(str(file_path), dpi=dpi)

        images: List[np.ndarray] = []
        for pil_img in pil_images:
            img = np.array(pil_img)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            images.append(img)

        return images

    async def _load_tiff(self, file_path: Path) -> List[np.ndarray]:
        """Load TIFF file with multi-page support."""
        return await asyncio.to_thread(self._load_tiff_sync, file_path)

    def _load_tiff_sync(self, file_path: Path) -> List[np.ndarray]:
        """Synchronously load TIFF pages."""
        images: List[np.ndarray] = []

        try:
            with Image.open(file_path) as img:
                page_count = getattr(img, "n_frames", 1)

                for page_index in range(page_count):
                    if page_count > 1:
                        img.seek(page_index)

                    frame = np.array(img)
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    elif len(frame.shape) == 3 and frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

                    images.append(frame)
                    self.logger.debug(
                        "Loaded TIFF page %d/%d",
                        page_index + 1,
                        page_count,
                    )

            return images

        except Exception as exc:
            self.logger.warning(
                "PIL TIFF loading failed for %s, trying OpenCV: %s",
                file_path.name,
                exc,
            )
            img_cv = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
            if img_cv is not None:
                return [img_cv]

            raise PreprocessingError(f"Failed to load TIFF: {exc}") from exc

    async def _load_image(self, file_path: Path) -> List[np.ndarray]:
        """Load a standard raster image."""
        return await asyncio.to_thread(self._load_image_sync, file_path)

    def _load_image_sync(self, file_path: Path) -> List[np.ndarray]:
        """Synchronously load a standard image with PIL fallback."""
        img = cv2.imread(str(file_path), cv2.IMREAD_COLOR)

        if img is None:
            try:
                with Image.open(file_path) as pil_img:
                    img = np.array(pil_img)

                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif len(img.shape) == 3 and img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            except Exception as exc:
                raise PreprocessingError(f"Failed to load image: {exc}") from exc

        return [img]

    def _enhance_image(self, img: np.ndarray) -> np.ndarray:
        """Enhance image quality for OCR."""
        if img is None or img.size == 0:
            raise PreprocessingError("Invalid image data")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        settings = self.settings.preprocessing

        height, width = gray.shape[:2]
        if width > settings.resize_width:
            scale = settings.resize_width / width
            new_height = int(height * scale)
            gray = cv2.resize(
                gray,
                (settings.resize_width, new_height),
                interpolation=cv2.INTER_AREA,
            )

        if settings.deskew:
            gray = self._deskew(gray)

        if settings.denoise:
            gray = cv2.fastNlMeansDenoising(gray, h=settings.denoise_h)

        if settings.contrast_enhancement:
            clahe = cv2.createCLAHE(
                clipLimit=settings.clahe_clip_limit,
                tileGridSize=(
                    settings.clahe_tile_grid_size,
                    settings.clahe_tile_grid_size,
                ),
            )
            gray = clahe.apply(gray)

        if settings.threshold:
            gray = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                settings.adaptive_thresh_block_size,
                settings.adaptive_thresh_c,
            )

        return gray

    def _deskew(self, img: np.ndarray) -> np.ndarray:
        """Deskew an image using line-angle estimation."""
        try:
            edges = cv2.Canny(img, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

            if lines is None or len(lines) == 0:
                return img

            angles = [np.degrees(theta) - 90 for rho, theta in lines[:, 0]]
            median_angle = float(np.median(angles))

            if abs(median_angle) <= 0.5:
                return img

            height, width = img.shape[:2]
            center = (width // 2, height // 2)
            matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)

            return cv2.warpAffine(
                img,
                matrix,
                (width, height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )

        except Exception as exc:
            self.logger.warning("Deskew failed: %s", exc)
            return img