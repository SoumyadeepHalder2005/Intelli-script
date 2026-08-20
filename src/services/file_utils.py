"""
File utility service for reading and writing pipeline outputs.

Provides JSON, CSV, and text output helpers with thread-offloaded I/O,
basic validation, and safer file-writing behavior.
"""

import asyncio
import csv
import json
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Union

from src.config.settings import Settings
from src.core.constants import ENCODING, JSON_ENSURE_ASCII, OUTPUT_INDENT
from src.core.exceptions import FileProcessingError


class FileUtilsService:
    """Manage file I/O operations for pipeline outputs."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = logging.getLogger(__name__)

    async def save_json_result(
        self,
        data: Dict[str, Any],
        output_dir: Union[str, Path],
        filename: str,
    ) -> str:
        """Save a dictionary as a JSON file."""
        try:
            output_path = await self._prepare_output_path(output_dir, filename)
            await asyncio.to_thread(self._write_json_file, output_path, data)
            self.logger.info("Saved JSON results to %s", output_path)
            return str(output_path)
        except Exception as exc:
            error_msg = f"Failed to save JSON result: {exc}"
            self.logger.error("%s", error_msg)
            raise FileProcessingError(error_msg, file_path=str(output_dir)) from exc

    def save_json_result_sync(
        self,
        data: Dict[str, Any],
        output_dir: Union[str, Path],
        filename: str,
    ) -> str:
        """Save a dictionary as a JSON file synchronously."""
        try:
            output_path = Path(output_dir) / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_json_file(output_path, data)
            self.logger.info("Saved JSON results to %s", output_path)
            return str(output_path)
        except Exception as exc:
            error_msg = f"Failed to save JSON result: {exc}"
            self.logger.error("%s", error_msg)
            raise FileProcessingError(error_msg, file_path=str(output_dir)) from exc

    async def save_csv_result(
        self,
        data: List[Dict[str, Any]],
        output_dir: Union[str, Path],
        filename: str,
    ) -> str:
        """Save tabular records as a CSV file."""
        try:
            output_path = await self._prepare_output_path(output_dir, filename)
            await asyncio.to_thread(self._write_csv_file, output_path, data)
            self.logger.info("Saved CSV results to %s", output_path)
            return str(output_path)
        except Exception as exc:
            error_msg = f"Failed to save CSV result: {exc}"
            self.logger.error("%s", error_msg)
            raise FileProcessingError(error_msg, file_path=str(output_dir)) from exc

    async def save_report(
        self,
        data: Dict[str, Any],
        output_dir: Union[str, Path],
        filename: Optional[str] = None,
    ) -> Dict[str, str]:
        """Save a report in all relevant output formats."""
        try:
            if filename is None:
                filename = "report"

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            results: Dict[str, str] = {}

            results["json"] = await self.save_json_result(
                data,
                output_dir,
                f"{filename}.json",
            )

            extracted_entities = data.get("extracted_entities")
            if isinstance(extracted_entities, list) and extracted_entities:
                results["csv"] = await self.save_csv_result(
                    extracted_entities,
                    output_dir,
                    f"{filename}_entities.csv",
                )

            summary = data.get("summary")
            if summary:
                results["summary"] = await self.save_text_summary(
                    summary,
                    output_dir,
                    f"{filename}_summary.txt",
                )

            self.logger.info(
                "Saved report in %d output format(s)",
                len(results),
            )
            return results

        except Exception as exc:
            error_msg = f"Failed to save report: {exc}"
            self.logger.error("%s", error_msg)
            raise FileProcessingError(error_msg, file_path=str(output_dir)) from exc

    async def save_text_summary(
        self,
        summary_data: Union[str, Dict[str, Any]],
        output_dir: Union[str, Path],
        filename: str,
    ) -> str:
        """Save a human-readable text summary."""
        try:
            output_path = await self._prepare_output_path(output_dir, filename)

            if isinstance(summary_data, dict):
                content = self._format_summary_dict(summary_data)
            else:
                content = str(summary_data)

            await asyncio.to_thread(self._write_text_file, output_path, content)
            self.logger.info("Saved text summary to %s", output_path)
            return str(output_path)

        except Exception as exc:
            error_msg = f"Failed to save text summary: {exc}"
            self.logger.error("%s", error_msg)
            raise FileProcessingError(error_msg, file_path=str(output_dir)) from exc

    async def load_json(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load and parse a JSON file."""
        path = Path(file_path)
        try:
            return await asyncio.to_thread(self._read_json_file, path)
        except Exception as exc:
            error_msg = f"Failed to load JSON file: {exc}"
            self.logger.error("%s", error_msg)
            raise FileProcessingError(error_msg, file_path=str(path)) from exc

    async def _prepare_output_path(
        self,
        output_dir: Union[str, Path],
        filename: str,
    ) -> Path:
        """Create the parent directory for an output file and return its path."""
        try:
            output_path = Path(output_dir) / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return output_path
        except Exception as exc:
            raise FileProcessingError(
                f"Failed to prepare output path: {exc}",
                file_path=str(output_dir),
            ) from exc

    def _write_json_file(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Write JSON using a temporary file and atomic replacement."""
        temp_path: Optional[Path] = None

        try:
            with NamedTemporaryFile(
                mode="w",
                encoding=ENCODING,
                dir=file_path.parent,
                prefix=f"{file_path.stem}_",
                suffix=".tmp",
                delete=False,
            ) as tmp_file:
                json.dump(
                    data,
                    tmp_file,
                    indent=OUTPUT_INDENT,
                    ensure_ascii=JSON_ENSURE_ASCII,
                )
                tmp_file.flush()
                temp_path = Path(tmp_file.name)

            temp_path.replace(file_path)

        except Exception:
            if temp_path and temp_path.exists():
                temp_path.unlink()
            raise

    def _write_csv_file(self, file_path: Path, data: List[Dict[str, Any]]) -> None:
        """Write a list of dictionaries to CSV."""
        if not data:
            return

        fieldnames = self._collect_csv_fieldnames(data)

        with file_path.open("w", newline="", encoding=ENCODING) as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(data)

    def _write_text_file(self, file_path: Path, content: str) -> None:
        """Write plain text content to a file."""
        with file_path.open("w", encoding=ENCODING) as file_obj:
            file_obj.write(content)

    def _read_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Read and parse a JSON file."""
        with file_path.open("r", encoding=ENCODING) as file_obj:
            return json.load(file_obj)

    def _collect_csv_fieldnames(self, rows: List[Dict[str, Any]]) -> List[str]:
        """Collect stable CSV fieldnames from a list of row dictionaries."""
        seen: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.append(key)
        return seen

    def _format_summary_dict(self, summary_data: Dict[str, Any]) -> str:
        """Convert structured summary data into readable plain text."""
        lines: List[str] = []

        for key, value in summary_data.items():
            if key == "document_summary":
                lines.append(f"{'=' * 80}\n{value}\n{'=' * 80}\n")

            elif key == "key_findings" and isinstance(value, list):
                lines.append("\nKEY FINDINGS:")
                for finding in value:
                    lines.append(f"  • {finding}")

            elif key == "abnormal_values" and isinstance(value, list):
                lines.append("\nABNORMAL VALUES:")
                for abnormal in value:
                    if isinstance(abnormal, dict):
                        lines.append(
                            f"  • {abnormal.get('test_name', 'Unknown')}: "
                            f"{abnormal.get('value')} {abnormal.get('unit', '')} "
                            f"({abnormal.get('reason', 'N/A')})"
                        )
                    else:
                        lines.append(f"  • {abnormal}")

            elif key == "medications" and isinstance(value, list):
                lines.append("\nMEDICATIONS:")
                for med in value:
                    if isinstance(med, dict):
                        lines.append(
                            f"  • {med.get('name', 'Unknown')}: "
                            f"{med.get('dosage', '')} {med.get('unit', '')} "
                            f"({med.get('frequency', 'N/A')})"
                        )
                    else:
                        lines.append(f"  • {med}")

            elif key == "diagnoses" and isinstance(value, list):
                lines.append("\nPOTENTIAL DIAGNOSES:")
                for diagnosis in value:
                    lines.append(f"  • {diagnosis}")

            elif key == "layman_explanation":
                lines.append(f"\nPATIENT-FRIENDLY EXPLANATION:\n{value}\n")

        return "\n".join(lines).strip() + "\n"