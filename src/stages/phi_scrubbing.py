"""
PHI scrubbing stage.

Current implementation is a pass-through placeholder that prepares the
pipeline boundary for future PHI redaction logic.
"""

from typing import Any

from src.config.settings import Settings
from src.core.models import PipelineData
from src.stages.base import PipelineStage


class PHIScrubbingStage(PipelineStage):
    """Populate scrubbed_text from OCR output, with optional future PHI scrubbing."""

    REQUIRED_INPUTS = ["full_text"]

    def __init__(
        self,
        settings: Settings,
        stage_key: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(settings=settings, stage_key=stage_key, **kwargs)
        self.scrubbing_enabled = self.settings.phi_scrubbing.enabled

        if self.scrubbing_enabled:
            self.logger.info(
                "PHI scrubbing stage enabled in pass-through mode"
            )
        else:
            self.logger.warning(
                "PHI scrubbing stage disabled; raw text will pass downstream"
            )

    async def process(self, data: PipelineData) -> PipelineData:
        """Populate scrubbed_text for downstream stages."""
        if not data.full_text:
            data.scrubbed_text = data.full_text
            return data

        if not self.scrubbing_enabled:
            data.scrubbed_text = data.full_text
            return data

        self.logger.debug(
            "PHI scrubbing placeholder active; no redaction performed"
        )
        data.scrubbed_text = data.full_text
        return data