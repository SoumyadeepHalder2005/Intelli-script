"""
Abstract base class for all pipeline stages.
"""

from abc import ABC, abstractmethod
import logging
from typing import Any, List

from src.config.settings import Settings
from src.core.models import PipelineData


class PipelineStage(ABC):
    """Abstract base class for a single pipeline processing stage."""

    REQUIRED_INPUTS: List[str] = []

    def __init__(self, settings: Settings, stage_key: str, **kwargs: Any) -> None:
        super().__init__()
        self.settings = settings
        self.stage_key = stage_key
        self.logger = logging.getLogger(__name__)

    @property
    def name(self) -> str:
        """Return the stage name derived from the class name."""
        return self.__class__.__name__

    @abstractmethod
    async def process(self, data: PipelineData) -> PipelineData:
        """Execute the primary logic of this stage."""
        raise NotImplementedError