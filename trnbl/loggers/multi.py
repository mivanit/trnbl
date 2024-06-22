from typing import Any
from pathlib import Path

from trnbl.loggers.base import TrainingLoggerBase


class MultiLogger(TrainingLoggerBase):
	"""use multiple loggers at once"""

	def __init__(self, loggers: list[TrainingLoggerBase]) -> None:
		self.loggers: list[TrainingLoggerBase] = loggers

	def message(self, message: str, **kwargs) -> None:
		"""log a progress message"""
		for logger in self.loggers:
			logger.message(message, **kwargs)

	def metrics(self, data: dict[str, Any]) -> None:
		"""Log a dictionary of metrics"""
		for logger in self.loggers:
			logger.metrics(data)

	def artifact(
		self,
		path: Path,
		type: str,
		aliases: list[str] | None = None,
		metadata: dict | None = None,
	) -> None:
		"""log an artifact from a file"""
		for logger in self.loggers:
			logger.artifact(path=path, type=type, aliases=aliases, metadata=metadata)

	@property
	def url(self) -> list[str]:
		"""Get the URL for the current logging run"""
		return [logger.url for logger in self.loggers]

	@property
	def run_path(self) -> list[Path]:
		"""Get the path to the current logging run"""
		return [logger.run_path for logger in self.loggers]

	def flush(self) -> None:
		"""Flush the logger"""
		for logger in self.loggers:
			logger.flush()

	def finish(self) -> None:
		"""Finish logging"""
		for logger in self.loggers:
			logger.finish()
