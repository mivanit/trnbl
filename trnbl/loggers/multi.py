from typing import Any, TypeVar
from pathlib import Path

from trnbl.loggers.base import TrainingLoggerBase

# TODO: move this to muutils
T = TypeVar("T")


def maybe_flatten(lst: list[T | list[T]]) -> list[T]:
	"""flatten a list if it is nested"""
	flat_lst: list[T] = []
	for item in lst:
		if isinstance(item, list):
			flat_lst.extend(item)
		else:
			flat_lst.append(item)
	return flat_lst


class MultiLogger(TrainingLoggerBase):
	"""use multiple loggers at once"""

	def __init__(self, loggers: list[TrainingLoggerBase]) -> None:
		self.loggers: list[TrainingLoggerBase] = loggers

	def debug(self, message: str, **kwargs) -> None:
		"""log a debug message which will be saved, but not printed"""
		for logger in self.loggers:
			logger.debug(message, **kwargs)

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
		return maybe_flatten([logger.url for logger in self.loggers])

	@property
	def run_path(self) -> list[Path]:
		"""Get the paths to the current logging run"""
		return maybe_flatten([logger.run_path for logger in self.loggers])

	def flush(self) -> None:
		"""Flush the logger"""
		for logger in self.loggers:
			logger.flush()

	def finish(self) -> None:
		"""Finish logging"""
		for logger in self.loggers:
			logger.finish()
