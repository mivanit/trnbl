import datetime
import json
import logging
from typing import Any
from pathlib import Path
import sys

import wandb
from wandb.sdk.wandb_run import Run, Artifact

from trnbl.loggers.base import TrainingLoggerBase


class WandbLogger(TrainingLoggerBase):
	"""wrapper around wandb logging for `TrainingLoggerBase`. create using `WandbLogger.create(config, project, job_type)`"""

	def __init__(self, run: Run):
		self._run: Run = run

	@classmethod
	def create(
		cls,
		config: dict,
		project: str | None = None,
		job_type: str | None = None,
		logging_fmt: str = "%(asctime)s %(levelname)s %(message)s",
		logging_datefmt: str = "%Y-%m-%d %H:%M:%S",
		wandb_kwargs: dict | None = None,
	) -> "WandbLogger":
		logging.basicConfig(
			stream=sys.stdout,
			level=logging.INFO,
			format=logging_fmt,
			datefmt=logging_datefmt,
		)

		run: Run  # type: ignore[return-value]
		run = wandb.init(
			config=config,
			project=project,
			job_type=job_type,
			**(wandb_kwargs if wandb_kwargs else {}),
		)

		assert run is not None, f"wandb.init returned None: {wandb_kwargs}"

		logger: WandbLogger = WandbLogger(run)
		# TODO: why are we ignoring type checking here?
		logger.progress(f"{config =}") # type: ignore[attr-defined]
		return logger

	def debug(self, message: str, **kwargs) -> None:
		if kwargs:
			message += f" {kwargs =}"
		logging.debug(message)

	def message(self, message: str, **kwargs) -> None:
		if kwargs:
			message += f" {kwargs =}"
		logging.info(message)

	def metrics(self, data: dict[str, Any]) -> None:
		self._run.log(data)

	def artifact(
		self,
		path: Path,
		type: str,
		aliases: list[str] | None = None,
		metadata: dict | None = None,
	) -> None:
		artifact: Artifact = wandb.Artifact(name=self._run.id, type=type)
		artifact.add_file(str(path))
		if metadata:
			artifact.description = json.dumps(
				dict(
					timestamp=datetime.datetime.now().isoformat(),
					path=path.as_posix(),
					type=type,
					aliases=aliases,
					metadata=metadata if metadata else {},
				)
			)
		self._run.log_artifact(artifact, aliases=aliases)

	@property
	def url(self) -> str:
		# TODO: get_url returns `None` for offline runs. need to adjust allowed return types in superclass
		return str(self._run.get_url())

	@property
	def run_path(self) -> Path:
		return Path(self._run._get_path())

	def flush(self) -> None:
		self._run.save()

	def finish(self) -> None:
		"""Finish logging"""
		self._run.finish()
