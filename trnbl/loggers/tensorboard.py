import datetime
from typing import Any
from pathlib import Path
import json
import hashlib

from torch.utils.tensorboard import SummaryWriter

from trnbl.loggers.base import TrainingLoggerBase, rand_syllabic_string


class TensorBoardLogger(TrainingLoggerBase):
	def __init__(
		self,
		log_dir: str | Path,
		train_config: dict | None = None,
		name: str | None = None,
		**kwargs,
	) -> None:
		train_config_json: str = json.dumps(train_config, indent="\t")

		if name is None:
			_run_hash: str = hashlib.md5(train_config_json.encode()).hexdigest()
			name = f"h{_run_hash[:5]}-{datetime.datetime.now().strftime('%y%m%d_%H%M')}-{rand_syllabic_string()}"

		log_dir = (Path(log_dir) / name).as_posix()

		# Initialize the TensorBoard SummaryWriter with the specified log directory
		self._writer: SummaryWriter = SummaryWriter(log_dir=log_dir, **kwargs)

		# Store the run path
		self._run_path: Path = Path(log_dir)

		# Initialize the global step counter
		self._global_step: int = 0

		# Log the training configuration
		self._self_writer_add_text("config", train_config_json)
		self._self_writer_add_text("name", name)
		with open(self._run_path / "config.json", "w") as f:
			f.write(train_config_json)

	def _self_writer_add_text(self, tag: str, message: str, **kwargs) -> None:
		self._writer.add_text(
			tag,
			message + ("" if not kwargs else "\n" + json.dumps(kwargs, indent=4)),
			global_step=self._global_step,
		)

	def debug(self, message: str, **kwargs) -> None:
		self._self_writer_add_text("debug", message, **kwargs)

	def message(self, message: str, **kwargs) -> None:
		self._self_writer_add_text("message", message, **kwargs)

		# Also print the message
		print(message)

	def metrics(self, data: dict[str, Any]) -> None:
		# Log a dictionary of metrics using add_scalar in TensorBoard
		for key, value in data.items():
			self._writer.add_scalar(key, value, global_step=self._global_step)

		# Increment the global step counter
		self._global_step += 1

	def artifact(
		self,
		path: Path,
		type: str,
		aliases: list[str] | None = None,
		metadata: dict | None = None,
	) -> None:
		# Log an artifact file using add_artifact in TensorBoard
		self._writer.add_text(
			tag="artifact",
			text_string=json.dumps(
				dict(
					timestamp=datetime.datetime.now().isoformat(),
					path=path.as_posix(),
					type=type,
					aliases=aliases,
					metadata=metadata if metadata else {},
				)
			),
			global_step=self._global_step,
		)

	@property
	def url(self) -> str:
		# Return the command to launch TensorBoard with the specified log directory
		return f"tensorboard --logdir={self._run_path}"

	@property
	def run_path(self) -> Path:
		# Return the run path
		return self._run_path

	def flush(self) -> None:
		self._writer.flush()

	def finish(self) -> None:
		# Flush and close the TensorBoard SummaryWriter
		self._writer.flush()
		self._writer.close()
