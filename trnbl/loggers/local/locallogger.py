import datetime
import hashlib
import json
from typing import Any
from pathlib import Path
import io
import random
import inspect

from trnbl.loggers.base import TrainingLoggerBase


class FilePaths:
	TRAIN_CONFIG: Path = Path("config.json")
	LOGGER_META: Path = Path("meta.json")
	ARTIFACTS: Path = Path("artifacts.jsonl")
	METRICS: Path = Path("metrics.jsonl")
	LOG: Path = Path("log.jsonl")
	RUNS_MANIFEST: Path = Path(
		"runs.jsonl"
	)  # relative to project path instead of run path
	RUNS_DIR: Path = Path("runs")  # relative to project path instead of run path
	ERROR_FILE: Path = Path("ERROR.txt")
	HTML_INDEX: Path = Path("index.html")
	START_SERVER: Path = Path("start_server.py")


VOWELS: str = "aeiou"
CONSONANTS: str = "bcdfghjklmnpqrstvwxyz"


def rand_syllabic_string(length: int = 6):
	"""Generate a random string of alternating consonants and vowels to use as a unique identifier

	for a length of 2n, there are about 10^{2n} possible strings

	default is 6 characters, which gives 10^6 possible strings
	"""
	string: str = ""
	for i in range(length):
		if i % 2 == 0:
			string += random.choice(CONSONANTS)
		else:
			string += random.choice(VOWELS)
	return string


class LocalLogger(TrainingLoggerBase):
	def __init__(
		self,
		project: str,
		metric_names: list[str],
		name: str = "main",
		train_config: dict | None = None,
		base_path: str | Path = Path("trnbl-logs"),
		memusage_as_metrics: bool = True,
	):
		# set up lists
		self.log_list: list[dict] = list()
		self.metrics_list: list[dict] = list()
		self.artifacts_list: list[dict] = list()

		# copy kwargs
		self.train_config: dict = train_config
		self.project: str = project
		self.name: str = name
		self.base_path: Path = Path(base_path)

		# set up id
		self._syllabic_id: str = rand_syllabic_string()
		self.run_init_timestamp: datetime.datetime = datetime.datetime.now()
		self.run_id: str = self._get_run_id()

		# set up paths
		self.project_path: Path = self.base_path / project
		self._run_path: Path = self.project_path / FilePaths.RUNS_DIR / self.run_id
		# make sure the run path doesn't already exist
		assert not self._run_path.exists()
		self._run_path.mkdir(parents=True, exist_ok=True)

		# set up files and objects for logs, artifacts, and metrics
		# ----------------------------------------

		self.log_file: io.TextIOWrapper = open(self.run_path / FilePaths.LOG, "a")

		self.metrics_file: io.TextIOWrapper = open(
			self.run_path / FilePaths.METRICS, "a"
		)

		self.artifacts_file: io.TextIOWrapper = open(
			self.run_path / FilePaths.ARTIFACTS, "a"
		)

		# metric names (getting mem usage might cause problems if we have an error)
		self.metric_names: list[str] = metric_names
		if memusage_as_metrics:
			self.metric_names += list(self.get_mem_usage().keys())

		# put everything in a config
		self.logger_meta: dict = dict(
			run_id=self.run_id,
			run_path=self.run_path.as_posix(),
			syllabic_id=self.syllabic_id,
			project=self.project,
			run_init_timestamp=str(self.run_init_timestamp.isoformat()),
			metric_names=metric_names,
			train_config=train_config,  # TODO: this duplicates the contents of FilePaths.TRAIN_CONFIG, is that ok?
		)

		# write to the project jsonl
		with open(self.project_path / FilePaths.RUNS_MANIFEST, "a") as f:
			json.dump(self.logger_meta, f)
			f.write("\n")

		# write the index.html and start_server.py files
		# ----------------------------------------
		from trnbl.loggers.local.html_frontend import get_html_frontend
		with open(self.run_path / FilePaths.HTML_INDEX, "w") as f:
			f.write(get_html_frontend())
		
		import trnbl.loggers.local.start_server as start_server_module
		with open(self.run_path / FilePaths.START_SERVER, "w") as f:
			f.write(inspect.getsource(start_server_module))

		# write init files
		# ----------------------------------------

		# logger metadata
		with open(self.run_path / FilePaths.LOGGER_META, "w") as f:
			json.dump(self.logger_meta, f, indent="\t")

		# training/model/dataset config
		with open(self.run_path / FilePaths.TRAIN_CONFIG, "w") as f:
			json.dump(train_config, f, indent="\t")

		self.message("starting logger")

	@property
	def _run_hash(self) -> str:
		return hashlib.md5(str(self.train_config).encode()).hexdigest()

	@property
	def syllabic_id(self) -> str:
		return self._syllabic_id

	def _get_run_id(self) -> str:
		return f"run-{self.name}-{self._run_hash[:5]}-{self.run_init_timestamp.strftime('%y%m%d_%H%M')}-{self.syllabic_id}"

	def get_timestamp(self) -> str:
		return datetime.datetime.now().isoformat()

	def message(self, message: str, **kwargs) -> None:
		"""log a progress message"""
		# TODO: also log messages via regular logger to stdout
		msg_dict: dict = dict(
			message=message,
			timestamp=self.get_timestamp(),
		)
		if kwargs:
			msg_dict.update(kwargs)

		self.log_list.append(msg_dict)
		self.log_file.write(json.dumps(msg_dict) + "\n")
		print(message)

	def warning(self, message: str, **kwargs) -> None:
		"""log a warning message"""
		self.message(f"WARNING: {message}", __warning__=True, **kwargs)

	def error(self, message: str, **kwargs) -> None:
		"""log an error message"""
		self.message(f"ERROR: {message}", __error__=True, **kwargs)
		with open(self.run_path / FilePaths.ERROR_FILE, "w") as f:
			f.write(message)

	def metrics(self, data: dict[str, Any]) -> None:
		"""log a dictionary of metrics"""
		data["timestamp"] = self.get_timestamp()

		self.metrics_list.append(data)
		self.metrics_file.write(json.dumps(data) + "\n")

	def artifact(
		self,
		path: Path,
		type: str,
		aliases: list[str] | None = None,
		metadata: dict | None = None,
	) -> None:
		"""log an artifact from a file"""
		artifact_dict: dict = dict(
			timestamp=self.get_timestamp(),
			path=path.as_posix(),
			type=type,
			aliases=aliases,
			metadata=metadata if metadata else {},
		)

		self.artifacts_list.append(artifact_dict)
		self.artifacts_file.write(json.dumps(artifact_dict) + "\n")

	@property
	def url(self) -> str:
		"""Get the URL for the current logging run"""
		return self.run_path.as_posix()

	@property
	def run_path(self) -> Path:
		"""Get the path to the current logging run"""
		return self._run_path

	def flush(self) -> None:
		self.log_file.flush()
		self.metrics_file.flush()
		self.artifacts_file.flush()

	def finish(self) -> None:
		self.message("closing logger")

		self.log_file.flush()
		self.log_file.close()

		self.metrics_file.flush()
		self.metrics_file.close()

		self.artifacts_file.flush()
		self.artifacts_file.close()
