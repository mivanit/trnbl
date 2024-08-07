from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path
import warnings
import time
import random

from muutils.spinner import Spinner

# gpu utils
GPU_UTILS_AVAILABLE: bool
try:
	import GPUtil  # type: ignore[import-untyped]

	GPU_UTILS_AVAILABLE = True
except ImportError as e:
	warnings.warn(f"GPUtil not available: {e}")
	GPU_UTILS_AVAILABLE = False

# psutil
PSUTIL_AVAILABLE: bool
try:
	import psutil  # type: ignore[import-untyped]

	PSUTIL_AVAILABLE = True
except ImportError as e:
	warnings.warn(f"psutil not available: {e}")
	PSUTIL_AVAILABLE = False


VOWELS: str = "aeiou"
CONSONANTS: str = "bcdfghjklmnpqrstvwxyz"


def rand_syllabic_string(length: int = 6) -> str:
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


class LoggerSpinner(Spinner):
	"see `Spinner` for parameters. catches `update_value` and passes it to the `LocalLogger`"

	def __init__(
		self,
		*args,
		logger: "TrainingLoggerBase",
		**kwargs,
	):
		super().__init__(*args, **kwargs)
		self.logger: "TrainingLoggerBase" = logger

	def update_value(self, value: Any) -> None:
		"""update the value of the spinner and log it"""
		self.logger.message(
			message=self.message,
			spinner_value=value,
			spinner_elapsed_time=time.time() - self.start_time,
		)
		super().update_value(value)
		# the above should just be calling `self.current_value = value`

	def __enter__(self) -> "LoggerSpinner":
		self.start()
		return self

	def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
		self.stop()


class TrainingLoggerBase(ABC):
	"""Base class for training loggers"""

	@abstractmethod
	def debug(self, message: str, **kwargs) -> None:
		"""log a debug message which will be saved, but not printed"""
		pass

	@abstractmethod
	def message(self, message: str, **kwargs) -> None:
		"""log a progress message, which will be printed to stdout"""
		pass

	def warning(self, message: str, **kwargs) -> None:
		"""log a warning message, which will be printed to stderr"""
		self.message(f"WARNING: {message}", __warning__=True, **kwargs)

	def error(self, message: str, **kwargs) -> None:
		"""log an error message"""
		self.message(f"ERROR: {message}", __error__=True, **kwargs)

	@abstractmethod
	def metrics(self, data: dict[str, Any]) -> None:
		"""Log a dictionary of metrics"""
		pass

	@abstractmethod
	def artifact(
		self,
		path: Path,
		type: str,
		aliases: list[str] | None = None,
		metadata: dict | None = None,
	) -> None:
		"""log an artifact from a file"""
		pass

	@property
	@abstractmethod
	def url(self) -> str | list[str]:
		"""Get the URL for the current logging run"""
		pass

	@property
	@abstractmethod
	def run_path(self) -> Path:
		"""Get the path to the current logging run"""
		pass

	@abstractmethod
	def flush(self) -> None:
		"""Flush the logger"""
		pass

	@abstractmethod
	def finish(self) -> None:
		"""Finish logging"""
		pass

	def get_mem_usage(self) -> dict:
		mem_usage: dict = {}

		try:
			# CPU/Memory usage (if available)
			if PSUTIL_AVAILABLE:
				cpu_percent = psutil.cpu_percent()
				mem_usage["cpu/percent"] = cpu_percent

				# Memory usage
				virtual_mem = psutil.virtual_memory()
				mem_usage["ram/used"] = virtual_mem.used
				mem_usage["ram/percent"] = virtual_mem.percent

			# GPU information (if available)
			if GPU_UTILS_AVAILABLE:
				gpus = GPUtil.getGPUs()
				for gpu in gpus:
					gpu_id = gpu.id
					mem_usage[f"gpu:{gpu_id}/load"] = gpu.load
					mem_usage[f"gpu:{gpu_id}/memory_used"] = gpu.memoryUsed
					mem_usage[f"gpu:{gpu_id}/temperature"] = gpu.temperature
		except Exception as e:
			self.warning(f"Error getting memory usage: {e}")

		return mem_usage

	def spinner_task(self, **kwargs) -> LoggerSpinner:
		"Create a spinner task. kwargs are passed to `Spinner`."
		return LoggerSpinner(logger=self, **kwargs)

	# def seq_task(self, **kwargs) -> LoggerSpinner:
	# 	"Create a sequential task with progress bar. kwargs are passed to `tqdm`."
	# 	return LoggerSpinner(message=message, logger=self, **kwargs)
