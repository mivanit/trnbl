import time
from types import TracebackType
from typing import Any, Callable, Iterable, Type, TypeVar, Generator
from pathlib import Path
import warnings

import tqdm

# torch
try:
	import torch
except ImportError:
	warnings.warn("PyTorch not found, this might break things!")

# trnbl
from trnbl.loggers.base import TrainingLoggerBase
from trnbl.training_interval import TrainingInterval, CastableToTrainingInterval

# evaluation function should take a model and return some metrics
EvalFunction = Callable[["torch.nn.Module"], dict]


class TrainingManagerInitError(Exception):
	pass


T = TypeVar("T")


def wrapped_iterable(
	iterable: Iterable[T],
	manager: "TrainingManager",
	is_epoch: bool = False,
	use_tqdm: bool | None = None,
	tqdm_kwargs: dict[str, Any] | None = None,
) -> Generator[T, None, None]:
	length: int = len(iterable)

	# update the manager if it's not fully initialized
	# ------------------------------------------------------------
	if not manager.init_complete:
		if is_epoch:
			# if epoch loop, set the total epochs
			manager.epochs_total = length
		else:
			# if batch loop, set other things
			manager.batches_per_epoch = length
			try:
				manager.batch_size = iterable.batch_size
				manager.samples_per_epoch = len(iterable.dataset)
			except AttributeError as e:
				raise TrainingManagerInitError(
					"could not get the batch size or dataset size from the dataloader passed to `TrainingManager().batch_loop()`. ",
					"pass either a `torch.utils.data.DataLoader` ",
					"or an iterable with a `batch_size: int` attribute and a `dataset: Iterable` attribute.",
				) from e

		# try to compute counters and finish init of TrainingManager
		manager.try_compute_counters()

	# set up progress bar with tqdm
	# ------------------------------------------------------------
	use_tqdm = (
		use_tqdm
		if use_tqdm is not None  # do what the user says
		else is_epoch  # otherwise, use tqdm if we are the epoch loop
	)

	if use_tqdm:
		# tqdm kwargs with defaults
		_tqdm_kwargs: dict[str, Any] = dict(
			desc="training run"
			if is_epoch
			else f"epoch {manager.epochs+1}/{manager.epochs_total}",
			unit=" epochs" if is_epoch else " batches",
			total=length,
		)
		if tqdm_kwargs is not None:
			_tqdm_kwargs.update(tqdm_kwargs)

		# wrap with tqdm
		iterable = tqdm.tqdm(iterable, **_tqdm_kwargs)

	# yield the items, and update the manager
	# ------------------------------------------------------------
	for item in iterable:
		yield item
		if is_epoch:
			manager.epoch_update()
		# no need to call batch_update, since the user has to call batch_update to log metrics


class TrainingManager:
	"""context manager for training a model, with logging, evals, and checkpoints

	# Parameters:
	- `model : torch.nn.Module`
	    ref to model being trained - used for saving checkpoints
	- `dataloader : torch.utils.data.DataLoader`
	    ref to dataloader being used - used for calculating training progress
	- `logger : TrainingLoggerBase`
	    logger, which can be local or interface with wandb
	- `epochs : int`
	    number of epochs to train for
	    (defaults to `1`)
	- `evals : Iterable[tuple[TrainingInterval | str, EvalFunction]] | None`
	    list of pairs of (interval, eval_fn) to run evals on the model. See `TrainingInterval` for interval options.
	    (defaults to `None`)
	- `checkpoint_interval : TrainingInterval | str`
	    interval at which to save model checkpoints
	    (defaults to `TrainingInterval(1, "epochs")`)
	- `print_metrics_interval : TrainingInterval | str`
	    interval at which to print metrics
	    (defaults to `TrainingInterval(0.1, "runs")`)
	- `save_model : Callable[[torch.nn.Module, Path], None]`
	    function to save the model (defaults to `torch.save`)
	    (defaults to `torch.save`)
	- `model_save_path : str`
	    format string for saving model checkpoints. uses `_get_format_kwargs` for formatting, along with an `alias` kwarg
	    (defaults to `"{run_path}/checkpoints/model.checkpoint-{latest_checkpoint}.pt"`)
	- `model_save_path_special : str`
	    format string for saving special model checkpoints (final, exception, etc.). uses `_get_format_kwargs` for formatting, along with an `alias` kwarg
	    (defaults to `"{run_path}/model.{alias}.pt"`)

	# Usage:
	```python
	with TrainingManager(
	    model=model, dataloader=TRAIN_LOADER, logger=logger, epochs=500,
	    evals={
	        "1 epochs": eval_func,
	        "0.1 runs": lambda model: logger.get_mem_usage(),
	    }.items(),
	    checkpoint_interval="50 epochs",
	) as tp:

	    # Training loop
	    model.train()
	    for epoch in range(epochs):
	        for inputs, targets in TRAIN_LOADER:
	            # the usual
	            optimizer.zero_grad()
	            outputs = model(inputs)
	            loss = criterion(outputs, targets)
	            loss.backward()
	            optimizer.step()

	            # compute accuracy
	            accuracy = torch.sum(torch.argmax(outputs, dim=1) == targets).item() / len(targets)

	            # log metrics
	            tp.batch_update(
	                # pass in number of samples in your batch (or it will be inferred from the batch size)
	                samples=len(targets),
	                # any other metrics you compute every loop
	                **{"train/loss": loss.item(), "train/acc": accuracy},
	            )
	            # batch_update will automatically run evals and save checkpoints as needed

	        tp.epoch_update()
	```

	"""

	def __init__(
		self,
		model: "torch.nn.Module",
		logger: TrainingLoggerBase,
		# required if you don't wrap the loops
		dataloader: "torch.utils.data.DataLoader|None" = None,
		epochs_total: int | None = None,
		save_model: Callable[["torch.nn.Module", Path], None] = torch.save,
		# everything with intervals
		evals: Iterable[tuple[CastableToTrainingInterval, EvalFunction]] | None = None,
		checkpoint_interval: CastableToTrainingInterval = TrainingInterval(1, "epochs"),
		print_metrics_interval: CastableToTrainingInterval = TrainingInterval(
			0.1, "runs"
		),
		# everything with paths
		model_save_path: str = "{run_path}/checkpoints/model.checkpoint-{latest_checkpoint}.pt",
		model_save_path_special: str = "{run_path}/model.{alias}.pt",
	):
		# save start time
		self.start_time: float = time.time()
		# non path and non-interval args get copied over directly
		self.model: "torch.nn.Module" = model
		self.logger: TrainingLoggerBase = logger
		self.save_model: Callable[["torch.nn.Module", Path], None] = save_model

		self.logger.message("starting training manager initialization")

		# model save paths
		self.model_save_path: str = model_save_path
		self.model_save_path_special: str = model_save_path_special

		# temp intervals for processing later in `try_compute_counters`
		self._evals: Iterable[tuple[TrainingInterval, EvalFunction]]
		if evals is None:
			self._evals = []
		else:
			self._evals = [
				(TrainingInterval.from_any(interval), eval_fn)
				for interval, eval_fn in evals
			]
		self._checkpoint_interval: TrainingInterval = TrainingInterval.from_any(
			checkpoint_interval
		)
		self._print_metrics_interval: TrainingInterval = TrainingInterval.from_any(
			print_metrics_interval
		)

		self.evals: list[tuple[int, EvalFunction]] | None = None
		self.checkpoint_interval: int | None = None
		self.print_metrics_interval: int | None = None

		# counters for epochs, batches, samples, and checkpoints
		self.epochs: int = 0
		self.batches: int = 0
		self.samples: int = 0
		self.checkpoints: int = 0

		# total numbers of epochs, batches, and samples
		# pass via init kwarg or wrapped epochs loop
		self.epochs_total: int | None = epochs_total
		# from dataloader or dataloader in wrapped loop
		self.batches_per_epoch: int | None = None
		self.batch_size: int | None = None
		self.samples_per_epoch: int | None = None
		# computed dynamically from the above
		self.batches_total: int | None = None
		self.samples_total: int | None = None

		# whether the init is finished
		self.init_complete: bool = False

		# if we have a dataloader, we can compute some of the above
		if dataloader is not None:
			self.batches_per_epoch = len(dataloader)
			self.batch_size = dataloader.batch_size
			self.samples_per_epoch = len(dataloader.dataset)

		self.try_compute_counters()

	def try_compute_counters(self) -> None:
		# we depend on either the TrainingManager init or the wrapped loops
		# getting the epochs_total and dataloader
		# everything else is computed dynamically

		if any(
			x is None
			for x in [
				self.epochs_total,
				self.batches_per_epoch,
				self.batch_size,
				self.samples_per_epoch,
			]
		):
			# if we don't have all the info we need, return early
			return

		self.batches_total: int = self.batches_per_epoch * self.epochs_total
		self.samples_total: int = self.samples_per_epoch * self.epochs_total

		# check if the dataloader has a finite nonzero length
		if self.samples_per_epoch == 0:
			raise TrainingManagerInitError(
				f"Dataloader has no samples. Please provide a dataloader with a non-zero length. {self.samples_per_epoch = }"
			)

		if self.batches_per_epoch == 0:
			raise TrainingManagerInitError(
				f"Dataloader has no batches. Please provide a dataloader with a non-zero length. {self.batches_per_epoch = }"
			)

		if self.batch_size == 0:
			raise TrainingManagerInitError(
				f"Dataloader has a batch size of 0. Please provide a dataloader with a non-zero batch size. {self.batch_size = }"
			)

		# normalize intervals for checkpoints, metrics printing, and evals
		_batch_info_kwargs: dict[str, int] = dict(
			batches_per_epoch=self.batches_per_epoch,
			batchsize=self.batch_size,
			epochs=self.epochs_total,
		)
		self.checkpoint_interval: int = TrainingInterval.process_to_batches(
			interval=self._checkpoint_interval,
			**_batch_info_kwargs,
		)
		self.print_metrics_interval: int = TrainingInterval.process_to_batches(
			interval=self._print_metrics_interval,
			**_batch_info_kwargs,
		)
		self.evals: list[tuple[int, EvalFunction]] = [
			(
				TrainingInterval.process_to_batches(interval, **_batch_info_kwargs),
				eval_fn,
			)
			for interval, eval_fn in self._evals
		]

		# log this info
		self.init_complete = True
		self.logger.message(
			"initialized training manager",
			__training_manager_init__=True,
			epochs_total=self.epochs_total,
			batches_per_epoch=self.batches_per_epoch,
			batch_size=self.batch_size,
			samples_per_epoch=self.samples_per_epoch,
			samples_total=self.samples_total,
			checkpoint_interval_batches=self.checkpoint_interval,
			print_metrics_interval_batches=self.print_metrics_interval,
			model_save_path=self.model_save_path,
			model_save_path_special=self.model_save_path_special,
			**self.training_status(),
		)

	def __enter__(self):
		return self

	def __exit__(self, exc_type: Type, exc_val: Exception, exc_tb: TracebackType):
		# if error
		if exc_type is not None:
			# add exception info
			self.logger.error(
				str(exc_val),
				exc_type=str(exc_type),
				exc_val=str(exc_val),
				exc_tb=str(exc_tb),
			)
			# save the model
			self._save_checkpoint(alias="exception")

			# close the logger
			self.logger.finish()
		else:
			# if no error, log and save the final model
			self.logger.message(
				"training complete",
				__complete__=True,
			)
			self._save_checkpoint(alias="final")
			self.logger.finish()

	def epoch_loop(
		self,
		epochs: Iterable[int],
		use_tqdm: bool = True,
		**tqdm_kwargs,
	) -> Generator[int, None, None]:
		return wrapped_iterable(
			iterable=epochs,
			manager=self,
			is_epoch=True,
			use_tqdm=use_tqdm,
			tqdm_kwargs=tqdm_kwargs,
		)

	def batch_loop(
		self,
		batches: Iterable[int],
		use_tqdm: bool = False,
		**tqdm_kwargs,
	) -> Generator[int, None, None]:
		return wrapped_iterable(
			iterable=batches,
			manager=self,
			is_epoch=False,
			use_tqdm=use_tqdm,
			tqdm_kwargs=tqdm_kwargs,
		)

	def check_is_initialized(self):
		if not self.init_complete:
			raise TrainingManagerInitError(
				"TrainingManager not correctly initialized. ",
				"This is likely due to failing to specify the epoch count, or failing to specify batch size/count. "
				"you must either wrap your epoch loop with `TrainingManager.epoch_loop` or specify `epochs_total`",
				"AND you must either wrap your batch loop with `TrainingManager.batch_loop` or pass a `torch.utils.data.DataLoader` to the TrainingManager constructor.",
				"please note, if not wrapping the epoch loop, you must also call `TrainingManager.epoch_update` at the end of each epoch.",
			)

	def get_elapsed_time(self) -> float:
		"""return the elapsed time in seconds since the start of training"""
		return time.time() - self.start_time

	def training_status(self) -> dict[str, int | float]:
		"""status of elapsed time, samples, batches, epochs, and checkpoints"""
		return dict(
			# timestamp handled in logger
			elapsed_time=self.get_elapsed_time(),
			samples=self.samples,
			batches=self.batches,
			epochs=self.epochs,
			latest_checkpoint=self.checkpoints,
		)

	def _get_format_kwargs(self) -> dict[str, str | int | float]:
		"""keyword args for formatting model save paths. calls `TrainingManager.training_status`

		# Provides:
		- `run_path : str` - path where the run is being logged and artifacts are being saved
		- `elapsed_time : float` - the elapsed time in seconds since the start of training
		- `samples : int` - samples seen so far
		- `batches : int` - batches seen so far
		- `epochs : int` - the latest epoch number
		- `latest_checkpoint : int` - the latest checkpoint number

		"""
		return {
			"run_path": self.logger.run_path.as_posix(),
			**self.training_status(),
		}

	def batch_update(self, samples: int | None, **kwargs):
		"""call this at the end of every batch. Pass `samples` or it will be inferred from the batch size, and any other metrics as kwargs

		This function will:
		- update internal counters
		- run evals as needed (based on the intervals passed)
		- log all metrics and training status
		- save a checkpoint as needed (based on the checkpoint interval)


		"""
		# check init is finished
		if not self.init_complete:
			self.try_compute_counters()
			self.check_is_initialized()

		# update counters
		self.batches += 1
		if samples is not None:
			self.samples += samples
		else:
			self.samples += self.batch_size

		# run evals if needed
		metrics: dict = dict()
		for interval, eval_fn in self.evals:
			if (self.batches % interval == 0) or (self.batches == self.batches_total):
				metrics.update(eval_fn(self.model))

		# log metrics & training status
		self.logger.metrics({**kwargs, **metrics, **self.training_status()})

		# print metrics if needed

		# save checkpoint if needed
		if self.batches % self.checkpoint_interval == 0:
			self._save_checkpoint()

	def epoch_update(self):
		"""call this at the end of every epoch. This function will log the completion of the epoch and update the epoch counter"""
		self.logger.debug(f"completed epoch {self.epochs+1}/{self.epochs_total}")
		self.epochs += 1

	def _save_checkpoint(self, alias: str | None = None):
		"""wrapper for saving checkpoint as an artifact to the logger and incrementing the checkpoint counter"""
		# if no alias, then it's a regular checkpoint
		no_alias: bool = alias is None
		if no_alias:
			alias = f"checkpoint-{self.checkpoints}"

		# TODO: store training hist with model?

		# put together a path
		checkpoint_path: Path
		if no_alias:
			# format the model save path for a normal checkpoint
			checkpoint_path = Path(
				self.model_save_path.format(
					**self._get_format_kwargs(),
					alias=alias,
				)
			)
		else:
			# for a special checkpoint, use the special path
			checkpoint_path = Path(
				self.model_save_path_special.format(
					**self._get_format_kwargs(),
					alias=alias,
				)
			)

		# make sure directory exists
		checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

		# save the model
		self.save_model(self.model, checkpoint_path)

		# log the checkpoint as an artifact
		self.logger.artifact(
			checkpoint_path,
			"model",
			metadata=self.training_status(),
			aliases=[alias] if alias else None,
		)

		# increment checkpoint counter
		self.checkpoints += 1
