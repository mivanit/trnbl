import time
from types import TracebackType
from typing import Callable, Iterable, Type
from pathlib import Path
import warnings

# torch
try:
	import torch
except ImportError:
	warnings.warn("PyTorch not found, this might break things!")

# trnbl
from trnbl.loggers.base import TrainingLoggerBase
from trnbl.training_interval import TrainingInterval

# evaluation function should take a model and return some metrics
EvalFunction = Callable[["torch.nn.Module"], dict]


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
		dataloader: "torch.utils.data.DataLoader",
		logger: TrainingLoggerBase,
		epochs: int = 1,
		save_model: Callable[["torch.nn.Module", Path], None] = torch.save,
		# everything with intervals
		evals: Iterable[tuple[TrainingInterval | str, EvalFunction]] | None = None,
		checkpoint_interval: TrainingInterval | str = TrainingInterval(1, "epochs"),
		print_metrics_interval: TrainingInterval | str = TrainingInterval(0.1, "runs"),
		# everything with paths
		model_save_path: str = "{run_path}/checkpoints/model.checkpoint-{latest_checkpoint}.pt",
		model_save_path_special: str = "{run_path}/model.{alias}.pt",
	):
		# save start time
		self.start_time: float = time.time()
		# non path and non-interval args get copied over directly
		self.model: "torch.nn.Module" = model
		self.dataloader: "torch.utils.data.DataLoader" = dataloader
		self.logger: TrainingLoggerBase = logger
		self.epochs: int = epochs
		self.save_model: Callable[["torch.nn.Module", Path], None] = save_model

		# number of epochs, batches, and samples
		self.epochs_total: int = epochs
		self.epochs: int = 0
		self.batches_per_epoch: int = len(dataloader)
		self.batch_size: int = dataloader.batch_size
		self.batches_total: int = self.batches_per_epoch * epochs
		self.batches: int = 0
		self.samples_per_epoch: int = len(dataloader.dataset)
		self.samples_total: int = self.samples_per_epoch * epochs
		self.samples: int = 0
		self.checkpoints: int = 0

		# normalize intervals
		_batch_info_kwargs: dict[str, int] = dict(
			batches_per_epoch=self.batches_per_epoch,
			batchsize=self.batch_size,
			epochs=self.epochs_total,
		)
		self.checkpoint_interval: int = TrainingInterval.process_to_batches(
			interval=checkpoint_interval,
			**_batch_info_kwargs,
		)
		self.print_metrics_interval: int = TrainingInterval.process_to_batches(
			interval=print_metrics_interval,
			**_batch_info_kwargs,
		)
		if evals is None:
			evals = []
		self.evals: list[tuple[int, EvalFunction]] = [
			(
				TrainingInterval.process_to_batches(interval, **_batch_info_kwargs),
				eval_fn,
			)
			for interval, eval_fn in evals
		]

		# model save paths
		self.model_save_path: str = model_save_path
		self.model_save_path_special: str = model_save_path_special

		# log this info
		logger.message(
			"initialized training manager",
			__training_manager_init__=True,
			epochs_total=epochs,
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
			latest_epoch=self.epochs,
			latest_checkpoint=self.checkpoints,
		)

	def _get_format_kwargs(self) -> dict[str, str | int | float]:
		"""keyword args for formatting model save paths. calls `TrainingManager.training_status`

		# Provides:
		- `run_path : str` - path where the run is being logged and artifacts are being saved
		- `elapsed_time : float` - the elapsed time in seconds since the start of training
		- `samples : int` - samples seen so far
		- `batches : int` - batches seen so far
		- `latest_epoch : int` - the latest epoch number
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
		# update counters
		self.batches += 1
		if samples is not None:
			self.samples += samples
		else:
			self.samples += self.batch_size

		# TODO: update progress bar

		# run evals if needed
		metrics: dict = dict()
		for interval, eval_fn in self.evals:
			if (self.batches % interval == 0) or (self.batches == self.batches_total):
				metrics.update(eval_fn(self.model))

		# log metrics & training status
		self.logger.metrics({**kwargs, **metrics, **self.training_status()})

		# TODO: print metrics if needed

		# save checkpoint if needed
		if self.batches % self.checkpoint_interval == 0:
			self._save_checkpoint()

	def epoch_update(self):
		"""call this at the end of every epoch. This function will log the completion of the epoch and update the epoch counter"""
		self.logger.message(f"completed epoch {self.epochs+1}/{self.epochs_total}")
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
