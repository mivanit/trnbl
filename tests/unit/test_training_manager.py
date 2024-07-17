from typing import Dict, Any, Union, Callable
import time
from unittest.mock import MagicMock
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytest

from trnbl.training_interval import IntervalValueError
from trnbl.training_manager import TrainingManager
from trnbl.loggers.local import LocalLogger
from trnbl.loggers.base import TrainingLoggerBase

# Temporary directory for testing
TEMP_PATH: Path = Path("tests/_temp")
TEMP_PATH.mkdir(parents=True, exist_ok=True)


# Define a simple model
class SimpleModel(nn.Module):
	def __init__(self) -> None:
		super(SimpleModel, self).__init__()
		self.fc: nn.Linear = nn.Linear(10, 1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.fc(x)


# Dummy dataset and dataloader
inputs: torch.Tensor = torch.randn(100, 10)
targets: torch.Tensor = torch.randn(100, 1)
dataset: TensorDataset = TensorDataset(inputs, targets)
dataloader: DataLoader = DataLoader(dataset, batch_size=10)

# Logger configuration
logger_config: Dict[str, Any] = {
	"project": "test_project",
	"metric_names": ["loss"],
	"name": "test_run",
	"train_config": {"batch_size": 10, "learning_rate": 0.001, "epochs": 10},
	"base_path": TEMP_PATH,
}

# Initialize the model, criterion, and optimizer
model: nn.Module = SimpleModel()
criterion: nn.Module = nn.MSELoss()
optimizer: optim.Optimizer = optim.SGD(model.parameters(), lr=0.001)


def test_training_manager_initialization():
	logger = LocalLogger(**logger_config)
	training_manager = TrainingManager(
		model=model, dataloader=dataloader, logger=logger, epochs=10
	)
	assert training_manager.model == model
	assert training_manager.dataloader == dataloader
	assert training_manager.logger == logger
	assert training_manager.epochs_total == 10
	training_manager.logger.finish()


def test_training_manager_batch_update():
	logger = LocalLogger(**logger_config)
	training_manager = TrainingManager(
		model=model, dataloader=dataloader, logger=logger, epochs=1
	)
	training_manager._save_checkpoint = MagicMock()

	# Simulate a training batch update
	inputs, targets = next(iter(dataloader))
	outputs = model(inputs)
	loss = criterion(outputs, targets)
	training_manager.batch_update(samples=len(targets), train_loss=loss.item())

	# Check if metrics were logged
	assert len(training_manager.logger.metrics_list) > 0
	assert training_manager.logger.metrics_list[-1]["train_loss"] == loss.item()

	# Check if a checkpoint was saved (based on interval)
	if training_manager.batches % training_manager.checkpoint_interval == 0:
		training_manager._save_checkpoint.assert_called()

	training_manager.logger.finish()


def test_training_manager_epoch_update():
	logger = LocalLogger(**logger_config)
	training_manager = TrainingManager(
		model=model, dataloader=dataloader, logger=logger, epochs=1
	)

	# Simulate an epoch update
	training_manager.epoch_update()
	assert training_manager.epochs == 1
	assert len(training_manager.logger.log_list) > 0
	assert training_manager.logger.log_list[-1]["message"] == "completed epoch 1/1"
	training_manager.logger.finish()


def test_training_manager_checkpoint_saving():
	logger = LocalLogger(**logger_config)
	training_manager = TrainingManager(
		model=model, dataloader=dataloader, logger=logger, epochs=1
	)
	training_manager._save_checkpoint(alias="test_checkpoint")

	# Check if the checkpoint artifact was logged
	assert len(training_manager.logger.artifacts_list) > 0
	assert "test_checkpoint" in training_manager.logger.artifacts_list[-1]["aliases"]
	training_manager.logger.finish()


@pytest.fixture
def training_manager() -> TrainingManager:
	logger: LocalLogger = LocalLogger(**logger_config)
	return TrainingManager(model=model, dataloader=dataloader, logger=logger, epochs=1)


def test_training_manager_initialization_comprehensive(
	training_manager: TrainingManager,
) -> None:
	assert isinstance(training_manager.model, nn.Module)
	assert isinstance(training_manager.dataloader, DataLoader)
	assert isinstance(training_manager.logger, TrainingLoggerBase)
	assert training_manager.epochs_total == 1
	assert training_manager.epochs == 0
	assert training_manager.batches_per_epoch == len(dataloader)
	assert training_manager.batch_size == dataloader.batch_size
	assert training_manager.batches_total == len(dataloader)
	assert training_manager.batches == 0
	assert training_manager.samples_per_epoch == len(dataloader.dataset)
	assert training_manager.samples_total == len(dataloader.dataset)
	assert training_manager.samples == 0
	assert training_manager.checkpoints == 0


def test_training_manager_enter(training_manager: TrainingManager) -> None:
	with training_manager as tm:
		assert tm == training_manager


def test_training_manager_exit_normal(training_manager: TrainingManager) -> None:
	training_manager._save_checkpoint = MagicMock()
	training_manager.logger.finish = MagicMock()

	with training_manager:
		pass

	training_manager._save_checkpoint.assert_called_with(alias="final")
	training_manager.logger.finish.assert_called_once()


def test_training_manager_exit_exception(training_manager: TrainingManager) -> None:
	training_manager._save_checkpoint = MagicMock()
	training_manager.logger.error = MagicMock()
	training_manager.logger.finish = MagicMock()

	with pytest.raises(ValueError):
		with training_manager:
			raise ValueError("Test exception")

	training_manager._save_checkpoint.assert_called_with(alias="exception")
	training_manager.logger.error.assert_called_once()
	training_manager.logger.finish.assert_called_once()


def test_training_manager_get_elapsed_time(training_manager: TrainingManager) -> None:
	start_time: float = time.time()
	training_manager.start_time = start_time
	training_manager.get_elapsed_time()


def test_training_manager_training_status(training_manager: TrainingManager) -> None:
	status: Dict[str, Union[int, float]] = training_manager.training_status()
	assert all(
		key in status
		for key in [
			"elapsed_time",
			"samples",
			"batches",
			"latest_epoch",
			"latest_checkpoint",
		]
	)
	assert all(isinstance(value, (int, float)) for value in status.values())


def test_training_manager_get_format_kwargs(training_manager: TrainingManager) -> None:
	kwargs: Dict[str, Union[str, int, float]] = training_manager._get_format_kwargs()
	assert all(
		key in kwargs
		for key in [
			"run_path",
			"elapsed_time",
			"samples",
			"batches",
			"latest_epoch",
			"latest_checkpoint",
		]
	)
	assert isinstance(kwargs["run_path"], str)
	assert all(
		isinstance(value, (int, float))
		for key, value in kwargs.items()
		if key != "run_path"
	)


def test_training_manager_batch_update_new(training_manager: TrainingManager) -> None:
	training_manager._save_checkpoint = MagicMock()

	initial_samples: int = training_manager.samples
	initial_batches: int = training_manager.batches

	training_manager.batch_update(samples=10, loss=0.5)

	assert training_manager.samples == initial_samples + 10
	assert training_manager.batches == initial_batches + 1
	assert len(training_manager.logger.metrics_list) > 0
	assert training_manager.logger.metrics_list[-1]["loss"] == 0.5

	if training_manager.batches % training_manager.checkpoint_interval == 0:
		training_manager._save_checkpoint.assert_called_once()


def test_training_manager_epoch_update_new(training_manager: TrainingManager) -> None:
	initial_epochs: int = training_manager.epochs

	training_manager.epoch_update()

	assert training_manager.epochs == initial_epochs + 1
	assert len(training_manager.logger.log_list) > 0
	assert (
		f"completed epoch {initial_epochs+1}/{training_manager.epochs_total}"
		in training_manager.logger.log_list[-1]["message"]
	)


def test_training_manager_save_checkpoint(training_manager: TrainingManager) -> None:
	initial_checkpoints: int = training_manager.checkpoints

	training_manager._save_checkpoint(alias="test_checkpoint")

	assert training_manager.checkpoints == initial_checkpoints + 1
	assert len(training_manager.logger.artifacts_list) > 0
	assert "test_checkpoint" in training_manager.logger.artifacts_list[-1]["aliases"]


def test_training_manager_full_training_loop() -> None:
	logger: LocalLogger = LocalLogger(**logger_config)
	training_manager: TrainingManager = TrainingManager(
		model=model,
		dataloader=dataloader,
		logger=logger,
		epochs=2,
		checkpoint_interval="1 epochs",
		evals=[
			("1 epochs", lambda m: {"eval_loss": criterion(m(inputs), targets).item()})
		],
	)

	with training_manager as tm:
		for epoch in range(2):
			for batch_inputs, batch_targets in dataloader:
				outputs: torch.Tensor = model(batch_inputs)
				loss: torch.Tensor = criterion(outputs, batch_targets)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				tm.batch_update(samples=len(batch_targets), train_loss=loss.item())

			tm.epoch_update()

	assert tm.epochs_total == 2
	assert tm.batches == 20  # 2 epochs * 10 batches per epoch
	assert tm.samples == 200  # 2 epochs * 100 samples per epoch
	assert tm.checkpoints == 3  # 1 checkpoint per epoch
	assert len(tm.logger.metrics_list) == 20  # 1 metric log per batch
	assert (
		len(tm.logger.log_list) >= 4
	)  # At least 1 log for init, 2 for epoch completions, 1 for training complete
	assert (
		len(tm.logger.artifacts_list) == 3
	)  # 2 epoch checkpoints + 1 final checkpoint


def test_training_manager_zero_epochs() -> None:
	logger: LocalLogger = LocalLogger(**logger_config)
	with pytest.warns(IntervalValueError):
		TrainingManager(model=model, dataloader=dataloader, logger=logger, epochs=0)


def test_training_manager_negative_epochs() -> None:
	logger: LocalLogger = LocalLogger(**logger_config)
	with pytest.warns(IntervalValueError):
		TrainingManager(model=model, dataloader=dataloader, logger=logger, epochs=-1)


def test_training_manager_custom_save_model() -> None:
	logger: LocalLogger = LocalLogger(**logger_config)
	custom_save_model: Callable[[nn.Module, Path], None] = MagicMock()
	training_manager: TrainingManager = TrainingManager(
		model=model,
		dataloader=dataloader,
		logger=logger,
		epochs=1,
		save_model=custom_save_model,
	)
	training_manager._save_checkpoint()
	custom_save_model.assert_called_once()


def test_training_manager_custom_intervals() -> None:
	logger: LocalLogger = LocalLogger(**logger_config)
	training_manager: TrainingManager = TrainingManager(
		model=model,
		dataloader=dataloader,
		logger=logger,
		epochs=1,
		checkpoint_interval="0.5 epochs",
		print_metrics_interval="0.25 epochs",
		evals=[("0.1 epochs", lambda m: {"eval_loss": 0.5})],
	)
	assert training_manager.checkpoint_interval == len(dataloader) // 2
	assert training_manager.print_metrics_interval == len(dataloader) // 4
	assert training_manager.evals[0][0] == len(dataloader) // 10


def test_training_manager_custom_model_save_paths() -> None:
	logger: LocalLogger = LocalLogger(**logger_config)
	custom_path: str = "{run_path}/custom_checkpoints/model-{latest_checkpoint}.pt"
	custom_special_path: str = "{run_path}/custom_special/model-{alias}.pt"
	training_manager: TrainingManager = TrainingManager(
		model=model,
		dataloader=dataloader,
		logger=logger,
		epochs=1,
		model_save_path=custom_path,
		model_save_path_special=custom_special_path,
	)
	assert training_manager.model_save_path == custom_path
	assert training_manager.model_save_path_special == custom_special_path


def test_training_manager_batch_update_no_samples() -> None:
	logger: LocalLogger = LocalLogger(**logger_config)
	training_manager: TrainingManager = TrainingManager(
		model=model, dataloader=dataloader, logger=logger, epochs=1
	)
	initial_samples: int = training_manager.samples
	training_manager.batch_update(samples=None, loss=0.5)
	assert training_manager.samples == initial_samples + training_manager.batch_size


def test_training_manager_multiple_evals() -> None:
	logger: LocalLogger = LocalLogger(**logger_config)
	eval1: Callable[[nn.Module], Dict[str, float]] = lambda m: {"eval1": 0.5}  # noqa: E731
	eval2: Callable[[nn.Module], Dict[str, float]] = lambda m: {"eval2": 0.7}  # noqa: E731
	training_manager: TrainingManager = TrainingManager(
		model=model,
		dataloader=dataloader,
		logger=logger,
		epochs=1,
		evals=[("1 batch", eval1), ("2 batches", eval2)],
	)
	training_manager.batch_update(samples=10, loss=0.3)
	assert "eval1" in training_manager.logger.metrics_list[-1]
	assert "eval2" not in training_manager.logger.metrics_list[-1]
	training_manager.batch_update(samples=10, loss=0.3)
	assert "eval1" in training_manager.logger.metrics_list[-1]
	assert "eval2" in training_manager.logger.metrics_list[-1]


@pytest.mark.parametrize(
	"interval, expected",
	[
		("1 epochs", len(dataloader)),
		("0.5 epochs", len(dataloader) // 2),
		("10 batches", 10),
		("10 samples", 1),
	],
)
def test_training_manager_interval_processing(interval: str, expected: int) -> None:
	logger: LocalLogger = LocalLogger(**logger_config)
	training_manager: TrainingManager = TrainingManager(
		model=model,
		dataloader=dataloader,
		logger=logger,
		epochs=1,
		checkpoint_interval=interval,
	)
	assert training_manager.checkpoint_interval == expected


def test_training_manager_empty_dataloader() -> None:
	empty_dataloader: DataLoader = DataLoader(
		TensorDataset(torch.Tensor([]), torch.Tensor([])), batch_size=1
	)
	logger: LocalLogger = LocalLogger(**logger_config)
	with pytest.raises(ValueError):
		TrainingManager(
			model=model, dataloader=empty_dataloader, logger=logger, epochs=1
		)


def test_training_manager_0_batchsize() -> None:
	with pytest.raises(ValueError):
		empty_dataloader: DataLoader = DataLoader(
			TensorDataset(torch.Tensor([]), torch.Tensor([])), batch_size=0
		)
		logger: LocalLogger = LocalLogger(**logger_config)
		TrainingManager(
			model=model, dataloader=empty_dataloader, logger=logger, epochs=1
		)
