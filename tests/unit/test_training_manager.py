from unittest.mock import MagicMock
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from trnbl.training_manager import TrainingManager
from trnbl.loggers.local import LocalLogger

# Temporary directory for testing
TEMP_PATH = Path("tests/_temp")
TEMP_PATH.mkdir(parents=True, exist_ok=True)


# Define a simple model
class SimpleModel(nn.Module):
	def __init__(self):
		super(SimpleModel, self).__init__()
		self.fc = nn.Linear(10, 1)

	def forward(self, x):
		return self.fc(x)


# Dummy dataset and dataloader
inputs = torch.randn(100, 10)
targets = torch.randn(100, 1)
dataset = TensorDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=10)

# Logger configuration
logger_config = {
	"project": "test_project",
	"metric_names": ["loss"],
	"name": "test_run",
	"train_config": {"batch_size": 10, "learning_rate": 0.001, "epochs": 10},
	"base_path": TEMP_PATH,
}

# Initialize the model, criterion, and optimizer
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)


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
