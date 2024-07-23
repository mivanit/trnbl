import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from trnbl.training_manager import TrainingManager
from trnbl.loggers.local import LocalLogger

DATASET_LEN: int = 50
BATCH_SIZE: int = 10
N_EPOCHS: int = 5


class Model(nn.Module):
	def __init__(self) -> None:
		super(Model, self).__init__()
		self.fc: nn.Linear = nn.Linear(1, 1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.fc(x)


class MockedDataset(torch.utils.data.Dataset):
	def __init__(
		self,
		length: int,
		channels: int = 2,
	) -> None:
		self.dataset = torch.randn(length, channels, 1)

	def __getitem__(self, idx: int):
		return self.dataset[idx][0], self.dataset[idx][1]

	def __len__(self):
		return len(self.dataset)


def test_tm_integration_epoch_wrapped_batch_wrapped():
	model = Model()
	optimizer = optim.SGD(model.parameters(), lr=0.1)
	criterion = nn.MSELoss()

	logger = LocalLogger(
		project="integration-tests",
		metric_names=["train/loss", "train/acc", "val/loss", "val/acc"],
		train_config=dict(
			model=str(model),
			dataset="dummy",
			optimizer=str(optimizer),
			criterion=str(criterion),
		),
		base_path="tests/_temp",
	)

	train_loader: DataLoader = DataLoader(
		MockedDataset(DATASET_LEN), batch_size=BATCH_SIZE
	)

	with TrainingManager(
		model=model,
		logger=logger,
		evals={
			"1 epochs": lambda model: {"wgt_mean": torch.mean(model.fc.weight).item()},
			"1/2 epochs": lambda model: logger.get_mem_usage(),
		}.items(),
		checkpoint_interval="2 epochs",
	) as tr:
		# Training loop
		for epoch in tr.epoch_loop(range(N_EPOCHS)):
			for inputs, targets in tr.batch_loop(train_loader):
				optimizer.zero_grad()
				outputs = model(inputs)
				loss = criterion(outputs, targets)
				loss.backward()
				optimizer.step()

				accuracy = torch.sum(
					torch.argmax(outputs, dim=1) == targets
				).item() / len(targets)

				tr.batch_update(
					samples=len(targets),
					**{"train/loss": loss.item(), "train/acc": accuracy},
				)


def test_tm_integration_epoch_wrapped_batch_explicit():
	model = Model()
	optimizer = optim.SGD(model.parameters(), lr=0.1)
	criterion = nn.MSELoss()

	logger = LocalLogger(
		project="integration-tests",
		metric_names=["train/loss", "train/acc", "val/loss", "val/acc"],
		train_config=dict(
			model=str(model),
			dataset="dummy",
			optimizer=str(optimizer),
			criterion=str(criterion),
		),
		base_path="tests/_temp",
	)

	train_loader: DataLoader = DataLoader(
		MockedDataset(DATASET_LEN), batch_size=BATCH_SIZE
	)

	with TrainingManager(
		model=model,
		dataloader=train_loader,
		logger=logger,
		evals={
			"1 epochs": lambda model: {"wgt_mean": torch.mean(model.fc.weight).item()},
			"1/2 epochs": lambda model: logger.get_mem_usage(),
		}.items(),
		checkpoint_interval="2 epochs",
	) as tr:
		# Training loop
		for epoch in tr.epoch_loop(range(N_EPOCHS)):
			for inputs, targets in train_loader:
				optimizer.zero_grad()
				outputs = model(inputs)
				loss = criterion(outputs, targets)
				loss.backward()
				optimizer.step()

				accuracy = torch.sum(
					torch.argmax(outputs, dim=1) == targets
				).item() / len(targets)

				tr.batch_update(
					samples=len(targets),
					**{"train/loss": loss.item(), "train/acc": accuracy},
				)


def test_tm_integration_epoch_explicit_batch_wrapped():
	model = Model()
	optimizer = optim.SGD(model.parameters(), lr=0.1)
	criterion = nn.MSELoss()

	logger = LocalLogger(
		project="integration-tests",
		metric_names=["train/loss", "train/acc", "val/loss", "val/acc"],
		train_config=dict(
			model=str(model),
			dataset="dummy",
			optimizer=str(optimizer),
			criterion=str(criterion),
		),
		base_path="tests/_temp",
	)

	train_loader: DataLoader = DataLoader(
		MockedDataset(DATASET_LEN), batch_size=BATCH_SIZE
	)

	with TrainingManager(
		model=model,
		epochs_total=N_EPOCHS,
		logger=logger,
		evals={
			"1 epochs": lambda model: {"wgt_mean": torch.mean(model.fc.weight).item()},
			"1/2 epochs": lambda model: logger.get_mem_usage(),
		}.items(),
		checkpoint_interval="2 epochs",
	) as tr:
		# Training loop
		for epoch in range(N_EPOCHS):
			for inputs, targets in tr.batch_loop(train_loader):
				optimizer.zero_grad()
				outputs = model(inputs)
				loss = criterion(outputs, targets)
				loss.backward()
				optimizer.step()

				accuracy = torch.sum(
					torch.argmax(outputs, dim=1) == targets
				).item() / len(targets)

				tr.batch_update(
					samples=len(targets),
					**{"train/loss": loss.item(), "train/acc": accuracy},
				)


def test_tm_integration_epoch_explicit_batch_explicit():
	model = Model()
	optimizer = optim.SGD(model.parameters(), lr=0.1)
	criterion = nn.MSELoss()

	logger = LocalLogger(
		project="integration-tests",
		metric_names=["train/loss", "train/acc", "val/loss", "val/acc"],
		train_config=dict(
			model=str(model),
			dataset="dummy",
			optimizer=str(optimizer),
			criterion=str(criterion),
		),
		base_path="tests/_temp",
	)

	train_loader: DataLoader = DataLoader(
		MockedDataset(DATASET_LEN), batch_size=BATCH_SIZE
	)

	with TrainingManager(
		model=model,
		dataloader=train_loader,
		epochs_total=N_EPOCHS,
		logger=logger,
		evals={
			"1 epochs": lambda model: {"wgt_mean": torch.mean(model.fc.weight).item()},
			"1/2 epochs": lambda model: logger.get_mem_usage(),
		}.items(),
		checkpoint_interval="2 epochs",
	) as tr:
		# Training loop
		for epoch in range(N_EPOCHS):
			for inputs, targets in train_loader:
				optimizer.zero_grad()
				outputs = model(inputs)
				loss = criterion(outputs, targets)
				loss.backward()
				optimizer.step()

				accuracy = torch.sum(
					torch.argmax(outputs, dim=1) == targets
				).item() / len(targets)

				tr.batch_update(
					samples=len(targets),
					**{"train/loss": loss.item(), "train/acc": accuracy},
				)
