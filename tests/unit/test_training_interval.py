import pytest
from trnbl.training_interval import TrainingInterval


def test_as_batch_count():
	assert (
		TrainingInterval(1, "runs").as_batch_count(
			batchsize=32, batches_per_epoch=100, epochs=10
		)
		== 1000
	)
	assert (
		TrainingInterval(5, "epochs").as_batch_count(
			batchsize=32, batches_per_epoch=100
		)
		== 500
	)
	assert (
		TrainingInterval(200, "batches").as_batch_count(
			batchsize=32, batches_per_epoch=100
		)
		== 200
	)
	assert (
		TrainingInterval(6400, "samples").as_batch_count(
			batchsize=32, batches_per_epoch=100
		)
		== 200
	)


def test_normalized():
	interval = TrainingInterval(1, "runs")
	normalized_interval = interval.normalized(
		batchsize=32, batches_per_epoch=100, epochs=10
	)
	assert (
		normalized_interval.quantity == 1000 and normalized_interval.unit == "batches"
	)

	interval = TrainingInterval(5, "epochs")
	normalized_interval = interval.normalized(batchsize=32, batches_per_epoch=100)
	assert normalized_interval.quantity == 500 and normalized_interval.unit == "batches"


def test_from_str():
	assert TrainingInterval.from_str("5 epochs") == TrainingInterval(5, "epochs")
	assert TrainingInterval.from_str("100 batches") == TrainingInterval(100, "batches")
	assert TrainingInterval.from_str("0.1 runs") == TrainingInterval(0.1, "runs")
	assert TrainingInterval.from_str("1/5 runs") == TrainingInterval(0.2, "runs")


def test_from_any():
	assert TrainingInterval.from_any("5 epochs") == TrainingInterval(5, "epochs")
	assert TrainingInterval.from_any("5", "epochs") == TrainingInterval(5, "epochs")
	assert TrainingInterval.from_any(("5", "epochs")) == TrainingInterval(5, "epochs")
	assert TrainingInterval.from_any(["5", "epochs"]) == TrainingInterval(5, "epochs")
	assert TrainingInterval.from_any(TrainingInterval(5, "epochs")) == TrainingInterval(
		5, "epochs"
	)

	assert TrainingInterval.from_any("100 batches") == TrainingInterval(100, "batches")
	assert TrainingInterval.from_any("100", "batches") == TrainingInterval(
		100, "batches"
	)
	assert TrainingInterval.from_any(("100", "batches")) == TrainingInterval(
		100, "batches"
	)
	assert TrainingInterval.from_any(["100", "batches"]) == TrainingInterval(
		100, "batches"
	)
	assert TrainingInterval.from_any(
		TrainingInterval(100, "batches")
	) == TrainingInterval(100, "batches")

	assert TrainingInterval.from_any("0.1 runs") == TrainingInterval(0.1, "runs")
	assert TrainingInterval.from_any("0.1", "runs") == TrainingInterval(0.1, "runs")
	assert TrainingInterval.from_any(("0.1", "runs")) == TrainingInterval(0.1, "runs")
	assert TrainingInterval.from_any(["0.1", "runs"]) == TrainingInterval(0.1, "runs")
	assert TrainingInterval.from_any(TrainingInterval(0.1, "runs")) == TrainingInterval(
		0.1, "runs"
	)

	assert TrainingInterval.from_any("1/5 runs") == TrainingInterval(0.2, "runs")
	assert TrainingInterval.from_any("1/5", "runs") == TrainingInterval(0.2, "runs")
	assert TrainingInterval.from_any(("1/5", "runs")) == TrainingInterval(0.2, "runs")
	assert TrainingInterval.from_any(["1/5", "runs"]) == TrainingInterval(0.2, "runs")
	assert TrainingInterval.from_any(
		TrainingInterval(1 / 5, "runs")
	) == TrainingInterval(0.2, "runs")


def test_process_to_batches():
	assert (
		TrainingInterval.process_to_batches(
			"5 epochs", batchsize=32, batches_per_epoch=100
		)
		== 500
	)
	assert (
		TrainingInterval.process_to_batches(
			("100", "batches"), batchsize=32, batches_per_epoch=100
		)
		== 100
	)
	assert (
		TrainingInterval.process_to_batches(
			TrainingInterval(0.1, "runs"),
			batchsize=32,
			batches_per_epoch=100,
			epochs=10,
		)
		== 100
	)
	assert (
		TrainingInterval.process_to_batches(
			("1/5", "runs"), batchsize=32, batches_per_epoch=100, epochs=10
		)
		== 200
	)


def test_edge_cases():
	assert (
		TrainingInterval(0, "runs").as_batch_count(
			batchsize=32, batches_per_epoch=100, epochs=10
		)
		== 0
	)
	assert TrainingInterval(1e6, "batches").as_batch_count(
		batchsize=32, batches_per_epoch=100, epochs=10
	) == int(1e6)
	assert (
		TrainingInterval(14, "samples").as_batch_count(
			batchsize=10, batches_per_epoch=100, epochs=10
		)
		== 1
	)


def test_invalid_inputs():
	with pytest.raises(ValueError):
		TrainingInterval.from_str("5 decades")

	with pytest.raises(ValueError):
		TrainingInterval.from_any((100,))

	with pytest.raises(ValueError):
		TrainingInterval.from_any(123)

	with pytest.raises(ValueError):
		TrainingInterval.from_any(("5", "epochs", "lol"))

	with pytest.raises(ValueError):
		TrainingInterval.from_any("5", "epochs", "lol")


def test_boundary_cases():
	assert (
		TrainingInterval(1, "runs").as_batch_count(
			batchsize=1, batches_per_epoch=100, epochs=10
		)
		== 1000
	)
	assert (
		TrainingInterval(1, "epochs").as_batch_count(batchsize=32, batches_per_epoch=1)
		== 1
	)
	assert (
		TrainingInterval(1, "runs").as_batch_count(
			batchsize=32, batches_per_epoch=100, epochs=1
		)
		== 100
	)


def test_unpacking():
	quantity, unit = TrainingInterval(5, "epochs")
	assert quantity == 5 and unit == "epochs"
	quantity, unit = TrainingInterval(100, "batches")
	assert quantity == 100 and unit == "batches"
	quantity, unit = TrainingInterval(0.1, "runs")
	assert quantity == 0.1 and unit == "runs"
	quantity, unit = TrainingInterval(1 / 12, "runs")
	assert quantity == 1 / 12 and unit == "runs"
