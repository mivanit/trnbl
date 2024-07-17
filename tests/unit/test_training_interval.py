from typing import Union, Any
import pytest
from trnbl.training_interval import TrainingInterval, IntervalValueError


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
	with pytest.warns(IntervalValueError):
		assert (
			TrainingInterval(0, "runs").as_batch_count(
				batchsize=32, batches_per_epoch=100, epochs=10
			)
			== 1
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
		TrainingInterval(0.9, "runs").as_batch_count(
			batchsize=1, batches_per_epoch=100, epochs=10
		)
		== 900
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


@pytest.mark.parametrize(
	"quantity, unit",
	[
		(0.1, "runs"),
		(0.1, "epochs"),
		(0.0001, "runs"),
		(1e-10, "epochs"),
	],
)
def test_very_small_values(quantity: Union[int, float], unit: str) -> None:
	interval = TrainingInterval(quantity, unit)
	assert interval.quantity == quantity
	assert interval.unit == unit


def test_zero_samples() -> None:
	with pytest.warns(IntervalValueError):
		TrainingInterval(0, "samples")


@pytest.mark.parametrize("quantity", [0.51, 0.9, 1.1, 1.49])
def test_samples_rounding(quantity: float) -> None:
	if quantity < 1:
		with pytest.warns(IntervalValueError):
			interval = TrainingInterval(quantity, "samples")
	else:
		interval = TrainingInterval(quantity, "samples")
	assert interval.quantity == 1
	assert interval.unit == "samples"


@pytest.mark.parametrize(
	"quantity, unit, batchsize, batches_per_epoch, epochs, expected",
	[
		(1, "samples", 32, 100, 10, 1),
		(0.000001, "runs", 32, 100, 10, 1),
		(0.0001, "epochs", 32, 100, 10, 1),
		(1e-10, "runs", 32, 100, 10, 1),
		(1e-10, "epochs", 32, 100, 10, 1),
	],
)
def test_as_batch_count_edge_cases(
	quantity: Union[int, float],
	unit: str,
	batchsize: int,
	batches_per_epoch: int,
	epochs: int,
	expected: int,
) -> None:
	interval = TrainingInterval(quantity, unit)
	with pytest.warns(IntervalValueError):
		result = interval.as_batch_count(batchsize, batches_per_epoch, epochs)
	assert result == expected, f"Expected {expected}, but got {result} for {interval}"


def test_as_batch_count_without_epochs() -> None:
	interval = TrainingInterval(0.1, "runs")
	with pytest.raises(AssertionError):
		interval.as_batch_count(32, 100)


@pytest.mark.parametrize(
	"input_data, expected",
	[
		("0.1 runs", (0.1, "runs")),
		("0.1 epochs", (0.1, "epochs")),
		("1 batches", (1, "batches")),
		("0.1 runs", (0.1, "runs")),
		("1/1000 epochs", (0.001, "epochs")),
	],
)
def test_from_str_edge_cases(
	input_data: str, expected: tuple[float | int, str]
) -> None:
	result = TrainingInterval.from_str(input_data)
	assert result == TrainingInterval(
		*expected
	), f"Expected {expected}, but got {result} for input '{input_data}'"


@pytest.mark.parametrize(
	"input_data",
	[
		"invalid unit",
		"1.5.5 epochs",
		"123",
		"1/2/3 batches",
		"0.0.0 batches",
		"ten samples",
		"1/2/3 samples",
	],
)
def test_from_str_invalid_inputs(input_data: str) -> None:
	with pytest.raises(ValueError):
		TrainingInterval.from_str(input_data)


@pytest.mark.parametrize(
	"input_data, expected",
	[
		((0.1, "runs"), (0.1, "runs")),
		(["0.1", "epochs"], (0.1, "epochs")),
		("0.1 runs", (0.1, "runs")),
		(("1/1000", "epochs"), (0.001, "epochs")),
	],
)
def test_from_any_edge_cases_nowarn(
	input_data: Any, expected: tuple[float | int, str]
) -> None:
	"no warnings because batchsize is unknown"
	result = TrainingInterval.from_any(input_data)
	assert result == TrainingInterval(
		*expected
	), f"Expected {expected}, but got {result} for input {input_data}"


@pytest.mark.parametrize(
	"input_data, expected",
	[
		((1e-10, "batches"), (1, "batches")),
		((1e-10, "batches"), (1, "batches")),
		((0, "batches"), (1, "batches")),
		((0, "batches"), (1, "batches")),
		(("1/2 batches"), (1, "batches")),
		("0.0 batches", (1, "batches")),
		((0, "samples"), (1, "samples")),
	],
)
def test_from_any_edge_cases_warn(
	input_data: Any, expected: tuple[float | int, str]
) -> None:
	"no warnings because batchsize is unknown"
	with pytest.warns(IntervalValueError):
		result = TrainingInterval.from_any(input_data)
	assert result == TrainingInterval(
		*expected
	), f"Expected {expected}, but got {result} for input {input_data}"


@pytest.mark.parametrize(
	"input_data",
	[
		(0, "potatoes"),
		"invalid unit",
		(1.5, 5, "epochs"),
		123,
		("1", "batches", "lol"),
	],
)
def test_from_any_invalid_inputs(input_data: Any) -> None:
	with pytest.raises(ValueError):
		TrainingInterval.from_any(input_data)


@pytest.mark.parametrize(
	"interval, batchsize, batches_per_epoch, epochs, expected",
	[
		("0 runs", 32, 100, 10, 1),
		("1e-10 epochs", 32, 100, 10, 1),
		("0.1 batches", 32, 100, 10, 1),
		("1 samples", 32, 100, 10, 1),
	],
)
def test_process_to_batches_edge_cases(
	interval: Union[str, tuple, TrainingInterval],
	batchsize: int,
	batches_per_epoch: int,
	epochs: int,
	expected: int,
) -> None:
	with pytest.warns(IntervalValueError):
		result = TrainingInterval.process_to_batches(
			interval, batchsize, batches_per_epoch, epochs
		)
	assert result == expected, f"Expected {expected}, but got {result} for {interval}"


def test_normalization_edge_cases() -> None:
	interval = TrainingInterval(0.1, "runs")
	normalized = interval.normalized(batchsize=32, batches_per_epoch=100, epochs=10)
	assert normalized.quantity == 100
	assert normalized.unit == "batches"

	interval = TrainingInterval(1e-10, "epochs")
	with pytest.warns(IntervalValueError):
		normalized = interval.normalized(batchsize=32, batches_per_epoch=100)
	assert normalized.quantity == 1
	assert normalized.unit == "batches"


def test_equality_edge_cases() -> None:
	assert TrainingInterval(0.1, "runs") == TrainingInterval(0.1, "runs")
	assert TrainingInterval(0.1, "runs") != TrainingInterval(0.1, "epochs")

	with pytest.warns(IntervalValueError):
		assert TrainingInterval(1e-10, "batches") == TrainingInterval(1, "batches")


def test_iteration_and_indexing() -> None:
	interval = TrainingInterval(0.1, "runs")
	quantity, unit = interval
	assert quantity == 0.1
	assert unit == "runs"

	assert interval[0] == 0.1
	assert interval[1] == "runs"

	with pytest.raises(IndexError):
		_ = interval[2]
