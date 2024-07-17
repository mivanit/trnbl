from typing import Any, Iterable, Literal, Callable
from dataclasses import dataclass

from muutils.misc import str_to_numeric
from muutils.errormode import ErrorMode
from muutils.interval import Interval

_EPSILON: float = 1e-6

# units of training intervals -- we convert this all to batches
TrainingIntervalUnit = Literal["runs", "epochs", "batches", "samples"]

_TRAINING_INTERVAL_UNITS_RANGES: dict[TrainingIntervalUnit, Interval] = {
	# epochs and runs should not actually be closed, but we allow it
	"runs": Interval(0, 1, is_closed=True),
	"epochs": Interval(0, float("inf"), is_closed=True),
	"batches": Interval(1, float("inf"), is_closed=True),
	"samples": Interval(1, float("inf"), is_closed=True),
}

_TRAINING_INTERVAL_UNITS_CAST: dict[TrainingIntervalUnit, Callable] = {
	"runs": lambda x: x,
	"epochs": lambda x: x,
	"batches": lambda x: int(round(x)),
	"samples": lambda x: int(round(x)),
}

_TRAINING_INTERVAL_UNIT_ALIASES: dict[str, str] = {
	"run": "runs",
	"epoch": "epochs",
	"batch": "batches",
	"sample": "samples",
}

# what to do if interval is < 1 batch
# if WARN or IGNORE, set it to 1 batch
WhenIntervalLessThanBatch: ErrorMode = ErrorMode.WARN


class IntervalValueError(UserWarning):
	"""Error for when the interval is less than 1 batch"""

	pass


@dataclass(frozen=True)
class TrainingInterval:
	"""A training interval, which can be specified in a few different units.

	# Attributes:
	- `quantity: int|float` - the quantity of the interval
	- `unit: TrainingIntervalUnit` - the unit of the interval, one of "runs", "epochs", "batches", or "samples"

	# Methods:
	- `TrainingInterval.from_str(raw: str) -> TrainingInterval` - parse a string into a TrainingInterval object
	- `TrainingInterval.as_batch_count(batchsize: int, batches_per_epoch: int, epochs: int|None) -> int` - convert the interval to a raw number of batches
	- `TrainingInterval.process_to_batches(interval: str|TrainingInterval, batchsize: int, batches_per_epoch: int, epochs: int|None) -> int` - any representation to a number of batches
	- `TrainingInterval.normalized(batchsize: int, batches_per_epoch: int, epochs: int|None) -> None` - current interval, with units switched to batches

	Provides methods for reading from a string or tuple, and normalizing to batches.
	"""

	quantity: int | float
	unit: TrainingIntervalUnit

	def __iter__(self):
		yield self.quantity
		yield self.unit

	def __getitem__(self, index: int):
		if index == 0:
			return self.quantity
		elif index == 1:
			return self.unit
		else:
			raise IndexError(f"invalid index {index} for TrainingInterval")

	def __post_init__(self):
		try:
			assert isinstance(
				self.quantity, (int, float)
			), "quantity should be an integer or float"

			if self.unit not in TrainingIntervalUnit.__args__:
				unit_dealised: str | None = _TRAINING_INTERVAL_UNIT_ALIASES.get(
					self.unit.lower(), None
				)
				if isinstance(unit_dealised, str):
					self.__dict__["unit"] = unit_dealised
				else:
					raise ValueError(f"invalid unit {self.unit = }")

			assert (
				self.unit in TrainingIntervalUnit.__args__
			), f"invalid unit {self.unit}"
		except AssertionError as e:
			raise AssertionError(
				f"Error initializing TrainingInterval\n{self}\n{e}"
			) from e

		# check values in proper ranges
		expected_interval: Interval = _TRAINING_INTERVAL_UNITS_RANGES[self.unit]
		print(f"{self.quantity = } {expected_interval = }")
		if self.quantity not in expected_interval:
			WhenIntervalLessThanBatch.process(
				f"interval {self} has invalid quantity, expected in interval {expected_interval}, will set to closest bound if not erroring out",
				except_cls=IntervalValueError,
				warn_cls=IntervalValueError,
			)
			self.__dict__["quantity"] = expected_interval.clamp(self.quantity)

		# cast if necessary
		self.__dict__["quantity"] = _TRAINING_INTERVAL_UNITS_CAST[self.unit](
			self.quantity
		)

	def __eq__(self, other: Any) -> bool:
		if not isinstance(other, self.__class__):
			raise TypeError(
				f"invalid type {type(other)} for comparison with TrainingInterval"
			)
		return (
			abs(self.quantity - other.quantity) < _EPSILON and self.unit == other.unit
		)

	def as_batch_count(
		self,
		batchsize: int,
		batches_per_epoch: int,
		epochs: int | None = None,
	) -> int:
		"""given the batchsize, number of batches per epoch, and number of epochs, return the interval as a number of batches

		# Parameters:
		 - `batchsize: int`
		   the size of a batch
		 - `batches_per_epoch: int`
		   the number of batches in an epoch
		 - `epochs: int|None`
		   the number of epochs to run (only required if the interval is in "runs")

		# Returns:
		 - `int`
		   the interval as a number of batches

		# Raises:
		 - `ValueError`
		   if the interval is less than 1 batch, and the `trnbl.training_interval.WhenIntervalLessThanBatch` is set to `muutils.errormode.ErrorMode.ERROR`
		   otherwise, will warn or ignore and set the interval to 1 batch
		 - `ValueError`
		   if the unit is not one of "runs", "epochs", "batches", or "samples"


		"""

		output: int | float

		match self.unit:
			case "runs":
				assert (
					epochs is not None
				), "epochs must be provided to convert runs to batches"
				output = self.quantity * epochs * batches_per_epoch
			case "epochs":
				output = self.quantity * batches_per_epoch
			case "batches":
				output = self.quantity
			case "samples":
				output = self.quantity / batchsize
			case _:
				raise ValueError(f"invalid unit {self.unit}")

		# check if interval is less than 1 batch
		if output < 1:
			WhenIntervalLessThanBatch.process(
				f"interval {self} is less than 1 batch, will set to 1 batch if not erroring out",
				except_cls=IntervalValueError,
				warn_cls=IntervalValueError,
			)
			output = 1

		return int(round(output))

	def normalized(
		self,
		batchsize: int,
		batches_per_epoch: int,
		epochs: int | None = None,
	) -> "TrainingInterval":
		"""convert the units of the interval to batches, by calling `as_batch_count` and setting the `unit` to "batches"""
		quantity: int | float = self.as_batch_count(
			batches_per_epoch=batches_per_epoch,
			batchsize=batchsize,
			epochs=epochs,
		)
		unit: str = "batches"
		return self.__class__(quantity, unit)

	@classmethod
	def from_str(cls, raw: str) -> "TrainingInterval":
		"""parse a string into a TrainingInterval object

		# Examples:

		>>> TrainingInterval.from_str("5 epochs")
		TrainingInterval(5, 'epochs')
		>>> TrainingInterval.from_str("100 batches")
		TrainingInterval(100, 'batches')
		>>> TrainingInterval.from_str("0.1 runs")
		TrainingInterval(0.1, 'runs')
		>>> TrainingInterval.from_str("1/5 runs")
		TrainingInterval(0.2, 'runs')

		"""
		try:
			# remove prefix and suffix (optionally)
			raw = raw.removeprefix("TrainingInterval(").removesuffix(")")

			# process quantity
			raw_split: str
			quantity_str: str
			if "," in raw:
				raw_split = raw.split(",")
				quantity_str = ",".join(raw_split[:-1])
			else:
				raw_split = raw.split()
				quantity_str = " ".join(raw_split[:-1])

			quantity: int | float = str_to_numeric(quantity_str)

			# process unit
			unit: str = raw_split[-1]
			unit.strip().strip("'\"").strip()

			# unit should be one of the allowed units
			unit_dealised: str | None
			if unit.lower() in TrainingIntervalUnit.__args__:
				unit_dealised = unit.lower()
			else:
				unit_dealised = _TRAINING_INTERVAL_UNIT_ALIASES.get(unit.lower(), None)
			if isinstance(unit_dealised, str):
				unit = unit_dealised
			else:
				raise ValueError(f"invalid unit {unit}")

			assert unit in TrainingIntervalUnit.__args__
		except Exception as e:
			raise ValueError(f"Error parsing {raw} as a TrainingInterval\n{e}") from e

		return cls(quantity, unit)

	@classmethod
	def from_any(cls, *args, **kwargs) -> "TrainingInterval":
		"""parse a string or tuple into a TrainingInterval object"""

		try:
			# no kwargs allowed
			assert len(kwargs) == 0, "no kwargs allowed for from_any"

			# split up args
			data: Any
			match len(args):
				case 1:
					data = args[0]
				case 2:
					data = args
				case _:
					raise ValueError(
						f"invalid number of args {len(args)} for from_any: {args = }"
					)

			if isinstance(data, cls):
				return data
			elif isinstance(data, str):
				return cls.from_str(data)
			elif isinstance(data, Iterable):
				assert (
					len(data) == 2
				), f"invalid length {len(data)} for TrainingInterval: {data}"
				quantity, unit = data
				if isinstance(quantity, str):
					quantity = str_to_numeric(quantity)
				return cls(quantity, unit)
			else:
				raise ValueError(f"invalid type {type(data)} for TrainingInterval")

		except AssertionError as e:
			raise ValueError(f"Error parsing {data} as a TrainingInterval\n{e}") from e

	@classmethod
	def process_to_batches(
		cls,
		interval: "str|tuple|TrainingInterval",
		batchsize: int,
		batches_per_epoch: int,
		epochs: int | None = None,
	) -> int:
		"""directly from any representation to a number of batches"""

		interval_ti: TrainingInterval = cls.from_any(interval)

		return interval_ti.as_batch_count(
			batches_per_epoch=batches_per_epoch,
			batchsize=batchsize,
			epochs=epochs,
		)
