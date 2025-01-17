"""
.. include:: ../README.md
"""

from trnbl.training_interval import TrainingInterval, TrainingIntervalUnit
from trnbl.loggers.base import TrainingLoggerBase
from trnbl.training_manager import TrainingManager

__all__ = [
	"TrainingInterval",
	"TrainingIntervalUnit",
	"TrainingLoggerBase",
	"TrainingManager",
	# submodules
	"loggers",
	"training_interval",
	"training_manager",
]
