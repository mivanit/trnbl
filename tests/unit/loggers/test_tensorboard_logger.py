import json
import pytest
import math
from pathlib import Path
from datetime import datetime

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

from trnbl.loggers.tensorboard import TensorBoardLogger

# Temporary directory for testing
TEMP_PATH = Path("tests/_temp/tensorboard")
TEMP_PATH.mkdir(parents=True, exist_ok=True)


def read_event_file(event_file):
	loader = EventFileLoader(event_file.as_posix())
	for event in loader.Load():
		yield event


def test_tensorboard_logger_initialization():
	log_dir = TEMP_PATH / "init"
	logger = TensorBoardLogger(log_dir=log_dir)
	assert logger._run_path == log_dir
	logger.finish()


def test_tensorboard_logger_message():
	log_dir = TEMP_PATH / "message"
	logger = TensorBoardLogger(log_dir=log_dir)

	logger.message("Test message", key="value")
	logger.flush()

	event_files = list(Path(log_dir).glob("**/events.out.tfevents.*"))
	print(f"{event_files = }")

	assert len(event_files) > 0  # Ensure at least one event file is created

	logger.flush()

	# Inspect the content of the event files
	found_content: bool = False
	for event_file in event_files:
		print(f"{event_file = }")
		for event in read_event_file(event_file):
			print(f"\t{event = }")
			if event.HasField("summary"):
				for value in event.summary.value:
					if value.tag == "message/text_summary":
						message = value.tensor.string_val[0].decode("utf-8")
						assert "Test message" in message
						assert '"key": "value"' in message
						found_content = True

	assert found_content

	logger.finish()


def test_tensorboard_logger_metrics():
	log_dir = TEMP_PATH / "metrics"
	logger = TensorBoardLogger(log_dir=log_dir)

	metrics_data = {"accuracy": 0.95, "loss": 0.05}
	logger.metrics(metrics_data)

	event_files = list(Path(log_dir).glob("**/events.out.tfevents.*"))
	assert len(event_files) > 0  # Ensure at least one event file is created

	logger.flush()

	# Inspect the content of the event files
	found_accuracy: bool = False
	found_loss: bool = False
	for event_file in event_files:
		print(f"{event_file = }")
		print(f"====================================")
		for event in read_event_file(event_file):
			print(f"{event = }")
			if event.HasField("summary"):
				for value in event.summary.value:
					if value.tag == "accuracy":
						found_accuracy = True
						assert math.isclose(
							value.tensor.float_val[0], 0.95, rel_tol=1e-5
						)
					if value.tag == "loss":
						found_loss = True
						assert math.isclose(
							value.tensor.float_val[0], 0.05, rel_tol=1e-5
						)

	assert found_accuracy
	assert found_loss

	logger.finish()


def test_tensorboard_logger_artifact():
	log_dir = TEMP_PATH / "artifact"
	logger = TensorBoardLogger(log_dir=log_dir)

	artifact_path = Path(TEMP_PATH / "test_artifact.txt")
	artifact_path.write_text("This is a test artifact.")

	logger.artifact(
		artifact_path, type="text", aliases=["alias1"], metadata={"key": "value"}
	)
	logger.flush()

	event_files = list(Path(log_dir).glob("**/events.out.tfevents.*"))
	assert len(event_files) > 0  # Ensure at least one event file is created

	logger.flush()

	# Inspect the content of the event files
	found_artifact = False
	for event_file in event_files:
		print(f"{event_file = }")
		for event in read_event_file(event_file):
			print(f"{event = }")
			if hasattr(event, 'summary'):
				for value in event.summary.value:
					if value.tag == "artifact/text_summary":
						found_artifact = True
						artifact_data = json.loads(value.tensor.string_val[0])
						
						# Check artifact data
						assert Path(artifact_data["path"]).as_posix() == artifact_path.as_posix(), f"Expected path {artifact_path}, got {artifact_data['path']}"
						assert artifact_data["type"] == "text", f"Expected type 'text', got {artifact_data['type']}"
						assert artifact_data["aliases"] == ["alias1"], f"Expected aliases ['alias1'], got {artifact_data['aliases']}"
						assert artifact_data["metadata"] == {"key": "value"}, f"Expected metadata {{'key': 'value'}}, got {artifact_data['metadata']}"
						
						# Check timestamp format (assuming it's ISO format)						
						timestamp_datetime = datetime.fromisoformat(artifact_data["timestamp"])
						# Check if the timestamp is within the last 5 minutes
						assert (datetime.now() - timestamp_datetime).total_seconds() < 300, f"Timestamp is not within the last 5 minutes: {timestamp_datetime}"

	assert found_artifact, "Artifact data not found in event files"
	logger.finish()


def test_tensorboard_logger_url():
	log_dir = TEMP_PATH / "url"
	logger = TensorBoardLogger(log_dir=log_dir)
	assert logger.url == f"tensorboard --logdir={logger._run_path}"
	logger.finish()


def test_tensorboard_logger_run_path():
	log_dir = TEMP_PATH / "run_path"
	logger = TensorBoardLogger(log_dir=log_dir)
	assert logger.run_path == Path(log_dir)
	logger.finish()

