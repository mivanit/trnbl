import json
from pathlib import Path


from trnbl.loggers.local import LocalLogger, FilePaths

# Temporary directory for testing
TEMP_PATH = Path("tests/_temp")
TEMP_PATH.mkdir(parents=True, exist_ok=True)


# Sample training configuration
train_config = {"batch_size": 32, "learning_rate": 0.001, "epochs": 10}


# Define the test functions
def test_logger_initialization():
	logger = LocalLogger(
		project="test_project",
		metric_names=["accuracy", "loss"],
		group="test_run",
		train_config=train_config,
		base_path=TEMP_PATH,
	)
	assert logger.project == "test_project"
	assert "accuracy" in logger.metric_names
	assert logger.train_config == train_config
	logger.finish()


def test_logger_message():
	logger = LocalLogger(
		project="test_project",
		metric_names=["accuracy", "loss"],
		group="test_run",
		train_config=train_config,
		base_path=TEMP_PATH,
	)
	logger.message("Test message")
	assert len(logger.log_list) > 0
	assert logger.log_list[-1]["message"] == "Test message"
	logger.finish()


def test_logger_metrics():
	logger = LocalLogger(
		project="test_project",
		metric_names=["accuracy", "loss"],
		group="test_run",
		train_config=train_config,
		base_path=TEMP_PATH,
	)
	logger.metrics({"accuracy": 0.95, "loss": 0.05})
	assert len(logger.metrics_list) > 0
	assert logger.metrics_list[-1]["accuracy"] == 0.95
	assert logger.metrics_list[-1]["loss"] == 0.05
	logger.finish()


def test_logger_artifact():
	logger = LocalLogger(
		project="test_project",
		metric_names=["accuracy", "loss"],
		group="test_run",
		train_config=train_config,
		base_path=TEMP_PATH,
	)
	artifact_path = Path("tests/_temp/test_artifact.txt")
	artifact_path.parent.mkdir(parents=True, exist_ok=True)
	artifact_path.write_text("This is a test artifact.")
	logger.artifact(artifact_path, type="text")
	assert len(logger.artifacts_list) > 0
	assert logger.artifacts_list[-1]["path"] == artifact_path.as_posix()
	artifact_path.unlink()
	logger.finish()


def test_logger_files():
	# Define a specific path for this test run
	test_run_path = TEMP_PATH / "test_run"
	test_run_path.mkdir(parents=True, exist_ok=True)

	# Initialize the logger with the test run path
	logger = LocalLogger(
		project="test_project",
		metric_names=["accuracy", "loss"],
		group="test_run",
		train_config=train_config,
		base_path=test_run_path,
	)

	# Log message
	logger.message("Test message")
	logger.flush()
	log_file_path = logger.run_path / FilePaths.LOG
	print(f"{log_file_path = }")
	with open(log_file_path, "r") as log_file:
		log_content = log_file.readlines()
	print("Log file content:", log_content)  # Debug print
	assert len(log_content) > 0
	assert "Test message" in log_content[-1]

	# Log metrics
	logger.metrics({"accuracy": 0.95, "loss": 0.05})
	logger.flush()
	metrics_file_path = logger.run_path / FilePaths.METRICS
	with open(metrics_file_path, "r") as metrics_file:
		metrics_content = metrics_file.readlines()
	print("Metrics file content:", metrics_content)  # Debug print
	assert len(metrics_content) > 0
	metrics_data = json.loads(metrics_content[-1])
	assert metrics_data["accuracy"] == 0.95
	assert metrics_data["loss"] == 0.05

	# Log artifact
	artifact_path = logger.run_path / "test_artifact.txt"
	artifact_path.write_text("This is a test artifact.")
	logger.artifact(artifact_path, type="text")
	logger.flush()
	artifacts_file_path = logger.run_path / FilePaths.ARTIFACTS
	with open(artifacts_file_path, "r") as artifacts_file:
		artifacts_content = artifacts_file.readlines()
	print("Artifacts file content:", artifacts_content)  # Debug print
	assert len(artifacts_content) > 0
	artifact_data = json.loads(artifacts_content[-1])
	assert artifact_data["path"] == artifact_path.as_posix()

	# Clean up after tests
	logger.finish()
