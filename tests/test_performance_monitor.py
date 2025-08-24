import builtins
import os
import time
from unittest.mock import Mock, patch

from jumper_extension.monitor import PerformanceMonitor


# Save the original isinstance before patching
original_isinstance = builtins.__dict__["isinstance"]


def is_mock_instance(obj, cls):
    # If obj is a Mock, return True
    if original_isinstance(obj, Mock):
        return True
    # Otherwise, call the original isinstance
    return original_isinstance(obj, cls)


def test_comprehensive_monitor_functionality(mock_cpu_gpu, temp_dir):
    """Test monitor initialization, GPU support, lifecycle, and data collection"""
    # Test basic initialization with GPU
    monitor = PerformanceMonitor(interval=0.1)
    assert monitor.interval == 0.1
    assert not monitor.running
    assert monitor.num_gpus == 1
    assert "gpu_util" in monitor.metrics
    assert monitor.gpu_name == "NVIDIA GeForce RTX 3080"

    # Test start/stop lifecycle
    monitor.start()
    assert monitor.running
    monitor.start()  # Test already running case

    # Test data collection with GPU metrics
    time.sleep(0.2)
    monitor.stop()
    assert not monitor.running

    # Verify data collection
    df = monitor.data.view("system")
    assert len(df) > 0
    assert "cpu_util_avg" in df.columns
    assert "gpu_util_avg" in df.columns

    # Test data export
    filename = f"{temp_dir}/test.csv"
    monitor.data.export(filename, level="system")
    assert os.path.exists(filename)


def test_cpu_only_and_slurm(mock_cpu_only):
    """Test CPU-only system and SLURM memory detection"""
    with patch("os.path.exists", return_value=True), patch(
        "builtins.open", create=True
    ) as mock_file, patch("os.getuid", return_value=1000), patch.dict(
        os.environ, {"SLURM_JOB_ID": "12345"}
    ), patch(
        "jumper_extension.monitor.PYNVML_AVAILABLE", False
    ):
        mock_file.return_value.__enter__.return_value.read.return_value = (
            b"8589934592"
        )
        monitor = PerformanceMonitor()
        assert monitor.num_gpus == 0
        assert monitor.memory_limits["slurm"] == 8.0
        assert "gpu_util" not in monitor.metrics


def test_gpu_failures():
    """Test GPU setup failure scenarios"""
    with patch("jumper_extension.monitor.PYNVML_AVAILABLE", True), patch(
        "pynvml.nvmlDeviceGetCount", side_effect=Exception("GPU error")
    ), patch("psutil.Process") as mock_proc:
        mock_proc.return_value.cpu_affinity.return_value = [0, 1]
        mock_proc.return_value.io_counters.return_value = Mock(
            read_count=100, write_count=50, read_bytes=1024, write_bytes=512
        )

        monitor = PerformanceMonitor()
        assert monitor.num_gpus == 0
