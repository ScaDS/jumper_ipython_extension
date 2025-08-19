import tempfile
from unittest.mock import Mock, patch

import pytest
from IPython.testing.globalipapp import get_ipython


@pytest.fixture
def ipython():
    # Try to get actual IPython instance, fallback to mock if not available
    ip = get_ipython()
    if ip is not None:
        return ip

    # Create a mock IPython instance with required attributes
    from IPython import InteractiveShell

    # Create a basic InteractiveShell instance for testing
    shell = InteractiveShell.instance()
    shell.events = Mock()
    shell.register_magics = Mock()
    shell.events.register = Mock()
    shell.events.unregister = Mock()

    return shell


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_cpu_only():
    """Mock system with 1 CPU (4 cores) and no GPU"""
    with patch("psutil.cpu_count", return_value=4), patch(
        "psutil.cpu_percent", return_value=[25.0, 30.0, 20.0, 35.0]
    ), patch("psutil.virtual_memory") as mock_mem, patch(
        "psutil.Process"
    ) as mock_proc, patch(
        "psutil.disk_io_counters"
    ) as mock_disk, patch(
        "jumper_extension.monitor.PYNVML_AVAILABLE", False
    ):
        mock_mem.return_value.total = 8 * 1024**3
        mock_mem.return_value.available = 4 * 1024**3
        mock_proc.return_value.cpu_affinity.return_value = [0, 1, 2, 3]
        mock_proc.return_value.cpu_percent.return_value = [
            25.0,
            25.0,
            25.0,
            25.0,
        ]
        mock_proc.return_value.memory_full_info.return_value.uss = 2 * 1024**3
        mock_proc.return_value.io_counters.return_value = Mock(
            read_count=100, write_count=50, read_bytes=1024, write_bytes=512
        )
        mock_disk.return_value = Mock(
            read_count=1000,
            write_count=500,
            read_bytes=10240,
            write_bytes=5120,
        )
        yield


@pytest.fixture
def mock_cpu_gpu(mock_cpu_only):
    """Mock system with 1 CPU (4 cores) and 1 GPU"""
    with patch("jumper_extension.monitor.PYNVML_AVAILABLE", True), patch(
        "pynvml.nvmlInit"
    ), patch("pynvml.nvmlDeviceGetCount", return_value=1), patch(
        "pynvml.nvmlDeviceGetHandleByIndex", return_value=Mock()
    ), patch(
        "pynvml.nvmlDeviceGetName", return_value=b"NVIDIA GeForce RTX 3080"
    ), patch(
        "pynvml.nvmlDeviceGetMemoryInfo"
    ) as mock_mem, patch(
        "pynvml.nvmlDeviceGetUtilizationRates"
    ) as mock_util, patch(
        "pynvml.nvmlDeviceGetTemperature", return_value=65
    ), patch(
        "pynvml.nvmlDeviceGetComputeRunningProcesses", return_value=[]
    ):
        mock_mem.return_value = Mock(
            total=10 * 1024**3, used=2 * 1024**3, free=8 * 1024**3
        )
        mock_util.return_value = Mock(gpu=75, memory=20)
        yield
