import os
import time
import threading
import psutil
from .performance_data import PerformanceData

# GPU monitoring setup
PYNVML_AVAILABLE = False
try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except ImportError:
    print("Warning: pynvml not available. GPU monitoring disabled.")
except Exception:
    print("NVIDIA drivers not available. GPU monitoring disabled.")

class PerformanceMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.running = False
        self.start_time = None
        self.monitor_thread = None
        
        # Process info
        self.process = psutil.Process()
        self.cpu_handles = self.process.cpu_affinity()
        self.num_cpus = len(self.cpu_handles)
        
        # Memory detection (SLURM-aware)
        self.memory = self._detect_memory_limit()
        
        # GPU setup
        self.gpu_handles = []
        self.gpu_memory = 0
        self.gpu_name = ""
        if PYNVML_AVAILABLE:
            self._setup_gpu()
        self.num_gpus = len(self.gpu_handles)
        
        # Metrics list
        self.metrics = ['cpu', 'memory', 'io_read', 'io_write', 'io_read_count', 'io_write_count']
        if self.num_gpus:
            self.metrics.extend(['gpu_util', 'gpu_band', 'gpu_mem'])
        
        self.data = PerformanceData(self.num_cpus, self.num_gpus)
    
    def _detect_memory_limit(self):
        """Detect memory limit (SLURM-aware)"""
        slurm_path = f"/sys/fs/cgroup/memory/slurm/uid_{os.getuid()}/job_{os.environ.get('SLURM_JOB_ID', 0)}/memory.limit_in_bytes"
        if os.path.exists(slurm_path):
            with open(slurm_path) as f:
                return round(int(f.read().strip()) / (1024**3), 2)
        return round(psutil.virtual_memory().total / (1024**3), 2)
    
    def _setup_gpu(self):
        """Initialize GPU monitoring"""
        try:
            ngpus = pynvml.nvmlDeviceGetCount()
            self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(ngpus)]
            if self.gpu_handles:
                handle = self.gpu_handles[0]
                self.gpu_memory = round(pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**3), 2)
                name = pynvml.nvmlDeviceGetName(handle)
                self.gpu_name = name.decode() if isinstance(name, bytes) else name
        except Exception:
            self.gpu_handles = []
    
    def _collect_metrics(self):
        """Collect all performance metrics"""
        time_mark = time.time()
        
        # CPU and memory
        cpu_util_per_core = psutil.cpu_percent(percpu=True)
        available_cpu_util = [cpu_util_per_core[i] for i in self.cpu_handles]
        mem_util = (psutil.virtual_memory().total - psutil.virtual_memory().available) / (1024**3)
        
        # I/O
        io_data = self.process.io_counters()
        io_counters = [io_data.read_count, io_data.write_count, 
                      io_data.read_bytes / (1024**2), io_data.write_bytes / (1024**2)]
        
        # GPU metrics
        gpu_util, gpu_band, gpu_mem = [], [], []
        for handle in self.gpu_handles:
            util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_util.append(util_rates.gpu)
            gpu_band.append(util_rates.memory)
            gpu_mem.append((memory_info.total - memory_info.free) / (1024**3))
        
        return time_mark, available_cpu_util, mem_util, gpu_util, gpu_band, gpu_mem, io_counters
    
    def _collect_data(self):
        """Data collection loop"""
        while self.running:
            self.data.add_sample(*self._collect_metrics())
            time.sleep(self.interval)
    
    def start(self):
        """Start monitoring"""
        if self.running:
            print("Performance monitor already running")
            return
        
        self.start_time = time.time()
        self.running = True
        self.monitor_thread = threading.Thread(target=self._collect_data, daemon=True)
        self.monitor_thread.start()
        print(f"Performance monitoring started (PID: {os.getpid()}, Interval: {self.interval}s)")
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        print(f"Performance monitoring stopped (ran for {time.time() - self.start_time:.2f} seconds)")
    
    