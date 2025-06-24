import pandas as pd

class PerformanceData:
    def __init__(self, num_cpus, num_gpus):
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.data = []  # Store all samples as tuples
        
    def add_sample(self, time_mark, cpu_util_per_core, memory_usage, gpu_util, gpu_band, gpu_mem, io_counters):
        self.data.append((time_mark, cpu_util_per_core, memory_usage, gpu_util, gpu_band, gpu_mem, io_counters))
    
    def to_dataframe(self, slice_=None):
        """Convert performance data to DataFrame"""
        if not self.data:
            return pd.DataFrame()
        
        # Unpack data
        times, cpu_per_core, memory, gpu_utils, gpu_bands, gpu_mems, io_data = zip(*self.data)
        
        # Build base data dictionary
        flat_data = {
            'time': times,
            'memory_usage_gb': memory,
            'io_read_count': [io[0] for io in io_data],
            'io_write_count': [io[1] for io in io_data],
            'io_read_mb': [io[2] for io in io_data],
            'io_write_mb': [io[3] for io in io_data]
        }
        
        # Add CPU metrics
        flat_data['cpu_util_avg'] = [sum(cpu)/self.num_cpus for cpu in cpu_per_core]
        for i in range(self.num_cpus):
            flat_data[f'cpu_util_{i}'] = [cpu[i] for cpu in cpu_per_core]
        
        # Add GPU metrics if available
        if self.num_gpus > 0:
            flat_data['gpu_util_avg'] = [sum(gpu)/self.num_gpus for gpu in gpu_utils]
            flat_data['gpu_band_avg'] = [sum(gpu)/self.num_gpus for gpu in gpu_bands] 
            flat_data['gpu_mem_avg'] = [sum(gpu)/self.num_gpus for gpu in gpu_mems]
            for i in range(self.num_gpus):
                flat_data[f'gpu_util_{i}'] = [gpu[i] for gpu in gpu_utils]
                flat_data[f'gpu_band_{i}'] = [gpu[i] for gpu in gpu_bands]
                flat_data[f'gpu_mem_{i}'] = [mem[i] for mem in gpu_mems]
        
        df = pd.DataFrame(flat_data)
        return df.iloc[slice_[0]:slice_[1]+1] if slice_ else df
    
    def export(self, filename='performance_data.csv'):
        """Export metrics to CSV"""
        self.to_dataframe().to_csv(filename, index=False)
