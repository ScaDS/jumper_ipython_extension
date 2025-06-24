import pandas as pd
import matplotlib.pyplot as plt

class PerformanceVisualizer:
    def __init__(self, cpu_handles, memory, gpu_memory):
        self.cpu_handles = cpu_handles
        self.memory = memory
        self.gpu_memory = gpu_memory
        self.figsize = (15, 5)
        
        # Metric configurations
        self.metric_configs = {
            'cpu': {
                'prefix': 'cpu_util_',
                'avg_column': 'cpu_util_avg',
                'title': 'CPU Utilization (%)',
                'ylim': (0, 100),
                'type': 'multi_series'
            },
            'memory': {
                'column': 'memory_usage_gb',
                'title': 'Memory Usage (GB)',
                'ylim': (0, self.memory),
                'color': 'green',
                'type': 'single_series'
            },
            'gpu_util': {
                'prefix': 'gpu_util_',
                'avg_column': 'gpu_util_avg',
                'title': 'GPU Utilization (%)',
                'ylim': (0, 100),
                'type': 'multi_series'
            },
            'gpu_band': {
                'prefix': 'gpu_band_',
                'avg_column': 'gpu_band_avg',
                'title': 'GPU Bandwidth Usage (%)',
                'ylim': (0, 100),
                'type': 'multi_series'
            },
            'gpu_mem': {
                'prefix': 'gpu_mem_',
                'avg_column': 'gpu_mem_avg',
                'title': 'GPU Memory Usage (GB)',
                'ylim': (0, self.gpu_memory),
                'type': 'multi_series'
            },
            'io_read': {
                'column': 'io_read_mb',
                'title': 'I/O Read (MB)',
                'color': 'yellow',
                'type': 'single_series'
            },
            'io_write': {
                'column': 'io_write_mb',
                'title': 'I/O Write (MB)',
                'color': 'magenta',
                'type': 'single_series'
            },
            'io_read_count': {
                'column': 'io_read_count',
                'title': 'I/O Read Operations Count',
                'color': 'cyan',
                'type': 'single_series'
            },
            'io_write_count': {
                'column': 'io_write_count',
                'title': 'I/O Write Operations Count',
                'color': 'blue',
                'type': 'single_series'
            }
        }

    def _plot_single_series(self, df, config):
        """Plot a single data series"""
        fig, ax = plt.subplots(figsize=self.figsize)
        color = config.get('color', 'blue')
        ax.plot(df['time'], df[config['column']], color=color, linewidth=2)
        
        ax.set_title(config['title'])
        ax.set_xlabel('Time (seconds)')
        ax.grid(True)
        
        if 'ylim' in config:
            ax.set_ylim(config['ylim'])

    def _plot_multi_series(self, df, config):
        """Plot multiple series with average line"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot individual series
        series_cols = [col for col in df.columns if col.startswith(config['prefix']) and not col.endswith('avg')]
        for col in series_cols:
            series_idx = col.split('_')[-1]
            if config['prefix'].startswith('cpu_'):
                label = f'CPU {self.cpu_handles[int(series_idx)]}'
            else:
                label = f'GPU {series_idx}'
            ax.plot(df['time'], df[col], '-', alpha=0.5, label=label)
        
        # Plot average line
        if config['avg_column'] in df.columns:
            ax.plot(df['time'], df[config['avg_column']], 'b-', linewidth=2, label='Mean')
        
        ax.set_title(config['title'])
        ax.set_xlabel('Time (seconds)')
        ax.grid(True)
        ax.legend()
        
        if 'ylim' in config:
            ax.set_ylim(config['ylim'])

    def _plot_metric(self, df, metric):
        """Plot a single metric using its configuration"""
        config = self.metric_configs[metric]
        
        if config['type'] == 'single_series':
            self._plot_single_series(df, config)
        elif config['type'] == 'multi_series':
            self._plot_multi_series(df, config)

    def plot(self, df: pd.DataFrame, metrics_to_plot):
        """Plot performance metrics"""
        for metric in metrics_to_plot:
            self._plot_metric(df, metric)
        
        plt.tight_layout()
        plt.show()
