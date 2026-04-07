#!/usr/bin/env python3
"""
Minimal example of using the multinode SLURM monitor with srun to collect performance data
while running time.sleep on all selected nodes.

This script demonstrates:
1. Starting the multinode monitor using srun (no SSH passwords needed)
2. Running a simple workload (time.sleep) on all nodes
3. Collecting and analyzing performance data

Usage:
    # Submit to SLURM
    sbatch --nodes=2 --ntasks-per-node=1 --time=00:05:00 run_multinode_srun_example.sh

    # Or run directly if already in SLURM environment
    python multinode_srun_example.py
"""

import time
import os
import sys
import json
import logging
from pathlib import Path

# Add the jumper_extension to Python path
sys.path.insert(0, str(Path(__file__).parent))

from jumper_extension.monitor_slurm_multinode.monitor import SlurmMultinodeMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def run_workload_on_all_nodes(duration_seconds=30):
    """
    Simulate a workload on all nodes by sleeping.
    
    This function runs on the head node and represents the coordination
    of work across all SLURM nodes. The actual monitoring happens
    on each node via srun-launched agents.
    """
    logger.info(f"Starting workload: sleeping for {duration_seconds} seconds")
    logger.info("All nodes are being monitored via srun during this period")
    
    # Simulate work with periodic progress updates
    start_time = time.time()
    interval = duration_seconds / 10  # 10 progress updates
    
    for i in range(10):
        time.sleep(interval)
        elapsed = time.time() - start_time
        remaining = duration_seconds - elapsed
        logger.info(f"Work in progress... {elapsed:.1f}s elapsed, {remaining:.1f}s remaining")
    
    logger.info("Workload completed")


def analyze_performance_data(log_file="jumper_multinode.jsonl"):
    """
    Analyze the collected performance data from the multinode monitor.
    
    Args:
        log_file: Path to the JSONL file containing performance data
    """
    if not os.path.exists(log_file):
        logger.warning(f"No performance data file found: {log_file}")
        return
    
    logger.info(f"Analyzing performance data from {log_file}")
    
    # Read and parse the performance data
    samples = []
    node_info = {}
    
    with open(log_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'sample' in data:
                    samples.append(data)
                elif 'status' in data and data['status'] == 'ready':
                    node_info[data['node']] = data
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line: {e}")
    
    if not samples:
        logger.warning("No performance samples found")
        return
    
    # Basic statistics
    logger.info(f"Collected {len(samples)} performance samples")
    logger.info(f"Monitored nodes: {list(node_info.keys())}")
    
    # Node information
    for node, info in node_info.items():
        logger.info(f"Node {node}: {info.get('num_cpus', 0)} CPUs, {info.get('num_gpus', 0)} GPUs")
    
    # Performance metrics summary
    cpu_utils = []
    memory_usages = []
    
    for sample in samples:
        sample_data = sample.get('sample', {})
        if 'cpu_util' in sample_data and sample_data['cpu_util']:
            # Average CPU utilization across cores
            avg_cpu = sum(sample_data['cpu_util']) / len(sample_data['cpu_util'])
            cpu_utils.append(avg_cpu)
        
        if 'memory' in sample_data:
            memory_usages.append(sample_data['memory'])
    
    if cpu_utils:
        logger.info(f"Average CPU utilization: {sum(cpu_utils)/len(cpu_utils):.2f}")
        logger.info(f"Max CPU utilization: {max(cpu_utils):.2f}")
    
    if memory_usages:
        logger.info(f"Average memory usage: {sum(memory_usages)/len(memory_usages):.2f} MB")
        logger.info(f"Max memory usage: {max(memory_usages):.2f} MB")


def main():
    """Main execution function."""
    logger.info("=== Multinode SLURM Monitor Example (srun-based) ===")
    
    # Check if we're in a SLURM environment
    if not os.environ.get('SLURM_JOB_ID'):
        logger.warning("Not running in SLURM environment. This example requires SLURM.")
        logger.info("This script uses srun to launch monitoring agents on SLURM nodes.")
        return 1
    
    logger.info("Using srun for node communication (no SSH passwords required)")
    
    # Initialize the multinode monitor
    log_file = "jumper_multinode_srun_example.jsonl"
    monitor = SlurmMultinodeMonitor(log_path=log_file)
    
    try:
        # Start monitoring
        logger.info("Starting multinode performance monitor via srun...")
        monitor.start(interval=1.0)  # Sample every 1 second
        
        if not monitor.running:
            logger.error("Failed to start monitor. Check SLURM environment and srun availability.")
            return 1
        
        # Run the workload
        workload_duration = 30  # seconds
        run_workload_on_all_nodes(workload_duration)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        return 1
    finally:
        # Stop monitoring
        logger.info("Stopping monitor...")
        monitor.stop()
        
        # Analyze collected data
        analyze_performance_data(log_file)
    
    logger.info("=== Example completed ===")
    return 0


if __name__ == "__main__":
    exit(main())
