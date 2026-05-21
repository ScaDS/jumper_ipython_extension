#ifndef JUMPER_METRICS_CPU_H
#define JUMPER_METRICS_CPU_H

#include "monitor.h"

extern const CCollector cpu_collector;

/* Warm up the per-PID tick cache before the main loop so the first
   sample produces a valid CPU delta instead of zero.  Pass the initial
   process PID set and the number of system CPUs for the baseline read. */
void cpu_prime(int *pids, int npids, int num_sys_cpus);

#endif /* JUMPER_METRICS_CPU_H */
