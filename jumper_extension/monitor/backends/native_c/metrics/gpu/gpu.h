#ifndef JUMPER_METRICS_GPU_H
#define JUMPER_METRICS_GPU_H

#include "monitor.h"

extern const CCollector gpu_collector;

/* Accessors for GPU metadata populated during setup().
   Used by emit_ready() in collector_main.c. */
int         gpu_num_gpus(void);
double      gpu_memory_gb(void);
const char *gpu_name(void);

#endif /* JUMPER_METRICS_GPU_H */
