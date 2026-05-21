/*
 * Shared types and CCollector interface for the JUmPER native C monitor.
 *
 * Each metric backend (cpu, memory, io, gpu) lives in metrics/<name>/ and
 * implements the CCollector function-pointer interface.  Instances are
 * registered in the g_registry table in monitor_main.c.
 *
 * Active collectors are selected at startup via --collectors (comma-separated
 * list of names), driven by config/c/collectors.yaml on the Python side.
 */

#ifndef JUMPER_MONITOR_H
#define JUMPER_MONITOR_H

#include <stdio.h>

/* ------------------------------------------------------------------ */
/* Tunables                                                           */
/* ------------------------------------------------------------------ */
#define MAX_PIDS    4096
#define MAX_CPUS     512
#define MAX_LEVELS     4
#define LEVEL_PROCESS  0
#define LEVEL_USER     1
#define LEVEL_SYSTEM   2
#define LEVEL_SLURM    3

/* ------------------------------------------------------------------ */
/* Globals defined in monitor.c                                       */
/* ------------------------------------------------------------------ */
extern int  g_num_cpus;     /* CPUs available to the target process   */
extern int  g_num_sys_cpus; /* total online CPUs                      */
extern long g_clk_tck;      /* sysconf(_SC_CLK_TCK)                   */

/* ------------------------------------------------------------------ */
/* TickContext — per-tick PID state passed to every collector          */
/* ------------------------------------------------------------------ */
typedef struct {
    int pids_proc[MAX_PIDS];  int n_proc;
    int pids_user[MAX_PIDS];  int n_user;
    int pids_slurm[MAX_PIDS]; int n_slurm;
    int all_pids[MAX_PIDS];   int n_all;  /* union of above sets */
} TickContext;

/* ------------------------------------------------------------------ */
/* CCollector interface                                               */
/* ------------------------------------------------------------------ */
typedef struct {
    const char *name;

    /* Called once at startup after argument parsing. */
    void (*setup)(void);

    /* Called once per tick before the level loop.
       Reads /proc and fills the collector's private snapshot arrays. */
    void (*snapshot)(TickContext *ctx);

    /* Emits comma-prefixed column names for one level (no trailing comma).
       Called during the ready handshake for each active level. */
    void (*emit_columns)(FILE *fp, int level_idx, int n_cpus, int n_gpus);

    /* Emits comma-prefixed sample fields for one level (no trailing comma).
       Called inside the level loop for each active level. */
    void (*emit_sample)(FILE *fp, int level_idx, TickContext *ctx, double dt);

    /* Optional: called once per tick after all emit_sample calls.
       The CPU collector uses this to commit its per-PID tick cache.
       Set to NULL if unused. */
    void (*post_tick)(void);

    /* Optional: called at process exit for resource cleanup.
       The GPU collector uses this to shut down NVML.  NULL if unused. */
    void (*teardown)(void);
} CCollector;

/* ------------------------------------------------------------------ */
/* Shared utilities (defined in monitor.c)                            */
/* ------------------------------------------------------------------ */

/* Read a small file into buf.  Returns bytes read, or -1 on error. */
int read_file(const char *path, char *buf, size_t bufsz);

/* Emit avg/min/max + per-device values, comma-prefixed.
   Writes nothing when n <= 0. */
void emit_per_device_agg(FILE *fp, const char *prefix,
                         const double *vals, int n);

#endif /* JUMPER_MONITOR_H */
