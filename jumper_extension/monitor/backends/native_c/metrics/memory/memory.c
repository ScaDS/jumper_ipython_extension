/*
 * Memory metric backend — reads /proc/<pid>/statm and /proc/meminfo.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "memory.h"

/* ------------------------------------------------------------------ */
/* Private state                                                      */
/* ------------------------------------------------------------------ */

typedef struct { int pid; long rss_kb; } mem_snap_t;
static mem_snap_t g_mem_snap[MAX_PIDS];
static int        g_mem_snap_count = 0;

/* ------------------------------------------------------------------ */
/* Private helpers                                                    */
/* ------------------------------------------------------------------ */

static long read_pid_rss_kb(int pid) {
    char path[64], buf[128];
    snprintf(path, sizeof(path), "/proc/%d/statm", pid);
    if (read_file(path, buf, sizeof(buf)) < 0) return 0;
    long pages, dummy;
    if (sscanf(buf, "%ld %ld", &dummy, &pages) != 2) return 0;
    return pages * (sysconf(_SC_PAGESIZE) / 1024);
}

static long find_mem_snap(int pid) {
    for (int i = 0; i < g_mem_snap_count; i++)
        if (g_mem_snap[i].pid == pid) return g_mem_snap[i].rss_kb;
    return 0;
}

static double compute_pid_set_memory_gb(int *pids, int npids) {
    long total_kb = 0;
    for (int i = 0; i < npids; i++)
        total_kb += find_mem_snap(pids[i]);
    return (double)total_kb / (1024.0 * 1024.0);
}

static double system_memory_used_gb(void) {
    char buf[4096];
    if (read_file("/proc/meminfo", buf, sizeof(buf)) < 0) return 0.0;
    long total = 0, avail = 0;
    char *p = strstr(buf, "MemTotal:");
    if (p) total = strtol(p + 9, NULL, 10);
    p = strstr(buf, "MemAvailable:");
    if (p) avail = strtol(p + 13, NULL, 10);
    return (double)(total - avail) / (1024.0 * 1024.0);
}

/* ------------------------------------------------------------------ */
/* CCollector implementation                                          */
/* ------------------------------------------------------------------ */

static void mem_setup(void) {}

static void mem_snapshot(TickContext *ctx) {
    g_mem_snap_count = 0;
    for (int i = 0; i < ctx->n_all && g_mem_snap_count < MAX_PIDS; i++) {
        int dup = 0;
        for (int j = 0; j < g_mem_snap_count; j++)
            if (g_mem_snap[j].pid == ctx->all_pids[i]) { dup = 1; break; }
        if (dup) continue;
        g_mem_snap[g_mem_snap_count].pid    = ctx->all_pids[i];
        g_mem_snap[g_mem_snap_count].rss_kb = read_pid_rss_kb(ctx->all_pids[i]);
        g_mem_snap_count++;
    }
}

static void mem_emit_columns(FILE *fp, int level_idx, int n_cpus, int n_gpus) {
    (void)level_idx; (void)n_cpus; (void)n_gpus;
    fprintf(fp, ",\"memory\"");
}

static void mem_emit_sample(FILE *fp, int level_idx, TickContext *ctx, double dt) {
    (void)dt;
    double memory;
    if (level_idx == LEVEL_SYSTEM) {
        memory = system_memory_used_gb();
    } else {
        int *pids; int npids;
        switch (level_idx) {
            case LEVEL_PROCESS: pids = ctx->pids_proc;  npids = ctx->n_proc;  break;
            case LEVEL_USER:    pids = ctx->pids_user;  npids = ctx->n_user;  break;
            default:            pids = ctx->pids_slurm; npids = ctx->n_slurm; break;
        }
        memory = compute_pid_set_memory_gb(pids, npids);
    }
    fprintf(fp, ",\"memory\":%.6f", memory);
}

const CCollector memory_collector = {
    .name         = "memory",
    .setup        = mem_setup,
    .snapshot     = mem_snapshot,
    .emit_columns = mem_emit_columns,
    .emit_sample  = mem_emit_sample,
    .post_tick    = NULL,
    .teardown     = NULL,
};
