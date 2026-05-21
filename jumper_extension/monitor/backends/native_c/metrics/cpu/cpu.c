/*
 * CPU metric backend — reads /proc/<pid>/stat and /proc/stat.
 *
 * Maintains a per-PID tick cache across ticks to compute delta-based
 * CPU utilisation %.  System-level metrics use per-core deltas from
 * /proc/stat.
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cpu.h"

/* ------------------------------------------------------------------ */
/* Private state                                                      */
/* ------------------------------------------------------------------ */

typedef struct {
    int  pid;
    long prev_utime;
    long prev_stime;
    int  valid;
} pid_cpu_t;

static pid_cpu_t g_pid_cpu[MAX_PIDS];
static int       g_pid_cpu_count = 0;

/* Per-CPU previous ticks for system-level per-core utilisation */
static long g_prev_cpu_total[MAX_CPUS];
static long g_prev_cpu_idle[MAX_CPUS];
static int  g_prev_cpu_valid = 0;

/* Per-tick snapshot: current (utime, stime) for every observed PID */
typedef struct { int pid; long utime, stime; } cpu_snap_t;
static cpu_snap_t g_cpu_snap[MAX_PIDS];
static int        g_cpu_snap_count = 0;

/* ------------------------------------------------------------------ */
/* Private helpers                                                    */
/* ------------------------------------------------------------------ */

static int read_pid_cpu(int pid, long *utime, long *stime) {
    char path[64], buf[1024];
    snprintf(path, sizeof(path), "/proc/%d/stat", pid);
    if (read_file(path, buf, sizeof(buf)) < 0) return -1;
    /* Skip past the comm field (may contain spaces) to field 3 */
    char *p = strrchr(buf, ')');
    if (!p) return -1;
    p += 2; /* skip ') ' */
    long vals[16];
    int  idx = 3;
    char *tok = strtok(p, " ");
    while (tok && idx <= 15) {
        vals[idx++] = strtol(tok, NULL, 10);
        tok = strtok(NULL, " ");
    }
    if (idx <= 15) return -1;
    *utime = vals[14];
    *stime = vals[15];
    return 0;
}

static pid_cpu_t *get_pid_cpu(int pid) {
    for (int i = 0; i < g_pid_cpu_count; i++)
        if (g_pid_cpu[i].pid == pid) return &g_pid_cpu[i];
    if (g_pid_cpu_count >= MAX_PIDS) return NULL;
    pid_cpu_t *e = &g_pid_cpu[g_pid_cpu_count++];
    e->pid = pid;  e->prev_utime = 0;  e->prev_stime = 0;  e->valid = 0;
    return e;
}

static void snapshot_cpu_ticks(int *all_pids, int n_all) {
    g_cpu_snap_count = 0;
    for (int i = 0; i < n_all && g_cpu_snap_count < MAX_PIDS; i++) {
        int dup = 0;
        for (int j = 0; j < g_cpu_snap_count; j++)
            if (g_cpu_snap[j].pid == all_pids[i]) { dup = 1; break; }
        if (dup) continue;
        long ut = 0, st = 0;
        if (read_pid_cpu(all_pids[i], &ut, &st) < 0) continue;
        cpu_snap_t *s = &g_cpu_snap[g_cpu_snap_count++];
        s->pid = all_pids[i];  s->utime = ut;  s->stime = st;
    }
}

static cpu_snap_t *find_snap(int pid) {
    for (int i = 0; i < g_cpu_snap_count; i++)
        if (g_cpu_snap[i].pid == pid) return &g_cpu_snap[i];
    return NULL;
}

static double compute_pid_set_cpu(int *pids, int npids, double dt_sec) {
    double total = 0.0;
    for (int i = 0; i < npids; i++) {
        cpu_snap_t *s = find_snap(pids[i]);
        if (!s) continue;
        pid_cpu_t  *e = get_pid_cpu(pids[i]);
        if (!e || !e->valid) continue;
        long d = (s->utime - e->prev_utime) + (s->stime - e->prev_stime);
        total += (double)d / (g_clk_tck * dt_sec) * 100.0;
    }
    return total;
}

static void commit_pid_cpu_cache(void) {
    for (int i = 0; i < g_cpu_snap_count; i++) {
        pid_cpu_t *e = get_pid_cpu(g_cpu_snap[i].pid);
        if (!e) continue;
        e->prev_utime = g_cpu_snap[i].utime;
        e->prev_stime = g_cpu_snap[i].stime;
        e->valid = 1;
    }
}

static void prune_pid_cpu_cache(void) {
    for (int i = 0; i < g_pid_cpu_count; i++) {
        int found = 0;
        for (int j = 0; j < g_cpu_snap_count; j++)
            if (g_pid_cpu[i].pid == g_cpu_snap[j].pid) { found = 1; break; }
        if (!found) {
            g_pid_cpu[i] = g_pid_cpu[--g_pid_cpu_count];
            i--;
        }
    }
}

static int read_system_cpu_per_core(double *util_pct, int ncpus) {
    char buf[16384];
    if (read_file("/proc/stat", buf, sizeof(buf)) < 0) return -1;
    char *line = buf;
    int cpu_idx = 0;
    while ((line = strchr(line, '\n')) != NULL && cpu_idx < ncpus) {
        line++;
        if (strncmp(line, "cpu", 3) != 0 || !isdigit((unsigned char)line[3]))
            continue;
        long vals[8];
        char *p = line + 3;
        while (isdigit((unsigned char)*p)) p++;
        for (int i = 0; i < 8; i++) vals[i] = strtol(p, &p, 10);
        long total = 0;
        for (int i = 0; i < 8; i++) total += vals[i];
        long idle = vals[3] + vals[4]; /* idle + iowait */
        if (g_prev_cpu_valid && cpu_idx < MAX_CPUS) {
            long dt = total - g_prev_cpu_total[cpu_idx];
            long di = idle  - g_prev_cpu_idle[cpu_idx];
            util_pct[cpu_idx] = dt > 0
                ? (double)(dt - di) / (double)dt * 100.0 : 0.0;
        } else {
            util_pct[cpu_idx] = 0.0;
        }
        if (cpu_idx < MAX_CPUS) {
            g_prev_cpu_total[cpu_idx] = total;
            g_prev_cpu_idle[cpu_idx]  = idle;
        }
        cpu_idx++;
    }
    g_prev_cpu_valid = 1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* CCollector implementation                                          */
/* ------------------------------------------------------------------ */

static void cpu_setup(void) {}

static void cpu_snapshot(TickContext *ctx) {
    snapshot_cpu_ticks(ctx->all_pids, ctx->n_all);
}

static void cpu_emit_columns(FILE *fp, int level_idx, int n_cpus, int n_gpus) {
    (void)level_idx; (void)n_gpus;
    fprintf(fp, ",\"cpu_util_avg\",\"cpu_util_min\",\"cpu_util_max\"");
    for (int i = 0; i < n_cpus; i++)
        fprintf(fp, ",\"cpu_util_%d\"", i);
}

static void cpu_emit_sample(FILE *fp, int level_idx, TickContext *ctx, double dt) {
    double cpu_arr[MAX_CPUS];
    int ncpus_out;
    if (level_idx == LEVEL_SYSTEM) {
        ncpus_out = g_num_sys_cpus;
        read_system_cpu_per_core(cpu_arr, ncpus_out);
    } else {
        int *pids; int npids;
        switch (level_idx) {
            case LEVEL_PROCESS: pids = ctx->pids_proc;  npids = ctx->n_proc;  break;
            case LEVEL_USER:    pids = ctx->pids_user;  npids = ctx->n_user;  break;
            default:            pids = ctx->pids_slurm; npids = ctx->n_slurm; break;
        }
        ncpus_out = g_num_cpus;
        double per_core = compute_pid_set_cpu(pids, npids, dt) / g_num_cpus;
        for (int i = 0; i < ncpus_out; i++) cpu_arr[i] = per_core;
    }
    emit_per_device_agg(fp, "cpu_util_", cpu_arr, ncpus_out);
}

static void cpu_post_tick(void) {
    commit_pid_cpu_cache();
    prune_pid_cpu_cache();
}

/* ------------------------------------------------------------------ */
/* Public interface                                                   */
/* ------------------------------------------------------------------ */

void cpu_prime(int *pids, int npids, int num_sys_cpus) {
    for (int i = 0; i < npids; i++) {
        long ut = 0, st = 0;
        if (read_pid_cpu(pids[i], &ut, &st) < 0) continue;
        pid_cpu_t *e = get_pid_cpu(pids[i]);
        if (!e) continue;
        e->prev_utime = ut;
        e->prev_stime = st;
        e->valid = 1;
    }
    double dummy[MAX_CPUS];
    read_system_cpu_per_core(dummy, num_sys_cpus);
}

const CCollector cpu_collector = {
    .name         = "cpu",
    .setup        = cpu_setup,
    .snapshot     = cpu_snapshot,
    .emit_columns = cpu_emit_columns,
    .emit_sample  = cpu_emit_sample,
    .post_tick    = cpu_post_tick,
    .teardown     = NULL,
};
