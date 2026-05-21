/*
 * IO metric backend — reads /proc/<pid>/io and /proc/diskstats.
 *
 * Maintains per-level IO baselines across ticks and emits rates
 * (counts/s and bytes/s), mirroring the Python CumulativeRateHandler.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "io.h"

/* ------------------------------------------------------------------ */
/* Private types                                                      */
/* ------------------------------------------------------------------ */

typedef struct {
    long read_count;
    long write_count;
    long read_bytes;
    long write_bytes;
} io_counters_t;

typedef struct { int pid; io_counters_t io; } io_snap_t;

/* ------------------------------------------------------------------ */
/* Private state                                                      */
/* ------------------------------------------------------------------ */

static io_snap_t     g_io_snap[MAX_PIDS];
static int           g_io_snap_count = 0;

/* Per-level cumulative baseline for rate computation */
static io_counters_t g_io_prev[MAX_LEVELS];
static int           g_io_prev_valid[MAX_LEVELS];

/* ------------------------------------------------------------------ */
/* Private helpers                                                    */
/* ------------------------------------------------------------------ */

static io_counters_t read_pid_io(int pid) {
    io_counters_t c = {0, 0, 0, 0};
    char path[64], buf[1024];
    snprintf(path, sizeof(path), "/proc/%d/io", pid);
    if (read_file(path, buf, sizeof(buf)) < 0) return c;
    char *p;
    if ((p = strstr(buf, "syscr:")))       c.read_count  = strtol(p + 6,  NULL, 10);
    if ((p = strstr(buf, "syscw:")))       c.write_count = strtol(p + 6,  NULL, 10);
    if ((p = strstr(buf, "read_bytes:")))  c.read_bytes  = strtol(p + 11, NULL, 10);
    if ((p = strstr(buf, "write_bytes:"))) c.write_bytes = strtol(p + 12, NULL, 10);
    return c;
}

static io_counters_t find_io_snap(int pid) {
    for (int i = 0; i < g_io_snap_count; i++)
        if (g_io_snap[i].pid == pid) return g_io_snap[i].io;
    return (io_counters_t){0, 0, 0, 0};
}

static io_counters_t compute_pid_set_io(int *pids, int npids) {
    io_counters_t total = {0, 0, 0, 0};
    for (int i = 0; i < npids; i++) {
        io_counters_t c = find_io_snap(pids[i]);
        total.read_count  += c.read_count;
        total.write_count += c.write_count;
        total.read_bytes  += c.read_bytes;
        total.write_bytes += c.write_bytes;
    }
    return total;
}

/* Sum whole-disk devices from /proc/diskstats (matches psutil behaviour). */
static io_counters_t read_system_disk_io(void) {
    io_counters_t total = {0, 0, 0, 0};
    FILE *f = fopen("/proc/diskstats", "r");
    if (!f) return total;
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        unsigned int major, minor;
        char devname[128];
        long f1, f2, f3, f4, f5, f6, f7, f8;
        int n = sscanf(line,
            " %u %u %127s %ld %ld %ld %ld %ld %ld %ld %ld",
            &major, &minor, devname,
            &f1, &f2, &f3, &f4, &f5, &f6, &f7, &f8);
        if (n < 11) continue;
        /* Skip partitions (e.g. sda1, nvme0n1p1) — include whole disks only */
        int len = (int)strlen(devname);
        if (len > 0 && devname[len-1] >= '0' && devname[len-1] <= '9') {
            int j = len - 1;
            while (j > 0 && devname[j] >= '0' && devname[j] <= '9') j--;
            if (j > 0 && devname[j] == 'p' && j > 1) continue;
            if (j > 0 && ((devname[j] >= 'a' && devname[j] <= 'z') ||
                          (devname[j] >= 'A' && devname[j] <= 'Z'))) {
                if (!(devname[j] == 'n' && j > 0 &&
                      devname[j-1] >= '0' && devname[j-1] <= '9'))
                    continue;
            }
        }
        total.read_count  += f1;
        total.write_count += f5;
        total.read_bytes  += f3 * 512L;
        total.write_bytes += f7 * 512L;
    }
    fclose(f);
    return total;
}

/* ------------------------------------------------------------------ */
/* CCollector implementation                                          */
/* ------------------------------------------------------------------ */

static void io_setup(void) {}

static void io_snapshot(TickContext *ctx) {
    g_io_snap_count = 0;
    for (int i = 0; i < ctx->n_all; i++) {
        int dup = 0;
        for (int j = 0; j < g_io_snap_count; j++)
            if (g_io_snap[j].pid == ctx->all_pids[i]) { dup = 1; break; }
        if (dup || g_io_snap_count >= MAX_PIDS) continue;
        g_io_snap[g_io_snap_count].pid = ctx->all_pids[i];
        g_io_snap[g_io_snap_count].io  = read_pid_io(ctx->all_pids[i]);
        g_io_snap_count++;
    }
}

static void io_emit_columns(FILE *fp, int level_idx, int n_cpus, int n_gpus) {
    (void)level_idx; (void)n_cpus; (void)n_gpus;
    fprintf(fp, ",\"io_read_count\",\"io_write_count\",\"io_read\",\"io_write\"");
}

static void io_emit_sample(FILE *fp, int level_idx, TickContext *ctx, double dt) {
    io_counters_t io;
    if (level_idx == LEVEL_SYSTEM) {
        io = read_system_disk_io();
    } else {
        int *pids; int npids;
        switch (level_idx) {
            case LEVEL_PROCESS: pids = ctx->pids_proc;  npids = ctx->n_proc;  break;
            case LEVEL_USER:    pids = ctx->pids_user;  npids = ctx->n_user;  break;
            default:            pids = ctx->pids_slurm; npids = ctx->n_slurm; break;
        }
        io = compute_pid_set_io(pids, npids);
    }

    double io_rc = 0.0, io_wc = 0.0, io_rb = 0.0, io_wb = 0.0;
    if (g_io_prev_valid[level_idx] && dt > 0.0) {
        io_rc = (double)(io.read_count  - g_io_prev[level_idx].read_count)  / dt;
        io_wc = (double)(io.write_count - g_io_prev[level_idx].write_count) / dt;
        io_rb = (double)(io.read_bytes  - g_io_prev[level_idx].read_bytes)  / dt;
        io_wb = (double)(io.write_bytes - g_io_prev[level_idx].write_bytes) / dt;
        if (io_rc < 0.0) io_rc = 0.0;
        if (io_wc < 0.0) io_wc = 0.0;
        if (io_rb < 0.0) io_rb = 0.0;
        if (io_wb < 0.0) io_wb = 0.0;
    }
    g_io_prev[level_idx]       = io;
    g_io_prev_valid[level_idx] = 1;

    fprintf(fp, ",\"io_read_count\":%.6f,\"io_write_count\":%.6f"
            ",\"io_read\":%.6f,\"io_write\":%.6f",
            io_rc, io_wc, io_rb, io_wb);
}

const CCollector io_collector = {
    .name         = "io",
    .setup        = io_setup,
    .snapshot     = io_snapshot,
    .emit_columns = io_emit_columns,
    .emit_sample  = io_emit_sample,
    .post_tick    = NULL,
    .teardown     = NULL,
};
