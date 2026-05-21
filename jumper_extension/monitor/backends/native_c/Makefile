CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra
LDFLAGS ?= -lm -ldl
TARGET  := jumper_collector

DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

SRCS := \
    monitor.c                 \
    metrics/cpu/cpu.c         \
    metrics/memory/memory.c   \
    metrics/io/io.c           \
    metrics/gpu/gpu.c

OBJS := $(patsubst %.c,$(DIR)%.o,$(SRCS))

.PHONY: all clean

all: $(DIR)$(TARGET)

$(DIR)$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(DIR)%.o: $(DIR)%.c
	$(CC) $(CFLAGS) -I$(DIR) -c -o $@ $<

clean:
	rm -f $(OBJS) $(DIR)$(TARGET)
