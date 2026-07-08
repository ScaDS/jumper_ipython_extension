If the hardware includes one or more GPUs and the code contains a workload that can be efficiently parallelized:
  - Include at least one option that refactors the CPU implementation to run on the GPU, whichever fits the existing code with the least disruption.
  - Prefer using preinstalled libraries (check the list of "Available libraries").
Skip this if no GPU is available or the workload is not a good parallelization candidate (e.g. it is I/O-bound or inherently sequential).
