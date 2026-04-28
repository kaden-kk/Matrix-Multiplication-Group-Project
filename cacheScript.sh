#!/bin/bash -x
module load xl_r spectrum-mpi cuda

/gpfs/u/scratch/PCPG/PCPGlmrn/cache 32 32 32 0 1

/gpfs/u/scratch/PCPG/PCPGlmrn/cache 128 128 128 0 1

/gpfs/u/scratch/PCPG/PCPGlmrn/cache 256 256 256 0 1

/gpfs/u/scratch/PCPG/PCPGlmrn/cache 512 512 512 0 1

/gpfs/u/scratch/PCPG/PCPGlmrn/cache 1024 1024 1024 0 1

/gpfs/u/scratch/PCPG/PCPGlmrn/cache 2048 2048 2048 0 1

/gpfs/u/scratch/PCPG/PCPGlmrn/cache 4096 4096 4096 0 1

/gpfs/u/scratch/PCPG/PCPGlmrn/cache 8192 8192 8192 0 0

/gpfs/u/scratch/PCPG/PCPGlmrn/cache 16384 16384 16384 0 0

/gpfs/u/scratch/PCPG/PCPGlmrn/cache 32768 32768 32768 0 0
