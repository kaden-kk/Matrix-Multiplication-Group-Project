#!/bin/bash -x
module load xl_r spectrum-mpi cuda

mpirun -np 4 /gpfs/u/scratch/PCPG/PCPGlmrn/matrix 4 4 4 1 0 1 1

mpirun -np 4 /gpfs/u/scratch/PCPG/PCPGlmrn/matrix 8 8 8 1 0 1 1

mpirun -np 4 /gpfs/u/scratch/PCPG/PCPGlmrn/matrix 256 256 256 1 0 1

mpirun -np 16 /gpfs/u/scratch/PCPG/PCPGlmrn/matrix 256 256 256 1 0 1

mpirun -np 32 /gpfs/u/scratch/PCPG/PCPGlmrn/matrix 256 256 256 1 0 1

mpirun -np 32 /gpfs/u/scratch/PCPG/PCPGlmrn/matrix 1024 1024 1024 1 0 1

mpirun -np 32 /gpfs/u/scratch/PCPG/PCPGlmrn/matrix 1024 1024 1024 1 0 1

mpirun -np 32 /gpfs/u/scratch/PCPG/PCPGlmrn/matrix 2048 2048 2048 1 0 1

mpirun -np 32 /gpfs/u/scratch/PCPG/PCPGlmrn/matrix 2048 2048 2048 1 1 1

mpirun -np 32 /gpfs/u/scratch/PCPG/PCPGlmrn/matrix 4096 4096 4096 1 0 1

mpirun -np 32 /gpfs/u/scratch/PCPG/PCPGlmrn/matrix 4096 4096 4096 1 1 1
