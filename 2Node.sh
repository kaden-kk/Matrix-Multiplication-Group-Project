#!/bin/bash -x
module load xl_r spectrum-mpi cuda

mpirun -np 2 /gpfs/u/scratch/PCPG/PCPGlmrn/matrix 4 4 4 1 0 1 1

mpirun -np 4 /gpfs/u/scratch/PCPG/PCPGlmrn/matrix 8 8 8 1 0 1 1

mpirun -np 2 /gpfs/u/scratch/PCPG/PCPGlmrn/matrix 256 256 256 1 0 1

mpirun -np 4 /gpfs/u/scratch/PCPG/PCPGlmrn/matrix 256 256 256 1 0 1

mpirun -np 8 /gpfs/u/scratch/PCPG/PCPGlmrn/matrix 256 256 256 1 0 1

mpirun -np 2 /gpfs/u/scratch/PCPG/PCPGlmrn/matrix 1024 1024 1024 1 0 1

mpirun -np 4 /gpfs/u/scratch/PCPG/PCPGlmrn/matrix 1024 1024 1024 1 0 1
