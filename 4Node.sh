#!/bin/bash -x

echo "Running on nodes:"
scontrol show hostnames $SLURM_NODELIST
echo "-----"

module load xl_r spectrum-mpi cuda

make

mpirun -np 4 /gpfs/u/home/PCPG/PCPGkstr/scratch/group/matrix 4 4 4 1 0 1 1

mpirun -np 4 /gpfs/u/home/PCPG/PCPGkstr/scratch/group/matrix 8 8 8 1 0 1 1

mpirun -np 4 /gpfs/u/home/PCPG/PCPGkstr/scratch/group/matrix 256 256 256 1 0 1

mpirun -np 16 /gpfs/u/home/PCPG/PCPGkstr/scratch/group/matrix 256 256 256 1 0 1

mpirun -np 32 /gpfs/u/home/PCPG/PCPGkstr/scratch/group/matrix 256 256 256 1 0 1

mpirun -np 32 /gpfs/u/home/PCPG/PCPGkstr/scratch/group/matrix 1024 1024 1024 1 0 1

mpirun -np 32 /gpfs/u/home/PCPG/PCPGkstr/scratch/group/matrix 1024 1024 1024 1 0 1

mpirun -np 32 /gpfs/u/home/PCPG/PCPGkstr/scratch/group/matrix 2048 2048 2048 1 0 1

mpirun -np 32 /gpfs/u/home/PCPG/PCPGkstr/scratch/group/matrix 2048 2048 2048 1 1 1

mpirun -np 32 /gpfs/u/home/PCPG/PCPGkstr/scratch/group/matrix 4096 4096 4096 1 0 1

mpirun -np 32 /gpfs/u/home/PCPG/PCPGkstr/scratch/group/matrix 4096 4096 4096 1 1 1
