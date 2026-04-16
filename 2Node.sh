#!/bin/bash -x

echo "Running on nodes:"
scontrol show hostnames $SLURM_NODELIST
echo "-----"

module load xl_r spectrum-mpi cuda

make

mpirun -np 2 /gpfs/u/home/PCPG/PCPGkstr/scratch/group/matrix 4 4 4 1 0 1 1

mpirun -np 4 /gpfs/u/home/PCPG/PCPGkstr/scratch/group/matrix 8 8 8 1 0 1 1

mpirun -np 2 /gpfs/u/home/PCPG/PCPGkstr/scratch/group/matrix 256 256 256 1 0 1

mpirun -np 4 /gpfs/u/home/PCPG/PCPGkstr/scratch/group/matrix 256 256 256 1 0 1

mpirun -np 8 /gpfs/u/home/PCPG/PCPGkstr/scratch/group/matrix 256 256 256 1 0 1

mpirun -np 2 /gpfs/u/home/PCPG/PCPGkstr/scratch/group/matrix 1024 1024 1024 1 0 1

mpirun -np 4 /gpfs/u/home/PCPG/PCPGkstr/scratch/group/matrix 1024 1024 1024 1 0 1
