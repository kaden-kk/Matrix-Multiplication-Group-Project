CC = mpicc
NVCC = nvcc

CFLAGS = -O3
CUDAFLAGS = -O3 -arch=sm_70

all: matrix

kernel.o: kernel.cu
	$(NVCC) $(CUDAFLAGS) -c kernel.cu

mpi.o: mpi.c
	$(CC) $(CFLAGS) -c mpi.c

matrix: mpi.o kernel.o
	$(NVCC) -ccbin mpicc mpi.o kernel.o -o matrix

clean:
	rm -f *.o matrix

