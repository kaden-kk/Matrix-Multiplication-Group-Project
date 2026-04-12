matrix: mpi.c kernel.cu
	mpixlc -O3 mpi.c -c -o mpi-xlc.o
	nvcc -O3 -arch=sm_70 kernel.cu -c -o kernel-nvcc.o
	mpixlc -O3 mpi-xlc.o kernel-nvcc.o \
	-o matrix \
	-L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++
