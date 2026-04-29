all: matrix cacheTest

matrix: mpi.c kernel.cu
	mpicc -O3 -I/usr/local/cuda-11.2/include -c mpi.c -o mpi.o
	nvcc -O3 -arch=sm_70 -c kernel.cu -o kernel.o
	nvcc -ccbin mpicc mpi.o kernel.o -o matrix \
	-L/usr/local/cuda-11.2/lib64 -lcudadevrt -lcudart

cacheTest: cacheTest.cu
	nvcc -O3 -arch=sm_70 cacheTest.cu -o cacheTest \
	-L/usr/local/cuda-11.2/lib64 -lcudadevrt -lcudart

clean:
	rm -f *.o matrix cacheTest