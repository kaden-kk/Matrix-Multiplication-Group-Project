#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

typedef unsigned long long ticks;
#define TILE_WIDTH 32 // Must be <= 32 since it squared is the number of threads for multiplication
#define NUM_BLOCKS 1024
#define THREADS_PER_BLOCK (TILE_WIDTH * TILE_WIDTH) // One thread per element in output tile
#define SHARED_BUFFER_SIZE 48000 // Online, lowest shared memory size is 48 kB per block (should experimentally determine upper limit)
/* TILE_WIDTH squared rows, size of long is 8, this gives number of elements allowed per each row at once in shared memory
   as the nearest rounded down multiple of TILE_WIDTH for even division */


// IBM POWER9 System clock with 512MHZ resolution.
static __inline__ ticks getticks(void)
{
    unsigned int tbl, tbu0, tbu1;

    do {
        __asm__ __volatile__ ("mftbu %0" : "=r"(tbu0));
        __asm__ __volatile__ ("mftb %0" : "=r"(tbl));
        __asm__ __volatile__ ("mftbu %0" : "=r"(tbu1));
    } while (tbu0 != tbu1);

    return (((unsigned long long)tbu0) << 32) | tbl;
}

extern "C"
void allocateMatrix(unsigned int numRows, unsigned int numCols, long*** matrix, bool setZero, int device)
{
	cudaMallocManaged(matrix, numRows * sizeof(long*));
	for(unsigned int row = 0; row < numRows; row++)
	{
		cudaMallocManaged(&((*matrix)[row]), numCols * sizeof(long));
	}

	// Set all elements to 0
	if(setZero)
	{
		for(unsigned int row = 0; row < numRows; row++)
		{
			for(unsigned int col = 0; col < numCols; col++)
			{
				(*matrix)[row][col] = 0;
			}
		}
	}
}

extern "C"
void freeMatrix(unsigned int numRows, long** matrix)
{
	for(unsigned int row = 0; row < numRows; row++)
	{
		cudaFree(matrix[row]);
	}
	cudaFree(matrix);
}

void prefetchMatrix(unsigned int numRows, unsigned int numCols, long** matrix, int device)
{
	cudaMemPrefetchAsync(matrix, numRows * sizeof(long*), device, 0);
	for(unsigned int row = 0; row < numRows; row++)
	{
		cudaMemPrefetchAsync(matrix[row], numCols * sizeof(long), device, 0);
	}
}

extern "C"
void generateMatrix(unsigned int numRows, unsigned int numCols, long** matrix)
{
	for(unsigned int row = 0; row < numRows; row++)
	{
		for(unsigned int col = 0; col < numCols; col++)
		{
			matrix[row][col] = (rand() % 9) + 1; // Random number from 1 to 9 
		}
	}
}

void multiplyMatricesSerial(unsigned int leftRows, unsigned int shared, unsigned int rightCols, long** left, long** right, long** result)
{
	// Iterate through each row in left matrix
	for(unsigned int row = 0; row < leftRows; row++)
	{
		// Iterate through each column in right matrix
		for(unsigned int col = 0; col < rightCols; col++)
		{
			// Do dot product for each element in this row and col
			for(unsigned int k = 0; k < shared; k++)
			{
				result[row][col] += left[row][k] * right[k][col];
			}
		}
	}
}

extern "C"
void printMatrix(unsigned int numRows, unsigned int numCols, long** matrix)
{
	printf("[");
	for(int row = 0; row < numRows; row++)
	{
		if(row == 0)
		{
			printf(" ");
		}
		for(int col = 0; col < numCols; col++)
		{
			// Do consistent spacing
			if(col == 0)
			{
				if(row != 0)
					printf("  ");
			}
			else
			{
				printf("\t");
			}
			printf("%ld", matrix[row][col]);
		}
		if(row == numRows - 1)
		{
			printf("]");
		}
		printf("\n");
	}
}

// Transposes tile by tile (Unnecessary, just load into shared memory as if transposed)
// __global__ void transpose(unsigned int originalRows, unsigned int originalCols, long** original, long** transpose)
// {
// 	unsigned long blockId = blockIdx.x;

// 	unsigned int colDivisor = originalCols;
// 	// Bring to next multiple of TILE_WIDTH so there are suffiecient blocks per row for rectangles
// 	if(colDivisor % TILE_WIDTH != 0)
// 	{
// 		colDivisor = ((colDivisor / TILE_WIDTH) + 1) * TILE_WIDTH;
// 	}

// 	// Grid stride until it goes out of bounds
// 	while(true)
// 	{
// 		unsigned int row = (blockId / colDivisor)+ threadIdx.x;
// 		unsigned int col = (blockId % (colDivisor / TILE_WIDTH)) * TILE_WIDTH;

// 		// Grid-strided too far
// 		if(row >= originalRows || col >= originalCols)
// 		{
// 			break;
// 		}

// 		// Handle indices in this block only (for this thread's row)
// 		for(unsigned int i = 0; i < TILE_WIDTH && col + i < originalCols; i++)
// 		{
// 			transpose[col+i][row] = original[row][col+i];
// 		}

// 		blockId += gridDim.x;
// 	}
// }

__global__ void multiplyMatricesParallel(unsigned int leftRows, unsigned int shared, unsigned int rightCols,
	long** left, long** right, long** result, bool useSharedMem)
{
	unsigned long blockId = blockIdx.x;

	unsigned int blocksPerRow = (rightCols + TILE_WIDTH - 1) / TILE_WIDTH;

	while (true)
	{
		unsigned int firstRow = (blockId / blocksPerRow) * TILE_WIDTH;
		unsigned int firstCol = (blockId % blocksPerRow) * TILE_WIDTH;

		unsigned int rowOffset = threadIdx.x / TILE_WIDTH;
		unsigned int colOffset = threadIdx.x % TILE_WIDTH;

		unsigned int row = firstRow + rowOffset;
		unsigned int col = firstCol + colOffset;

		if (firstRow >= leftRows || firstCol >= rightCols)
			break;

		long sum = 0;

		if (useSharedMem)
		{
			// Shared memory tiles
			__shared__ long As[TILE_WIDTH][TILE_WIDTH];
			__shared__ long Bs[TILE_WIDTH][TILE_WIDTH];

			for (unsigned int tile = 0; tile < (shared + TILE_WIDTH - 1) / TILE_WIDTH; tile++)
			{
				unsigned int tiledCol = tile * TILE_WIDTH + colOffset;
				unsigned int tiledRow = tile * TILE_WIDTH + rowOffset;

				// Load A tile
				if (row < leftRows && tiledCol < shared)
					As[rowOffset][colOffset] = left[row][tiledCol];
				else
					As[rowOffset][colOffset] = 0;

				// Load B tile
				if (tiledRow < shared && col < rightCols)
					Bs[rowOffset][colOffset] = right[tiledRow][col];
				else
					Bs[rowOffset][colOffset] = 0;

				__syncthreads();

				for (int k = 0; k < TILE_WIDTH; k++)
				{
					sum += As[rowOffset][k] * Bs[k][colOffset];
				}

				__syncthreads();
			}
		}
		else
		{
			// Non-shared memory version
			if (row < leftRows && col < rightCols)
			{
				for (unsigned int k = 0; k < shared; k++)
				{
					sum += left[row][k] * right[k][col];
				}
			}
		}

		if (row < leftRows && col < rightCols)
		{
			result[row][col] += sum;
		}

		blockId += gridDim.x;
	}
}

extern "C"
void parallelMultiplication(unsigned int leftRows, unsigned int shared, unsigned int rightCols, long** left, 
	long** right, long** result, int device, bool useSharedMem)
{
    prefetchMatrix(leftRows, shared, left, device);
    prefetchMatrix(shared, rightCols, right, device);
    prefetchMatrix(leftRows, rightCols, result, device);

    cudaDeviceSynchronize();

    ticks start = getticks();

    multiplyMatricesParallel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
        leftRows, shared, rightCols, left, right, result, useSharedMem);

    cudaDeviceSynchronize();

    ticks finish = getticks();

    printf("GPU kernel time: %lf\n",
        (double)(finish - start) / 512000000.0);
}
