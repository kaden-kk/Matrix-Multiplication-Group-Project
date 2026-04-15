#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

typedef unsigned long long ticks;
#define TILE_WIDTH 32 // Must be <= 32 since it squared is the number of threads for multiplication
#define NUM_BLOCKS 1024
#define THREADS_PER_BLOCK (TILE_WIDTH * TILE_WIDTH) // One thread per element in output tile
#define SHARED_BUFFER_SIZE 49152 // Max total size of shared memory buffers per block

extern "C"
void allocateMatrix(unsigned int numRows, unsigned int numCols, void*** matrix, int dataSize)
{
	cudaMallocManaged(matrix, numRows * sizeof(void*));
	for(unsigned int row = 0; row < numRows; row++)
	{
		cudaMallocManaged(&((*matrix)[row]), numCols * dataSize);
	}
}

extern "C"
void freeMatrix(unsigned int numRows, void** matrix)
{
	for(unsigned int row = 0; row < numRows; row++)
	{
		cudaFree(matrix[row]);
	}
	cudaFree(matrix);
}

void prefetchMatrix(unsigned int numRows, unsigned int numCols, void** matrix, int dataSize, int device)
{
	cudaMemPrefetchAsync(matrix, numRows * sizeof(void*), device, 0);
	for(unsigned int row = 0; row < numRows; row++)
	{
		cudaMemPrefetchAsync(matrix[row], numCols * dataSize, device, 0);
	}
}

__global__ void generateMatrixParallel(unsigned int numRows, unsigned int numCols, short** matrix)
{
	unsigned long blockId = blockIdx.x;

	unsigned int blocksPerRow = numCols / TILE_WIDTH;
	// Bring to next multiple of TILE_WIDTH so there are suffiecient blocks per row for rectangles
	if(numCols % TILE_WIDTH != 0)
	{
		blocksPerRow++;
	}

	// Grid stride until it goes out of bounds
	while(true)
	{
		unsigned int row = ((blockId / blocksPerRow) * TILE_WIDTH) + threadIdx.x;
		unsigned int col = (blockId % blocksPerRow) * TILE_WIDTH;

		// Grid-strided too far
		if(row >= numRows || col >= numCols)
		{
			break;
		}

		// Handle indices in this block only (for this thread's row)
		for(unsigned int i = 0; i < TILE_WIDTH && col + i < numCols; i++)
		{
			// matrix[row][col+i] = (rand() % 9) + 1; // Random number from 1 to 9 (does not work in device code)
			matrix[row][col+i] = 1;
		}

		blockId += gridDim.x;
	}
}

extern "C"
void generateMatrix(unsigned int numRows, unsigned int numCols, short** matrix)
{
	generateMatrixParallel<<<NUM_BLOCKS, TILE_WIDTH>>>(numRows, numCols, matrix);
	cudaDeviceSynchronize();
}

extern "C"
void multiplyMatricesSerial(unsigned int leftRows, unsigned int shared, unsigned int rightCols, short** left, short** right, 
	int** result, bool useTranspose)
{
	if(useTranspose)
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
					result[row][col] += left[row][k] * right[col][k];
				}
			}
		}
	}
	else
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
}

extern "C"
void printMatrixShort(unsigned int numRows, unsigned int numCols, short** matrix)
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
			printf("%hd", matrix[row][col]);
		}
		if(row == numRows - 1)
		{
			printf("]");
		}
		printf("\n");
	}
}

extern "C"
void printMatrixInt(unsigned int numRows, unsigned int numCols, int** matrix)
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
			printf("%d", matrix[row][col]);
		}
		if(row == numRows - 1)
		{
			printf("]");
		}
		printf("\n");
	}
}

// Transposes tile by tile (Unnecessary, just load into shared memory as if transposed)
__global__ void transpose(unsigned int originalRows, unsigned int originalCols, short** original, short** transpose)
{
	unsigned long blockId = blockIdx.x;

	unsigned int blocksPerRow = originalCols / TILE_WIDTH;
	// Bring to next multiple of TILE_WIDTH so there are suffiecient blocks per row for rectangles
	if(originalCols % TILE_WIDTH != 0)
	{
		blocksPerRow++;
	}

	// Grid stride until it goes out of bounds
	while(true)
	{
		unsigned int row = ((blockId / blocksPerRow) * TILE_WIDTH) + threadIdx.x;
		unsigned int col = (blockId % blocksPerRow) * TILE_WIDTH;

		// Grid-strided too far
		if(row >= originalRows || col >= originalCols)
		{
			break;
		}

		// Handle indices in this block only (for this thread's row)
		for(unsigned int i = 0; i < TILE_WIDTH && col + i < originalCols; i++)
		{
			transpose[col+i][row] = original[row][col+i];
		}

		blockId += gridDim.x;
	}
}

// Just swaps indices any time right matrix is mentioned
__global__ void multiplyMatricesParallelTranspose(unsigned int leftRows, unsigned int shared, unsigned int rightCols,
	short** left, short** right, int** result, bool useSharedMem)
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
					Bs[rowOffset][colOffset] = right[col][tiledRow];
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
					sum += left[row][k] * right[col][k];
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

__global__ void multiplyMatricesParallel(unsigned int leftRows, unsigned int shared, unsigned int rightCols,
	short** left, short** right, int** result, bool useSharedMem)
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
void transposeMatrix(unsigned int rows, unsigned int cols, short** original, short** result, int device)
{
	cudaSetDevice(device);

	prefetchMatrix(rows, cols, (void**)original, sizeof(short), device);
	prefetchMatrix(cols, rows, (void**)result, sizeof(short), device);

	transpose<<<NUM_BLOCKS, TILE_WIDTH>>>(rows, cols, original, result);

	cudaDeviceSynchronize();
}

extern "C"
void parallelMultiplication(unsigned int leftRows, unsigned int shared, unsigned int rightCols, short** left, 
	short** right, int** result, int device, bool useSharedMem, bool transpose)
{
	cudaSetDevice(device);

    prefetchMatrix(leftRows, shared, (void**)left, sizeof(short), device);

    if(transpose)
    	prefetchMatrix(shared, rightCols, (void**)right, sizeof(short), device);
    else
    	prefetchMatrix(rightCols, shared, (void**)right, sizeof(short), device);

    prefetchMatrix(leftRows, rightCols, (void**)result, sizeof(int), device);

    cudaDeviceSynchronize();

    if(transpose)
    {
    	multiplyMatricesParallelTranspose<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
        leftRows, shared, rightCols, left, right, result, useSharedMem);
    }
    else
    {
    	multiplyMatricesParallel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
        leftRows, shared, rightCols, left, right, result, useSharedMem);
    }

    cudaDeviceSynchronize();
}

// Returns 1 if any index fails, 0 if it's all correct
__global__ void parallelCheck(unsigned int numRows, unsigned int numCols, int** parallel, int** serial, int correct, int* result)
{
	unsigned long blockId = blockIdx.x;

	unsigned int blocksPerRow = numCols / TILE_WIDTH;
	// Bring to next multiple of TILE_WIDTH so there are suffiecient blocks per row for rectangles
	if(numCols % TILE_WIDTH != 0)
	{
		blocksPerRow++;
	}

	bool failed = false;

	// All values should be equal to correct
	if(serial == NULL)
	{
		// Grid stride until it goes out of bounds
		while(true)
		{
			unsigned int row = ((blockId / blocksPerRow) * TILE_WIDTH) + threadIdx.x;
			unsigned int col = (blockId % blocksPerRow) * TILE_WIDTH;

			// Grid-strided too far
			if(row >= numRows || col >= numCols || failed)
			{
				break;
			}

			// Handle indices in this block only (for this thread's row)
			for(unsigned int i = 0; i < TILE_WIDTH && col + i < numCols; i++)
			{
				if(parallel[row][col+i] != correct)
					failed = true;
			}

			blockId += gridDim.x;
		}
	}
	// Each index must match
	else
	{
		// Grid stride until it goes out of bounds
		while(true)
		{
			unsigned int row = ((blockId / blocksPerRow) * TILE_WIDTH) + threadIdx.x;
			unsigned int col = (blockId % blocksPerRow) * TILE_WIDTH;

			// Grid-strided too far
			if(row >= numRows || col >= numCols || failed)
			{
				break;
			}

			// Handle indices in this block only (for this thread's row)
			for(unsigned int i = 0; i < TILE_WIDTH && col + i < numCols; i++)
			{
				if(parallel[row][col+i] != serial[row][col+i])
					failed = true;
			}

			blockId += gridDim.x;
		}
	}
	if(failed)
		*result = 1;
	else
		*result = 0;
}

extern "C"
int checkResults(unsigned int numRows, unsigned int numCols, int** parallel, int** serial, int correct, int device)
{
	cudaSetDevice(device);

    prefetchMatrix(numRows, numCols, (void**)parallel, sizeof(int), device);
    prefetchMatrix(numRows, numCols, (void**)serial, sizeof(int), device);

    cudaDeviceSynchronize();

    int result;
    parallelCheck<<<NUM_BLOCKS, TILE_WIDTH>>>(numRows, numCols, parallel, serial, correct, &result);

    cudaDeviceSynchronize();

    return result;
}
