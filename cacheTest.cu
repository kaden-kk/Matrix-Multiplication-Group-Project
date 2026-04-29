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
#define THREADS_PER_BLOCK TILE_WIDTH * TILE_WIDTH // One thread per element in output tile

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

int allocateMatrix(unsigned int numRows, unsigned int numCols, void*** matrix, int dataSize)
{
	cudaError_t err;
	err = cudaMallocManaged(matrix, numRows * sizeof(void*));
	if(err != cudaSuccess)
	{
		fprintf(stderr, "CudaMallocManaged failed\n");
		return 1;
	}
	for(unsigned int row = 0; row < numRows; row++)
	{
		err = cudaMallocManaged(&((*matrix)[row]), numCols * dataSize);
		if(err != cudaSuccess)
		{
			fprintf(stderr, "CudaMallocManaged failed\n");
			return 1;
		}
	}
	return 0;
}

void freeMatrix(unsigned int numRows, void** matrix)
{
	if(matrix == NULL)
		return;

	for(unsigned int row = 0; row < numRows; row++)
	{
		cudaFree(matrix[row]);
	}
	cudaFree(matrix);
}

void prefetchMatrix(unsigned int numRows, unsigned int numCols, void** matrix, int dataSize, int device)
{
	int totalSize = (numRows * sizeof(void*)) + (numRows * numCols * dataSize);
	cudaMemPrefetchAsync(matrix, totalSize, device, 0);
}

__global__ void generateMatrixParallel(unsigned int numRows, unsigned int numCols, short** matrix, bool uniform)
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
		unsigned int firstRow = (blockId / blocksPerRow) * TILE_WIDTH;
		unsigned int col = (blockId % blocksPerRow) * TILE_WIDTH;

		unsigned int row = firstRow + threadIdx.x;

		// Grid-strided too far
		if(firstRow >= numRows || col >= numCols)
		{
			break;
		}

		if(row < numRows)
		{
			// Set all values to 1
			if(uniform)
			{
				// Handle indices in this block only (for this thread's row)
				for(unsigned int i = 0; i < TILE_WIDTH && col + i < numCols; i++)
				{
					matrix[row][col+i] = 1;
				}
			}
			// Values aren't random, but differ enough to reveal problems with multiplicaiton / tranposing
			else
			{
				// Handle indices in this block only (for this thread's row)
				for(unsigned int i = 0; i < TILE_WIDTH && col + i < numCols; i++)
				{
					matrix[row][col+i] = ((col + i) % TILE_WIDTH) + 1;
				}
			}
		}

		blockId += gridDim.x;
	}
}

void generateMatrix(unsigned int numRows, unsigned int numCols, short** matrix, bool uniform)
{
	generateMatrixParallel<<<NUM_BLOCKS, TILE_WIDTH>>>(numRows, numCols, matrix, uniform);
	cudaDeviceSynchronize();
}

void multiplyMatricesSerial(unsigned int leftRows, unsigned int shared, unsigned int rightCols, short** left, short** right, int** result)
{
	// Iterate through each row in left matrix
	for(unsigned int row = 0; row < leftRows; row++)
	{
		// Iterate through each column in right matrix
		for(unsigned int col = 0; col < rightCols; col++)
		{
			int sum = 0;
			// Do dot product for each element in this row and col
			for(unsigned int k = 0; k < shared; k++)
			{
				sum += left[row][k] * right[k][col];
			}
			result[row][col] = sum;
		}
	}
}

void multiplyMatricesSerialRightTranspose(unsigned int leftRows, unsigned int shared, unsigned int rightCols, short** left, short** right, int** result)
{
	// Iterate through each row in left matrix
	for(unsigned int row = 0; row < leftRows; row++)
	{
		// Iterate through each column in right matrix
		for(unsigned int col = 0; col < rightCols; col++)
		{
			int sum = 0;
			// Do dot product for each element in this row and col
			for(unsigned int k = 0; k < shared; k++)
			{
				sum += left[row][k] * right[col][k];
			}
			result[row][col] = sum;
		}
	}
}

void multiplyMatricesSerialLeftTranspose(unsigned int leftRows, unsigned int shared, unsigned int rightCols, short** left, short** right, int** result)
{
	// Iterate through each row in left matrix
	for(unsigned int row = 0; row < leftRows; row++)
	{
		// Iterate through each column in right matrix
		for(unsigned int col = 0; col < rightCols; col++)
		{
			int sum = 0;
			// Do dot product for each element in this row and col
			for(unsigned int k = 0; k < shared; k++)
			{
				sum += left[k][row] * right[k][col];
			}
			result[row][col] = sum;
		}
	}
}

void multiplyMatricesSerialDoubleTranspose(unsigned int leftRows, unsigned int shared, unsigned int rightCols, short** left, short** right, int** result)
{
	// Iterate through each row in left matrix
	for(unsigned int row = 0; row < leftRows; row++)
	{
		// Iterate through each column in right matrix
		for(unsigned int col = 0; col < rightCols; col++)
		{
			int sum = 0;
			// Do dot product for each element in this row and col
			for(unsigned int k = 0; k < shared; k++)
			{
				sum += left[k][row] * right[col][k];
			}
			result[row][col] = sum;
		}
	}
}

__global__ void multiplyMatricesParallel(unsigned int leftRows, unsigned int shared, unsigned int rightCols, 
	short** left, short** right, int** result)
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

		if (row < leftRows && col < rightCols)
		{
			int sum = 0;

			for (unsigned int k = 0; k < shared; k++)
			{
				sum += left[row][k] * right[k][col];
			}

			result[row][col] = sum;
		}

		blockId += gridDim.x;
	}
}

__global__ void multiplyMatricesParallelRightTranspose(unsigned int leftRows, unsigned int shared, unsigned int rightCols, 
	short** left, short** right, int** result)
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

		if (row < leftRows && col < rightCols)
		{
			int sum = 0;

			for (unsigned int k = 0; k < shared; k++)
			{
				sum += left[row][k] * right[col][k];
			}

			result[row][col] = sum;
		}

		blockId += gridDim.x;
	}
}

__global__ void multiplyMatricesParallelLeftTranspose(unsigned int leftRows, unsigned int shared, unsigned int rightCols, 
	short** left, short** right, int** result)
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

		if (row < leftRows && col < rightCols)
		{
			int sum = 0;

			for (unsigned int k = 0; k < shared; k++)
			{
				sum += left[k][row] * right[k][col];
			}

			result[row][col] = sum;
		}

		blockId += gridDim.x;
	}
}

__global__ void multiplyMatricesParallelDoubleTranspose(unsigned int leftRows, unsigned int shared, unsigned int rightCols, 
	short** left, short** right, int** result)
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

		if (row < leftRows && col < rightCols)
		{
			int sum = 0;

			for (unsigned int k = 0; k < shared; k++)
			{
				sum += left[k][row] * right[col][k];
			}

			result[row][col] = sum;
		}

		blockId += gridDim.x;
	}
}

void parallelMultiplication(unsigned int leftRows, unsigned int shared, unsigned int rightCols, short** left, 
	short** right, int** result, int device, bool prefetchRightFirst)
{
	if(prefetchRightFirst)
	{
		prefetchMatrix(shared, rightCols, (void**)right, sizeof(short), device);
		prefetchMatrix(leftRows, shared, (void**)left, sizeof(short), device);
	}
	else
	{
		prefetchMatrix(leftRows, shared, (void**)left, sizeof(short), device);
		prefetchMatrix(shared, rightCols, (void**)right, sizeof(short), device);
	}
	prefetchMatrix(leftRows, rightCols, (void**)result, sizeof(int), device);

	printf("\nParallel timings:\n");
	cudaDeviceSynchronize();

	ticks start = getticks();
	multiplyMatricesParallel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(leftRows, shared, rightCols, left, right, result);

	cudaDeviceSynchronize();
	ticks end = getticks();
	printf("\tTime for normal: %lf \n", (double)(end - start) / (double)512000000.0);

	cudaDeviceSynchronize();

	start = getticks();
	multiplyMatricesParallelRightTranspose<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(leftRows, shared, rightCols, left, right, result);

	cudaDeviceSynchronize();
	end = getticks();
	printf("\tTime for right transpose: %lf \n", (double)(end - start) / (double)512000000.0);

	cudaDeviceSynchronize();

	start = getticks();
	multiplyMatricesParallelLeftTranspose<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(leftRows, shared, rightCols, left, right, result);

	cudaDeviceSynchronize();
	end = getticks();
	printf("\tTime for left transpose: %lf \n", (double)(end - start) / (double)512000000.0);

	cudaDeviceSynchronize();

	start = getticks();
	multiplyMatricesParallelDoubleTranspose<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(leftRows, shared, rightCols, left, right, result);

	cudaDeviceSynchronize();
	end = getticks();
	printf("\tTime for both transpose: %lf \n\n", (double)(end - start) / (double)512000000.0);
}

int main(int argc, char** argv)
{
	srand(time(NULL)); // Set random seed for this execution
	if(argc < 4)
	{
		fprintf(stderr, "ERROR: Format is ./cacheTest [square matrix size] [non-zero for prefetch right first] [non-zero for testing serial]\n");
		return 1;
	}

	unsigned int matrixSize = atoi(argv[1]);
	if(matrixSize <= 0)
	{
		fprintf(stderr, "ERROR: matrix size must be >= 1\n");
		return 1;
	}

	int prefetchSetting = atoi(argv[2]);
	bool prefetchRightFirst = prefetchSetting != 0;

	int useSerial = atoi(argv[3]);
	bool testSerial = useSerial != 0;

	int device = 0;

	short** right;
	allocateMatrix(matrixSize, matrixSize, (void***)&right, sizeof(short));

	short** left;
	allocateMatrix(matrixSize, matrixSize, (void***)&left, sizeof(short));

	int** serialResult;
	if(testSerial)
		allocateMatrix(matrixSize, matrixSize, (void***)&serialResult, sizeof(int));

	int** parallelResult;
	allocateMatrix(matrixSize, matrixSize, (void***)&parallelResult, sizeof(int));

	generateMatrix(matrixSize, matrixSize, left, true);
	generateMatrix(matrixSize, matrixSize, right, true);

	parallelMultiplication(matrixSize, matrixSize, matrixSize, left, right, parallelResult, device, prefetchRightFirst);

	if(testSerial)
	{
		printf("Serial timings:\n");
		ticks start = getticks();
		multiplyMatricesSerial(matrixSize, matrixSize, matrixSize, left, right, serialResult);
		ticks end = getticks();
		printf("\tTime for normal: %lf \n", (double)(end - start) / (double)512000000.0);

		start = getticks();
		multiplyMatricesSerialRightTranspose(matrixSize, matrixSize, matrixSize, left, right, serialResult);
		end = getticks();
		printf("\tTime for right transpose: %lf \n", (double)(end - start) / (double)512000000.0);

		start = getticks();
		multiplyMatricesSerialLeftTranspose(matrixSize, matrixSize, matrixSize, left, right, serialResult);
		end = getticks();
		printf("\tTime for left transpose: %lf \n", (double)(end - start) / (double)512000000.0);

		start = getticks();
		multiplyMatricesSerialDoubleTranspose(matrixSize, matrixSize, matrixSize, left, right, serialResult);
		end = getticks();
		printf("\tTime for double transpose: %lf \n\n", (double)(end - start) / (double)512000000.0);
	}

	freeMatrix(matrixSize, (void**)left);
	freeMatrix(matrixSize, (void**)right);
	if(testSerial)
		freeMatrix(matrixSize, (void**)serialResult);
	freeMatrix(matrixSize, (void**)parallelResult);

	return 0;
}
