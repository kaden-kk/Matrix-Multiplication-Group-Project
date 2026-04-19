#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <mpi.h>
#include <cuda_runtime.h>

#include "kernel.h"

int main(int argc, char** argv)
{
    MPI_Init(&argc,&argv);

    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device = rank % deviceCount;
    cudaSetDevice(device);

    if(argc < 7)
    {
        if(rank==0)
        {
            printf("Usage: ./matrix [leftRows] [shared] [rightCols] [useSharedMem] [useTranspose] [checkSerial] [optional print]\n");
        }

        MPI_Finalize();
        return 1;
    }

    unsigned int leftRows  = atoi(argv[1]);
    unsigned int shared    = atoi(argv[2]);
    unsigned int rightCols = atoi(argv[3]);

    int memInput = atoi(argv[4]);
    bool useSharedMem = memInput != 0;

    int transpose = atoi(argv[5]);
    bool useTranspose = transpose != 0;

    int check = atoi(argv[6]);
    bool checkSerial = check != 0;

    // Row distribution
    unsigned int rowsPerRank = leftRows / size;
    unsigned int localRows   = rowsPerRank;

    if(rank == size-1)
        localRows = leftRows - rowsPerRank*(size-1);

    // Allocate matrices
    short **leftLocal;
    short **right;
    int **resultLocal;

    if(allocateMatrix(localRows,shared,(void***)&leftLocal, sizeof(short)) == 1)
    {
        fprintf(stderr, "CudaMalloc failed. Use smaller matrices\n");
        return 1;
    }
    if(allocateMatrix(shared,rightCols,(void***)&right, sizeof(short)) == 1)
    {
        fprintf(stderr, "CudaMalloc failed. Use smaller matrices\n");
        return 1;
    }
    if(allocateMatrix(localRows,rightCols,(void***)&resultLocal, sizeof(int)) == 1)
    {
        fprintf(stderr, "CudaMalloc failed. Use smaller matrices\n");
        return 1;
    }

    short **leftFull = NULL;

    if(rank==0)
    {
        if(allocateMatrix(leftRows,shared,(void***)&leftFull, sizeof(short)) == 1)
        {
            fprintf(stderr, "CudaMalloc failed. Use smaller matrices\n");
        }
        generateMatrix(leftRows,shared,leftFull,!checkSerial);
        generateMatrix(shared,rightCols,right,!checkSerial);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    short **rightTranspose = NULL;

    if(useTranspose && rank == 0)
    {
        double transposeStart = MPI_Wtime();
        if(allocateMatrix(rightCols,shared,(void***)&rightTranspose, sizeof(short)) == 1)
        {
            fprintf(stderr, "CudaMalloc failed. Use smaller matrices\n");
            return 1;
        }
        transposeMatrix(shared, rightCols, right, rightTranspose,device);
        freeMatrix(shared,(void**)right);
        right = rightTranspose;
        double transposeEnd = MPI_Wtime();
        printf("Transpose runtime: %f seconds\n", transposeEnd - transposeStart);
    }

    // broadcast B

    for(unsigned int i=0;i<shared;i++)
    {
        if(useTranspose)
            MPI_Bcast(right[i],shared,MPI_SHORT,0,MPI_COMM_WORLD);
        else
            MPI_Bcast(right[i],rightCols,MPI_SHORT,0,MPI_COMM_WORLD);
    }

    // scatter rows of A 

    if(rank==0)
    {
        for(int r=0;r<size;r++)
        {
            int sendRows = rowsPerRank;

            if(r==size-1)
                sendRows = leftRows - rowsPerRank*(size-1);

            if(r==0)
            {
                for(unsigned int i=0;i<sendRows;i++)
                {
                    memcpy(leftLocal[i],leftFull[i],shared*sizeof(short));
                }
            }
            else
            {
                for(unsigned int i=0;i<sendRows;i++)
                {
                    MPI_Send(leftFull[r*rowsPerRank+i], shared, MPI_SHORT, r, 0, MPI_COMM_WORLD);
                }
            }
        }
    }
    else
    {
        for(unsigned int i=0;i<localRows;i++)
        {
            MPI_Recv(leftLocal[i], shared, MPI_SHORT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // GPU computation
    double parallelStart = MPI_Wtime();
    parallelMultiplication(localRows, shared, rightCols, leftLocal, right, resultLocal, device, useSharedMem, useTranspose);
    double parallelEnd = MPI_Wtime();
    if(rank == 0)
        printf("Parallel runtime: %f seconds\n", parallelEnd - parallelStart);

    MPI_Barrier(MPI_COMM_WORLD);

    // gather results
    int **finalResult=NULL;

    if(rank==0)
    {
        if(allocateMatrix(leftRows,rightCols,(void***)&finalResult, sizeof(int)) == 1)
        {
            fprintf(stderr, "CudaMalloc failed. Use smaller matrices\n");
            return 1;
        }
    }

    if(rank==0)
    {
        for(int r=0;r<size;r++)
        {
            int recvRows = rowsPerRank;

            if(r==size-1)
                recvRows = leftRows - rowsPerRank*(size-1);

            if(r==0)
            {
                for(unsigned int i=0;i<recvRows;i++)
                {
                    memcpy(finalResult[i], resultLocal[i], rightCols*sizeof(int));
                }
            }
            else
            {
                for(unsigned int i=0;i<recvRows;i++)
                {
                    MPI_Recv(finalResult[r*rowsPerRank+i], rightCols, MPI_INT, r, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }
    }
    else
    {
        for(unsigned int i=0;i<localRows;i++)
        {
            MPI_Send(resultLocal[i], rightCols,MPI_INT, 0, 1, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    // Print timing
    if(rank==0)
    {
        printf("Total runtime: %f seconds\n", end - start);
    }

    if(argc > 7 && rank==0)
    {
        printf("Left matrix:\n");
        printMatrixShort(leftRows,shared,leftFull);

        if(useTranspose)
        {
            printf("Right matrix: (transpose)\n");
            printMatrixShort(rightCols,shared,right);
        }
        else
        {
            printf("Right matrix:\n");
            printMatrixShort(shared,rightCols,right);
        }

        printf("Result matrix:\n");
        printMatrixInt(leftRows,rightCols,finalResult);
    }

    bool failed = false;
    int** serial = NULL;
    if(rank==0)
    {
        if(checkSerial)
        {
            if(allocateMatrix(leftRows,rightCols,(void***)&serial, sizeof(int)) == 1)
            {
                fprintf(stderr, "CudaMalloc failed. Use smaller matrices\n");
                return 1;
            }

            double serialStart = MPI_Wtime();
            multiplyMatricesSerial(leftRows, shared, rightCols, leftFull, right, serial, useTranspose);
            double serialEnd = MPI_Wtime();
            printf("Serial runtime: %f seconds\n", serialEnd - serialStart);
        }

        if(argc > 7 && checkSerial)
        {
            printf("Serial matrix:\n");
            printMatrixInt(leftRows,rightCols,serial);
        }

        // Remove this if statement if you want to check everytime (even when serial not computed)
        if(checkSerial)
        {
            printf("Checking results...\n");
            failed = checkResults(leftRows,rightCols,finalResult,serial,shared,device) != 0;
        }
    }
    if(rank == 0)
    {
        if(failed)
            printf("Multiplcation incorrect\n");
        else
            printf("Success\n");
    }

    // Free memory
    if(rank == 0)
    {
        printf("Freeing matrices...\n");
    }
    freeMatrix(localRows,(void**)leftLocal);
    if(useTranspose)
        freeMatrix(rightCols,(void**)right);
    else
        freeMatrix(shared,(void**)right);
    freeMatrix(localRows,(void**)resultLocal);

    if(rank==0)
    {
        freeMatrix(leftRows,(void**)leftFull);
        freeMatrix(leftRows,(void**)finalResult);
        freeMatrix(leftRows,(void**)serial);
    }

    MPI_Finalize();

    return 0;
}
