#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <mpi.h>

#include "kernel.h"

int main(int argc, char** argv)
{
    MPI_Init(&argc,&argv);

    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    if(argc < 5)
    {
        if(rank==0)
        {
            printf("Usage: ./matrix [leftRows] [shared] [rightCols] [useSharedMem] [optional print]\n");
        }

        MPI_Finalize();
        return 1;
    }

    unsigned int leftRows  = atoi(argv[1]);
    unsigned int shared    = atoi(argv[2]);
    unsigned int rightCols = atoi(argv[3]);

    int memInput = atoi(argv[4]);
    bool useSharedMem = memInput != 0;

    int device = 0;

    // Row distribution
    unsigned int rowsPerRank = leftRows / size;
    unsigned int localRows   = rowsPerRank;

    if(rank == size-1)
        localRows = leftRows - rowsPerRank*(size-1);

    // Allocate matrices
    long **leftLocal;
    long **right;
    long **resultLocal;

    allocateMatrix(localRows,shared,&leftLocal,false,device);
    allocateMatrix(shared,rightCols,&right,false,device);
    allocateMatrix(localRows,rightCols,&resultLocal,true,device);

    long **leftFull = NULL;

    if(rank==0)
    {
        allocateMatrix(leftRows,shared,&leftFull,false,device);
        generateMatrix(leftRows,shared,leftFull);
        generateMatrix(shared,rightCols,right);
    }

    // broadcast B

    for(unsigned int i=0;i<shared;i++)
    {
        MPI_Bcast(right[i],rightCols,MPI_LONG,0,MPI_COMM_WORLD);
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
                    memcpy(leftLocal[i],leftFull[i],shared*sizeof(long));
                }
            }
            else
            {
                for(unsigned int i=0;i<sendRows;i++)
                {
                    MPI_Send(leftFull[r*rowsPerRank+i], shared, MPI_LONG, r, 0, MPI_COMM_WORLD);
                }
            }
        }
    }
    else
    {
        for(unsigned int i=0;i<localRows;i++)
        {
            MPI_Recv(leftLocal[i], shared, MPI_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime();

    // GPU computation
    parallelMultiplication(localRows, shared, rightCols, leftLocal, right, resultLocal, device, useSharedMem);

    MPI_Barrier(MPI_COMM_WORLD);

    double end = MPI_Wtime();

    // gather results
    long **finalResult=NULL;

    if(rank==0)
    {
        allocateMatrix(leftRows,rightCols,&finalResult,true,device);
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
                    memcpy(finalResult[i], resultLocal[i], rightCols*sizeof(long));
                }
            }
            else
            {
                for(unsigned int i=0;i<recvRows;i++)
                {
                    MPI_Recv(finalResult[r*rowsPerRank+i], rightCols, MPI_LONG, r, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }
    }
    else
    {
        for(unsigned int i=0;i<localRows;i++)
        {
            MPI_Send(resultLocal[i], rightCols,MPI_LONG, 0, 1, MPI_COMM_WORLD);
        }
    }

    // Print timing
    if(rank==0)
    {
        printf("Total runtime: %f seconds\n", end - start);
    }

    if(argc > 5 && rank==0)
    {
        printf("Result matrix:\n");
        printMatrix(leftRows,rightCols,finalResult);
    }

    // Free memory
    freeMatrix(localRows,leftLocal);
    freeMatrix(shared,right);
    freeMatrix(localRows,resultLocal);

    if(rank==0)
    {
        freeMatrix(leftRows,leftFull);
        freeMatrix(leftRows,finalResult);
    }

    MPI_Finalize();

    return 0;
}
