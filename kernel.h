#ifndef KERNEL_H
#define KERNEL_H

#include <stdbool.h>

void allocateMatrix(unsigned int numRows, unsigned int numCols, long*** matrix, bool setZero, int device);
void freeMatrix(unsigned int numRows, long** matrix);
void generateMatrix(unsigned int numRows, unsigned int numCols, long** matrix);
void printMatrix(unsigned int numRows, unsigned int numCols, long** matrix);

void parallelMultiplication(unsigned int leftRows,
                            unsigned int shared,
                            unsigned int rightCols,
                            long** left,
                            long** right,
                            long** result,
                            int device,
                            bool useSharedMem);

#endif