#ifndef KERNEL_H
#define KERNEL_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

void allocateMatrix(unsigned int numRows, unsigned int numCols, void*** matrix, int dataSize);
void generateMatrix(unsigned int numRows, unsigned int numCols, short** matrix);
void freeMatrix(unsigned int numRows, void** matrix);
void printMatrixShort(unsigned int numRows, unsigned int numCols, short** matrix);
void printMatrixInt(unsigned int numRows, unsigned int numCols, int** matrix);
void transposeMatrix(unsigned int rows, unsigned int cols, short** original, short** result, int device);

void multiplyMatricesSerial(unsigned int leftRows, 
                            unsigned int shared, 
                            unsigned int rightCols, 
                            short** left, 
                            short** right, 
                            int** result,
                            bool useTranspose);

void parallelMultiplication(unsigned int leftRows,
                            unsigned int shared,
                            unsigned int rightCols,
                            short** left,
                            short** right,
                            int** result,
                            int device,
                            bool useSharedMem,
                            bool transpose);

int checkResults(unsigned int numRows, unsigned int numCols, int** parallel, int** serial, int correct, int device);

#ifdef __cplusplus
}
#endif

#endif