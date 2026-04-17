# Matrix Multiplication Group Project

## Setup

1. **Clone the repository**
```bash
   git clone https://github.com/kaden-kk/Matrix-Multiplication-Group-Project.git
   cd Matrix-Multiplication-Group-Project
```

2. **Copy files to the supercomputer**

3. **Compile**

Build the project:
```bash
make matrix
```

To remove compiled binaries and object files:
```bash
make clean
```

## Usage

```bash
mpirun -np <numProcesses> ./matrix <leftRows> <shared> <rightCols> <useSharedMem> <useTranspose> <checkSerial> [print]
```

| Argument | Description |
|----------|-------------|
| `numProcesses` | Number of MPI ranks (should not exceed available GPUs) |
| `leftRows` | Number of rows in the left matrix (A) |
| `shared` | Shared dimension — cols of A / rows of B |
| `rightCols` | Number of columns in the right matrix (B) |
| `useSharedMem` | Use GPU shared memory (`1` = yes, `0` = no) |
| `useTranspose` | Transpose B before multiplication (`1` = yes, `0` = no) |
| `checkSerial` | Verify result against CPU serial multiplication (`1` = yes, `0` = no) |
| `print` *(optional)* | Any extra argument triggers matrix printing |

### Example

```bash
mpirun -np 4 ./matrix 1024 1024 1024 1 1 0
```
