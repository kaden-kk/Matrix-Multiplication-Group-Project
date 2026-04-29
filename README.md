# Matrix Multiplication Group Project

## Setup

1. **Extract the tarball**
```bash
   tar -xvf matrix_project.tar.gz
   cd Matrix-Multiplication-Group-Project
```

2. **Compile**
```bash
   make all
```

   To remove compiled binaries and object files:
```bash
   make clean
```

3. **Request a GPU node on AiMOS**
```bash
   salloc -N 4 --partition=el8-rpi --gres=gpu:4 -t 10
```

4. **Load Modules**
```bash
   module load xl_r spectrum-mpi cuda/11.2
```

## Usage

### Matrix Multiplication (MPI + CUDA)

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

#### Quick Test
To verify the program runs correctly on a small matrix with serial checking:
```bash
mpirun -np 2 ./matrix 1024 1024 1024 1 1 1
```

### Cache Prefetch Test (CUDA only)

`cacheTest` is a standalone CUDA benchmark that tests all four transpose configurations
(normal, right transpose, left transpose, double transpose) and compares parallel vs serial timings.

```bash
./cacheTest <leftRows> <shared> <rightCols> <prefetchRightFirst> <testSerial>
```

| Argument | Description |
|----------|-------------|
| `leftRows` | Number of rows in the left matrix (A) |
| `shared` | Shared dimension — cols of A / rows of B |
| `rightCols` | Number of columns in the right matrix (B) |
| `prefetchRightFirst` | Prefetch right matrix to GPU before left (`1` = yes, `0` = no) |
| `testSerial` | Also run and time serial CPU versions (`1` = yes, `0` = no) |

#### Quick Test
```bash
./cacheTest 4096 4096 4096 0 0
```

## Experiments

### Strong Scaling (fixed 16384×16384 matrix)
```bash
# Optimized (shared memory + transpose)
mpirun -np 2  ./matrix 16384 16384 16384 1 1 0
mpirun -np 4  ./matrix 16384 16384 16384 1 1 0
mpirun -np 8  ./matrix 16384 16384 16384 1 1 0
mpirun -np 16 ./matrix 16384 16384 16384 1 1 0
mpirun -np 32 ./matrix 16384 16384 16384 1 1 0

# Shared memory only
mpirun -np 2  ./matrix 16384 16384 16384 1 0 0
mpirun -np 4  ./matrix 16384 16384 16384 1 0 0
mpirun -np 8  ./matrix 16384 16384 16384 1 0 0
mpirun -np 16 ./matrix 16384 16384 16384 1 0 0
mpirun -np 32 ./matrix 16384 16384 16384 1 0 0

# Transpose only
mpirun -np 2  ./matrix 16384 16384 16384 0 1 0
mpirun -np 4  ./matrix 16384 16384 16384 0 1 0
mpirun -np 8  ./matrix 16384 16384 16384 0 1 0
mpirun -np 16 ./matrix 16384 16384 16384 0 1 0
mpirun -np 32 ./matrix 16384 16384 16384 0 1 0

# Baseline (no optimizations)
mpirun -np 2  ./matrix 16384 16384 16384 0 0 0
mpirun -np 4  ./matrix 16384 16384 16384 0 0 0
mpirun -np 8  ./matrix 16384 16384 16384 0 0 0
mpirun -np 16 ./matrix 16384 16384 16384 0 0 0
mpirun -np 32 ./matrix 16384 16384 16384 0 0 0
```

### Weak Scaling (32768 rows per rank)
```bash
# Optimized (shared memory + transpose)
mpirun -np 2  ./matrix 65536   1024 1024 1 1 0
mpirun -np 4  ./matrix 131072  1024 1024 1 1 0
mpirun -np 8  ./matrix 262144  1024 1024 1 1 0
mpirun -np 16 ./matrix 524288  1024 1024 1 1 0
mpirun -np 32 ./matrix 1048576 1024 1024 1 1 0

# Shared memory only
mpirun -np 2  ./matrix 65536   1024 1024 1 0 0
mpirun -np 4  ./matrix 131072  1024 1024 1 0 0
mpirun -np 8  ./matrix 262144  1024 1024 1 0 0
mpirun -np 16 ./matrix 524288  1024 1024 1 0 0
mpirun -np 32 ./matrix 1048576 1024 1024 1 0 0

# Transpose only
mpirun -np 2  ./matrix 65536   1024 1024 0 1 0
mpirun -np 4  ./matrix 131072  1024 1024 0 1 0
mpirun -np 8  ./matrix 262144  1024 1024 0 1 0
mpirun -np 16 ./matrix 524288  1024 1024 0 1 0
mpirun -np 32 ./matrix 1048576 1024 1024 0 1 0

# Baseline (no optimizations)
mpirun -np 2  ./matrix 65536   1024 1024 0 0 0
mpirun -np 4  ./matrix 131072  1024 1024 0 0 0
mpirun -np 8  ./matrix 262144  1024 1024 0 0 0
mpirun -np 16 ./matrix 524288  1024 1024 0 0 0
mpirun -np 32 ./matrix 1048576 1024 1024 0 0 0
```