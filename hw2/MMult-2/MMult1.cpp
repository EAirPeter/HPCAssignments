// g++ -fopenmp -O3 -march=native MMult1.cpp && ./a.out

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

#define BLOCK_SIZE 40

// Note: matrices are stored in column major order; i.e. the array elements in
// the (m x n) matrix C are stored in the sequence: {C_00, C_10, ..., C_m0,
// C_01, C_11, ..., C_m1, C_02, ..., C_0n, C_1n, ..., C_mn}
void MMult0(long m, long n, long k, double *a, double *b, double *c) {
  for (long j = 0; j < n; j++) {
    for (long p = 0; p < k; p++) {
      for (long i = 0; i < m; i++) {
        double A_ip = a[i+p*m];
        double B_pj = b[p+j*k];
        double C_ij = c[i+j*m];
        C_ij = C_ij + A_ip * B_pj;
        c[i+j*m] = C_ij;
      }
    }
  }
}

void MMult1(
  long m, long n, long k,
  double* __restrict__ a, // Make our greatest effort to tell the compiler that
  double* __restrict__ b, // a, b, c do not overlap.
  double* __restrict__ c)
{
  // Make our greatest effort to tell the compiler that m, n, k are multiples
  // of BLOCK_SIZE.
  if (m % BLOCK_SIZE) __builtin_unreachable();
  if (n % BLOCK_SIZE) __builtin_unreachable();
  if (k % BLOCK_SIZE) __builtin_unreachable();
  // Make our greatest effort to tell the compiler that a, b, c are aligned to
  // 64-byte boundary, the typical size of a cache line, which also satisfies
  // the alignment requirements of SSE/AVX/AVX-512 instructions.
  a = (double* __restrict__) __builtin_assume_aligned(a, MEMORY_ALIGNMENT);
  b = (double* __restrict__) __builtin_assume_aligned(b, MEMORY_ALIGNMENT);
  c = (double* __restrict__) __builtin_assume_aligned(c, MEMORY_ALIGNMENT);
  // Hopefully, the auto-vectorization would work after writing such a long
  // prologue.
  for (long y = 0; y < n; y += BLOCK_SIZE) {
    for (long q = 0; q < k; q += BLOCK_SIZE) {
      for (long x = 0; x < m; x += BLOCK_SIZE) {
        for (long j = y; j < y + BLOCK_SIZE; ++j) {
          for (long p = q; p < q + BLOCK_SIZE; ++p) {
            for (long i = x; i < x + BLOCK_SIZE; ++i) {
              double A_ip = a[i + p * m];
              double B_pj = b[p + j * k];
              double C_ij = c[i + j * m];
              C_ij = C_ij + A_ip * B_pj;
              c[i + j * m] = C_ij;
            }
          }
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  const long PFIRST = BLOCK_SIZE;
  const long PLAST = 2000;
  const long PINC = std::max(50/BLOCK_SIZE,1) * BLOCK_SIZE; // multiple of BLOCK_SIZE

  printf(" Dimension       Time    Gflop/s       GB/s        Error\n");
  for (long p = PFIRST; p < PLAST; p += PINC) {
    long m = p, n = p, k = p;
    long NREPEATS = 1e9/(m*n*k)+1;
    double* a = (double*) aligned_malloc(m * k * sizeof(double)); // m x k
    double* b = (double*) aligned_malloc(k * n * sizeof(double)); // k x n
    double* c = (double*) aligned_malloc(m * n * sizeof(double)); // m x n
    double* c_ref = (double*) aligned_malloc(m * n * sizeof(double)); // m x n

    // Initialize matrices
    for (long i = 0; i < m*k; i++) a[i] = drand48();
    for (long i = 0; i < k*n; i++) b[i] = drand48();
    for (long i = 0; i < m*n; i++) c_ref[i] = 0;
    for (long i = 0; i < m*n; i++) c[i] = 0;

    for (long rep = 0; rep < NREPEATS; rep++) { // Compute reference solution
      MMult0(m, n, k, a, b, c_ref);
    }

    Timer t;
    t.tic();
    for (long rep = 0; rep < NREPEATS; rep++) {
      MMult1(m, n, k, a, b, c);
    }
    double time = t.toc();
    double flops = NREPEATS * m * n * k * 2 / 1e9 / time;
    double bandwidth = NREPEATS * m * n * k * 4 * sizeof(double) / 1e9 / time;
    printf("%10d %10f %10f %10f", p, time, flops, bandwidth);

    double max_err = 0;
    for (long i = 0; i < m*n; i++) max_err = std::max(max_err, fabs(c[i] - c_ref[i]));
    printf(" %10e\n", max_err);

    aligned_free(a);
    aligned_free(b);
    aligned_free(c);
  }

  return 0;
}

//  Processor: Intel Core i7-7700HQ
//  #Cores: 4 cores with hyper-threading
//  Cache Line: 64 Bytes
//  L1 Instruction Cache: 32 KiB per core; 8-way set associative
//  L1 Data Cache: 32 KiB per core; 8-way set associative
//  L2 Cache: 256 KiB per core; 4-way set associative
//  L3 Cache: 6MiB shared; 12-way set associative
//  Memory: 16 GiB
//  Operating System: Windows Subsystem for Linux (Debian)
//  Kernel: Linux 4.4.0-18362-Microsoft
//  Compiler: GCC 8.3.0 (Debian 8.3.0-6)
//
// * Using MMult0 as a reference, implement MMult1 and try to rearrange loops to
// maximize performance. Measure performance for different loop arrangements and
// try to reason why you get the best performance for a particular order?
//
//  The original arrangement of loops in MMult0 implementation has the optimal
//  performance. All 6 possible loop arrangements are tested and the results
//  are in placed in MMult-1/ directory where Mjpi.txt corresponds to the
//  original MMult0 implementation.
//
//  The reason is why the order j-p-i is optimal follows these observations:
//    - Read of a[i+p*m] is sequential as i increases one by one;
//    - Read/write of c[i+j*m] is also sequential as i increases one by one;
//  By letting i be the inner most loop ensures best locality in the above
//  operations.
//
//  Similarly, read of b[p+j*k] also has best locality when p increases one by
//  one. Hence the order j-p-i should be better than p-j-i, which is endorsed
//  by the test result. Hence, the j-p-i order should have the best
//  performance.
//
//  When preparing to write the next part, I noticed that, by giving some
//  stronger hint to the compiler would also help the performance as recorded
//  in Mhint.txt (has slightly better record than Mjpi.txt for large
//  dimensions). The final code for this part is copied to MMult-1/ directory,
//  which used the original j-p-i order with some additional hints to the
//  compiler.
//
// * You will notice that the performance degrades for larger matrix sizes that
// do not fit in the cache. To improve the performance for larger matrices,
// implement a one level blocking scheme by using BLOCK_SIZE macro as the block
// size. By partitioning big matrices into smaller blocks that fit in the cache
// and multiplying these blocks together at a time, we can reduce the number of
// accesses to main memory. This resolves the main memory bandwidth bottleneck
// for large matrices and improves performance.
//
// NOTE: You can assume that the matrix dimensions are multiples of BLOCK_SIZE.
//
//
// * Experiment with different values for BLOCK_SIZE (use multiples of 4) and
// measure performance.  What is the optimal value for BLOCK_SIZE?
//
//  I tested for BLOCK_SIZE = 4, 8, ..., 64 and the results are saved in
//  MMult-2/ directory. For two block sizes bs1 and bs2, it could tell which
//  one is better by comparing the time needed for dimensions that are
//  multiples of LCM(bs1, bs2). After carefully comparing among these files,
//  that the optimal value for BLOCK_SIZE is 40 on my machine.
//
//  Also, the final code of this part is copied to MMult-2/ directory.
//
// * Now parallelize your matrix-matrix multiplication code using OpenMP.
//
//
// * What percentage of the peak FLOP-rate do you achieve with your code?
//
//
// NOTE: Compile your code using the flag -march=native. This tells the compiler
// to generate the best output using the instruction set supported by your CPU
// architecture. Also, try using either of -O2 or -O3 optimization level flags.
// Be aware that -O2 can sometimes generate better output than using -O3 for
// programmer optimized code.
