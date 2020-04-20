#include <algorithm>
#include <cmath>
#include <random>

#include <omp.h>

#include "utils.h"

using Num = float;

constexpr size_t alignment = 128;

Num* allocNum(size_t n) {
  if (auto res = aligned_alloc(alignment, n * sizeof(Num)))
    return (Num*) __builtin_assume_aligned(res, alignment);
  std::fprintf(stderr, "Allocation of %zu bytes failed", n * sizeof(Num));
  std::exit(EXIT_FAILURE);
}


void dotProdRef(Num* __restrict__ pz,
  const Num* __restrict__ x, const Num* __restrict__ y, long n)
{
  x = (const Num*) __builtin_assume_aligned(x, alignment);
  y = (const Num*) __builtin_assume_aligned(y, alignment);

  Num z = 0;
  for (auto i = 0l; i < n; ++i)
    z += x[i] * y[i];

  *pz = z;
}

void dotProdOmp(Num* __restrict__ pz,
  const Num* __restrict__ x, const Num* __restrict__ y, long n)
{
  x = (const Num*) __builtin_assume_aligned(x, alignment);
  y = (const Num*) __builtin_assume_aligned(y, alignment);

  Num z = 0;
# pragma omp parallel for reduction(+: z) schedule(static)
  for (auto i = 0l; i < n; ++i)
    z += x[i] * y[i];

  *pz = z;
}

__global__
void initKernel(Num* __restrict__ z) {
  *z = 0;
}

__global__
void dotProdGpu(Num* __restrict__ z,
  const Num* __restrict__ x, const Num* __restrict__ y, long n)
{
  extern __shared__ Num tmp[];
  auto i = blockIdx.x * blockDim.x + threadIdx.x;

  tmp[threadIdx.x] = i < n ? x[i] * y[i] : 0;

  __syncthreads();

  for (auto h = blockDim.x / 2; h; h >>= 1) {
    if (threadIdx.x < h)
      tmp[threadIdx.x] += tmp[threadIdx.x + h];
    __syncthreads();
  }

  if (!threadIdx.x)
    atomicAdd(z, tmp[0]);
}

void processData(const char* name, double time,
  Num z, Num zRef, long n, long nIter)
{
  auto error = fabs(z - zRef);
  auto bandwidth = nIter * n * 2 * sizeof(Num) * 1e-9 / time;
  printf("%12s %12.6f %18.6f %16e\n", name, time, bandwidth, error);
}

int main() {
  constexpr int blkSize = 1024;
  constexpr long nIter = 100;
  constexpr long n = 1l << 20;

  dim3 dimGrid((n + blkSize - 1) / blkSize);
  dim3 dimBlock(blkSize);

  cudaDeviceProp prop;
  checkCuda(cudaGetDeviceProperties(&prop, 0));
  checkCuda(cudaSetDevice(0));
  printf("Device: %s\n", prop.name);
  printf("Vector Dimension: %d\n", n);
  printf("dimGrid: %d %d %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
  printf("dimBlock: %d %d %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
  
  // Data generation
  auto x    = allocNum(n);
  auto y    = allocNum(n);

  std::mt19937_64 rand{std::random_device{}()};
  auto randpm1 = [&] {
    return std::uniform_real_distribution<Num>(-1, 1)(rand);
  };

  std::generate(x, x + n, randpm1);
  std::generate(y, y + n, randpm1);

  // GPU memory initialization
  size_t memSize = n * sizeof(Num);
  size_t tmpSize = blkSize * sizeof(Num);
  Num *d_x, *d_y, *d_z;
  checkCuda(cudaMalloc(&d_x, memSize));
  checkCuda(cudaMalloc(&d_y, memSize));
  checkCuda(cudaMalloc(&d_z, sizeof(Num)));

  checkCuda(cudaMemcpy(d_x, x, memSize, cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_y, y, memSize, cudaMemcpyHostToDevice));

  Timer t;
  
  printf("\n%12s %12s %18s %16s\n",
    "Routine", "Time (s)", "Bandwidth (GB/s)", "Error");

  // Results
  Num zRef, zOmp, zGpu;

  // Serial/reference
  dotProdRef(&zRef, x, y, n);
  t.tic();
  for (volatile auto iter = 0l; iter < nIter; ++iter)
    dotProdRef(&zRef, x, y, n);
  processData("dotProdRef", t.toc(), zRef, zRef, n, nIter);

  // OpenMP
  dotProdOmp(&zOmp, x, y, n);
  t.tic();
  for (volatile auto iter = 0l; iter < nIter; ++iter)
    dotProdOmp(&zOmp, x, y, n);
  processData("dotProdOmp", t.toc(), zOmp, zRef, n, nIter);

  // GPU/CUDA
  initKernel<<<1, 1>>>(d_z);
  dotProdGpu<<<dimGrid, dimBlock, tmpSize>>>(d_z, d_x, d_y, n);
  t.tic();
  for (volatile auto iter = 0l; iter < nIter; ++iter) {
    initKernel<<<1, 1>>>(d_z);
    dotProdGpu<<<dimGrid, dimBlock, tmpSize>>>(d_z, d_x, d_y, n);
  }
  cudaDeviceSynchronize();
  auto time = t.toc();
  checkCuda(cudaMemcpy(&zGpu, d_z, sizeof(Num), cudaMemcpyDeviceToHost));
  processData("dotProdGpu", time, zGpu, zRef, n, nIter);

  checkCuda(cudaFree(d_x));
  checkCuda(cudaFree(d_y));
  checkCuda(cudaFree(d_z));

  free(x);
  free(y);
  return 0;
}
