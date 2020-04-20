#define NDEBUG

#include <algorithm>
#include <cmath>
#include <random>

#include <omp.h>

#include "utils.h"

using Num = double;

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

template<long blkSize>
__global__
void dotProdKernel(Num* __restrict__ z,
  const Num* __restrict__ x, const Num* __restrict__ y, long n)
{
  extern __shared__ Num tmp[];

  auto tid = threadIdx.x;
  auto i = tid + blockIdx.x * blkSize * 2;
  auto stride = gridDim.x * blkSize * 2;

  Num res = 0;

  while (i < n) {
    res += x[i] * y[i];
    if (i + blkSize < n)
      res += x[i + blkSize] * y[i + blkSize];
    i += stride;
  }

  tmp[tid] = res;
  __syncthreads();

  if (blkSize >= 1024) {
    if (tid < 512)
      tmp[tid] = res += tmp[tid + 512];
    __syncthreads();
  }

  if (blkSize >= 512) {
    if (tid < 256)
      tmp[tid] = res += tmp[tid + 256];
    __syncthreads();
  }

  if (blkSize >= 256) {
    if (tid < 128)
      tmp[tid] = res += tmp[tid + 128];
    __syncthreads();
  }

  if (blkSize >= 128) {
    if (tid < 64)
      tmp[tid] = res += tmp[tid + 64];
    __syncthreads();
  }

  if (tid < 32) {
    if (blkSize >= 64) {
      tmp[tid] = res += tmp[tid + 32];
      __syncthreads();
    }
    if (blkSize >= 32) {
      tmp[tid] = res += tmp[tid + 16];
      __syncthreads();
    }
    if (blkSize >= 16) {
      tmp[tid] = res += tmp[tid + 8];
      __syncthreads();
    }
    if (blkSize >= 8) {
      tmp[tid] = res += tmp[tid + 4];
      __syncthreads();
    }
    if (blkSize >= 4) {
      tmp[tid] = res += tmp[tid + 2];
      __syncthreads();
    }
    if (blkSize >= 2) {
      tmp[tid] = res += tmp[tid + 1];
      __syncthreads();
    }
  }

  if (!tid)
    z[blockIdx.x] = res;
}

template<long grdSize, long blkSize>
void dotProdGpu(Num* z, Num* __restrict__ d_z,
  const Num* __restrict__ d_x, const Num* __restrict__ d_y, long n)
{
  static Num tmp[grdSize] alignas(128);
  auto tmpSize = (blkSize < 64 ? 64 : blkSize) * sizeof(Num);
  auto nb = std::min(grdSize, (n + blkSize * 2 - 1) / (blkSize * 2));
  dotProdKernel<blkSize><<<nb, blkSize, tmpSize>>>(d_z, d_x, d_y, n);
  //while (nb > 1) {
  //  n = nb;
  //  nb = std::min(grdSize, (n + blkSize * 2 - 1) / (blkSize * 2));
  //  reductionKernel<blkSize><<<nb, blkSize, tmpSize>>>(d_z + n, d_z, n);
  //  d_z += n;
  //}
  //checkCuda(cudaMemcpy(z, d_z, sizeof(Num), cudaMemcpyDeviceToHost));
  checkCuda(cudaMemcpy(tmp, d_z, nb * sizeof(Num), cudaMemcpyDeviceToHost));
  Num res = 0;
  for (auto i = 0l; i < nb; ++i)
    res += tmp[i];
  *z = res;
}

void processData(const char* name, double time,
  Num z, Num zRef, long n, long nIter)
{
  auto error = fabs(z - zRef);
  auto bandwidth = nIter * n * 2 * sizeof(Num) * 1e-9 / time;
  printf("%12s %12.6f %18.6f %16e\n", name, time, bandwidth, error);
}

int main() {
  constexpr int grdSize = 128;
  constexpr int blkSize = 256;
  constexpr long nIter = 100;
  constexpr long n = 1l << 20;

  cudaDeviceProp prop;
  checkCuda(cudaGetDeviceProperties(&prop, 0));
  checkCuda(cudaSetDevice(0));
  printf("Device: %s\n", prop.name);
  printf("Vector Dimension: %d\n", n);
  
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
  Num *d_x, *d_y, *d_z;
  checkCuda(cudaMalloc(&d_x, memSize));
  checkCuda(cudaMalloc(&d_y, memSize));
  checkCuda(cudaMalloc(&d_z, memSize));

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
  dotProdGpu<grdSize, blkSize>(&zGpu, d_z, d_x, d_y, n);
  t.tic();
  for (volatile auto iter = 0l; iter < nIter; ++iter)
    dotProdGpu<grdSize, blkSize>(&zGpu, d_z, d_x, d_y, n);
  auto time = t.toc();
  processData("dotProdGpu", time, zGpu, zRef, n, nIter);

  checkCuda(cudaFree(d_x));
  checkCuda(cudaFree(d_y));
  checkCuda(cudaFree(d_z));

  free(x);
  free(y);
  return 0;
}
