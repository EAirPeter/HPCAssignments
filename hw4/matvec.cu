#define NDEBUG

#include <algorithm>
#include <cmath>
#include <random>

#include <omp.h>

#include "utils.h"

using Num = double;

constexpr size_t alignment = 128;
constexpr size_t alignmask = alignment / sizeof(Num) - 1;

Num* allocNum(size_t n) {
  if (auto res = aligned_alloc(alignment, n * sizeof(Num)))
    return (Num*) __builtin_assume_aligned(res, alignment);
  std::fprintf(stderr, "Allocation of %zu bytes failed", n * sizeof(Num));
  std::exit(EXIT_FAILURE);
}


void matVecProdRef(Num* __restrict__ v,
  const Num* __restrict__ A, const Num* __restrict__ u, long n)
{
  A = (const Num*) __builtin_assume_aligned(A, alignment);
  u = (const Num*) __builtin_assume_aligned(u, alignment);
  v = (Num*) __builtin_assume_aligned(v, alignment);

  for (auto i = 0l; i < n; ++i) {
    Num t = 0;
    for (auto j = 0l; j < n; ++j)
      t += A[i * n + j] * u[j];
    v[i] = t;
  }
}

void matVecProdOmp(Num* __restrict__ v,
  const Num* __restrict__ A, const Num* __restrict__ u, long n)
{
  A = (const Num*) __builtin_assume_aligned(A, alignment);
  u = (const Num*) __builtin_assume_aligned(u, alignment);
  v = (Num*) __builtin_assume_aligned(v, alignment);

# pragma omp parallel for schedule(static)
  for (auto i = 0l; i < n; ++i) {
    Num t = 0;
    for (auto j = 0l; j < n; ++j)
      t += A[i * n + j] * u[j];
    v[i] = t;
  }
}

template<long xMaxGrdSize, long xBlkSize>
__global__
void dotProdKernel(Num* __restrict__ v,
  const Num* __restrict__ A, const Num* __restrict__ u, long n)
{
  extern __shared__ Num tmp[];

  auto tid = threadIdx.x;
  auto i = threadIdx.y + blockIdx.y * blockDim.y;
  auto j = tid + blockIdx.x * xBlkSize * 2;
  auto stride = gridDim.x * xBlkSize * 2;

  Num res = 0;

  while (j < n) {
    res += A[i * n + j] * u[j];
    if (j + xBlkSize < n)
      res += A[i * n + j + xBlkSize] * u[j + xBlkSize];
    j += stride;
  }

  tmp[tid] = res;
  __syncthreads();

  auto h = xBlkSize / 2;

  while (h > 32) {
    if (tid < h)
      tmp[tid] = res += tmp[tid + h];
    __syncthreads();
    h >>= 1;
  }

  if (tid < 32) {
    while (h) {
      tmp[tid] = res += tmp[tid + h];
      __syncthreads();
      h >>= 1;
    }
  }

  if (!tid)
    v[i * gridDim.x + blockIdx.x] = res;
}

__global__
void reductionKernel(Num* __restrict__ v, const Num* __restrict__ u, long n) {
  auto i = threadIdx.x + blockIdx.x * blockDim.x;
  Num res = 0;
  for (auto j = 0; j < n; ++j)
    res += u[i * n + j];
  v[i] = res;
}

template<long xMaxGrdSize, long xBlkSize, long yBlkSize>
void matVecProdGpu(Num* __restrict__ v, Num* __restrict__ d_v,
  const Num* __restrict__ d_A, const Num* __restrict__ d_u, long n)
{
  v = (Num*) __builtin_assume_aligned(v, alignment);
  auto tmpSize = (xBlkSize < 64 ? 64 : xBlkSize) * sizeof(Num);
  auto xGrdSize = std::min(xMaxGrdSize, (n + xBlkSize * 2 - 1) / (xBlkSize * 2));
  if (xGrdSize < 2)
    dotProdKernel<xMaxGrdSize, xBlkSize><<<dim3(xGrdSize, n), xBlkSize, tmpSize>>>(d_v, d_A, d_u, n);
  else {
    auto d_v_ = d_v + ((n + alignmask) & ~alignmask);
    dotProdKernel<xMaxGrdSize, xBlkSize><<<dim3(xGrdSize, n), xBlkSize, tmpSize>>>(d_v_, d_A, d_u, n);
    auto yGrdSize = (n + yBlkSize - 1) / yBlkSize;
    reductionKernel<<<yGrdSize, yBlkSize>>>(d_v, d_v_, xGrdSize);
  }
  checkCuda(cudaMemcpy(v, d_v, n * sizeof(Num), cudaMemcpyDeviceToHost));
}

void processData(const char* name, double time,
  const Num* v, const Num* vRef, long n, long nIter)
{
  Num error = 0;
  for (auto i = 0; i < n; ++i)
    error = std::max(error, std::fabs(v[i] - vRef[i]));
  auto bandwidth = nIter * (n * n * 2 + n) * sizeof(Num) * 1e-9 / time;
  printf("%12s %12.6f %18.6f %16e\n", name, time, bandwidth, error);
}

int main() {
  constexpr long xMaxGrdSize = 128;
  constexpr long xBlkSize = 128;
  constexpr long yBlkSize = 1024;
  constexpr long nIter = 20;
  constexpr long n = 6000;

  cudaDeviceProp prop;
  checkCuda(cudaGetDeviceProperties(&prop, 0));
  checkCuda(cudaSetDevice(0));
  printf("Device: %s\n", prop.name);
  printf("Matrix/Vector Dimension: %d\n", n);
  
  // Data generation
  auto A = allocNum(n * n);
  auto u = allocNum(n);
  auto vRef = allocNum(n);
  auto vOmp = allocNum(n);
  auto vGpu = allocNum(n);

  std::mt19937_64 rand{std::random_device{}()};
  auto randpm1 = [&] {
    return std::uniform_real_distribution<Num>(-1, 1)(rand);
  };

  std::generate(A, A + n * n, randpm1);
  std::generate(u, u + n, randpm1);

  // GPU memory initialization
  Num *d_A, *d_u, *d_v;
  checkCuda(cudaMalloc(&d_A, n * n * sizeof(Num)));
  checkCuda(cudaMalloc(&d_u, n * sizeof(Num)));
  checkCuda(cudaMalloc(&d_v, (n * (xMaxGrdSize + 1) + alignmask) * sizeof(Num)));

  checkCuda(cudaMemcpy(d_A, A, n * n * sizeof(Num), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_u, u, n * sizeof(Num), cudaMemcpyHostToDevice));

  Timer t;
  
  printf("\n%12s %12s %18s %16s\n",
    "Routine", "Time (s)", "Bandwidth (GB/s)", "Error");

  // Serial/reference
  matVecProdRef(vRef, A, u, n);
  t.tic();
  for (volatile auto iter = 0l; iter < nIter; ++iter)
    matVecProdRef(vRef, A, u, n);
  processData("matVecProdRef", t.toc(), vRef, vRef, n, nIter);

  // OpenMP
  matVecProdOmp(vOmp, A, u, n);
  t.tic();
  for (volatile auto iter = 0l; iter < nIter; ++iter)
    matVecProdOmp(vOmp, A, u, n);
  processData("matVecProdOmp", t.toc(), vOmp, vRef, n, nIter);

  // GPU/CUDA
  matVecProdGpu<xMaxGrdSize, xBlkSize, yBlkSize>(vGpu, d_v, d_A, d_u, n);
  t.tic();
  for (volatile auto iter = 0l; iter < nIter; ++iter)
    matVecProdGpu<xMaxGrdSize, xBlkSize, yBlkSize>(vGpu, d_v, d_A, d_u, n);
  auto time = t.toc();
  processData("matVecProdGpu", time, vGpu, vRef, n, nIter);

  checkCuda(cudaFree(d_A));
  checkCuda(cudaFree(d_u));
  checkCuda(cudaFree(d_v));

  free(A);
  free(u);
  free(vRef);
  free(vOmp);
  free(vGpu);
  return 0;
}
