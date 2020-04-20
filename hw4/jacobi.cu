//#define NDEBUG

#include <algorithm>
#include <cmath>
#include <random>

#include <cuda_runtime.h>
#include <omp.h>

#include "utils.h"

int chooseDevice() {
  int nDev;
  checkCuda(cudaGetDeviceCount(&nDev));
  auto sel = -1;
  size_t freeMax = 0;
  for (auto i = 0; i < nDev; ++i) {
    size_t free, total;
    checkCuda(cudaSetDevice(i));
    if (cudaMemGetInfo(&free, &total) != cudaSuccess)
      continue;
    if (free > freeMax) {
      sel = i;
      freeMax = free;
    }
  }
  if (sel < 0) {
    std::fprintf(stderr, "No CUDA device available");
    std::exit(EXIT_FAILURE);
  }
  return sel;
}

using Num = double;

constexpr size_t alignment = 128;

template<class T>
T* align(T* p) {
  return (T*) ((((uintptr_t) p) + alignment - 1) & ~(alignment - 1));
}

Num* allocNum(size_t n) {
  if (auto res = aligned_alloc(alignment, n * sizeof(Num)))
    return (Num*) __builtin_assume_aligned(res, alignment);
  std::fprintf(stderr, "Allocation of %zu bytes failed", n * sizeof(Num));
  std::exit(EXIT_FAILURE);
}

bool checkArgs(int& n, int& nMaxIter, int& nThread, int nArg, char* args[]) {
  if (nArg < 2 || nArg > 4)
    return false;

  n = std::atoi(args[1]);
  if (n < 1)
    return false;

  nThread = nArg >= 3 ? std::atoi(args[2]) : 0;

#ifdef _OPENMP
  if (!nThread) {
#   pragma omp parallel
    {
#     pragma omp master
      nThread = omp_get_num_threads();
    }
  }
#else
  nThread = 1;
#endif

  nMaxIter = nArg >= 4 ? std::atoi(args[3]) : INT_MAX - 1;
  if (nMaxIter <= 0)
    return false;

  return true;
}

__host__ __device__
constexpr Num sqr(Num x) { return x * x; }

inline Num computeResidual(const Num* __restrict__ u, int n) {
  // |r| = |Au-f|
  // r[i,j] = sum{A[i,j][x,y]*u[x,y]} - 1
  //        = g^2(4u[i,j]-u[i-1,j]-u[i,j-1]-u[i+1,j]-u[i,j+1]) - 1
  auto n2 = n + 2;
  auto g2 = sqr(n + 1);
  Num res = 0;
  for (int i = 1; i <= n; ++i)
    for (int j = 1; j <= n; ++j) {
      res += sqr(g2 * (4 * u[j + i * n2] -
        u[j + (i - 1) * n2] - u[j - 1 + i * n2] -
        u[j + (i + 1) * n2] - u[j + 1 + i * n2]) - 1);
    }
  return std::sqrt(res);
}

int jacobiCpu(Num*& __restrict__ u,
  Num*& __restrict__ v, int n, int nMaxIter)
{
  u = (Num*) __builtin_assume_aligned(u, alignment);
  v = (Num*) __builtin_assume_aligned(v, alignment);

  auto n2 = n + 2;
  auto h2 = Num{1} / sqr(n + 1);
  auto nIter = 0;
  auto res = computeResidual(u, n);
  auto resInit = res;

  while (nIter < nMaxIter && res * 1e6 > resInit) {
    // v[i,j] = (h^2*f[i,j]+u[i-1,j]+u[i,j-1]+u[i+1,j]+u[i,j+1])/4
    //        = .25(h^2+u[i-1,j]+u[i,j-1]+u[i+1,j]+u[i,j+1])
# ifdef _OPENMP
#   pragma omp parallel for schedule(static)
# endif
    for (int i = 1; i <= n; ++i)
      for (int j = 1; j <= n; ++j) {
        v[j + i * n2] = .25 * (h2 +
          u[j + (i - 1) * n2] + u[j - 1 + i * n2] +
          u[j + (i + 1) * n2] + u[j + 1 + i * n2]);
      }
    std::swap(u, v);
    res = computeResidual(u, n);
    ++nIter;
  }
  return nIter;
}

__global__
void initKernel(Num* u, int n) {
  auto n2 = (n + 2) * (n + 2);
  for (auto i = 0; i < n2; ++i)
    u[i] = 0;
}

__global__
void residualKernel(Num* pres, Num* __restrict__ u, int n) {
  auto n2 = n + 2;
  auto g2 = sqr(n + 1);
  Num res = 0;
  for (int i = 1; i <= n; ++i)
    for (int j = 1; j <= n; ++j) {
      res += sqr(g2 * (4 * u[j + i * n2] -
        u[j + (i - 1) * n2] - u[j - 1 + i * n2] -
        u[j + (i + 1) * n2] - u[j + 1 + i * n2]) - 1);
    }
  *pres = std::sqrt(res);
}

__global__
void jacobiKernel(Num* __restrict__ u, Num* __restrict__ v, int n) {
  auto i = threadIdx.x + blockIdx.x * blockDim.x + 1;
  auto j = threadIdx.y + blockIdx.y * blockDim.y + 1;
  if (1 <= i && i <= n && 1 <= j && j <= n) {
    auto h2 = Num{1} / sqr(n + 1);
    auto n2 = n + 2;
    v[j + i * n2] = .25 * (h2 +
      u[j + (i - 1) * n2] + u[j - 1 + i * n2] +
      u[j + (i + 1) * n2] + u[j + 1 + i * n2]);
  }
}

int jacobiGpu(Num* __restrict__ u, Num* __restrict__ d_u,
  Num* __restrict__ d_v, Num* __restrict__ d_res, int n, int nMaxIter)
{
  auto n2 = (n + 2) * (n + 2);
  auto blkSize = std::min(n, 32);
  auto grdSize = (n + blkSize - 1) / blkSize;
  auto blockDim = dim3(blkSize, blkSize);
  auto gridDim = dim3(grdSize, grdSize);
  auto nIter = 0;
  Num res;
  residualKernel<<<1, 1>>>(d_res, d_u, n);
  checkCuda(cudaMemcpy(&res, d_res, sizeof(Num), cudaMemcpyDeviceToHost));
  auto resInit = res;

  while (nIter < nMaxIter && res * 1e6 > resInit) {
    jacobiKernel<<<gridDim, blockDim>>>(d_u, d_v, n);
    residualKernel<<<1, 1>>>(d_res, d_v, n);
    checkCuda(cudaMemcpy(&res, d_res, sizeof(Num), cudaMemcpyDeviceToHost));
    std::swap(d_u, d_v);
    ++nIter;
  }
  checkCuda(cudaMemcpy(u, d_u, n2 * sizeof(Num), cudaMemcpyDeviceToHost));
  return nIter;
}

int main(int nArg, char* args[]) {
  int n, nMaxIter, nThread;
  if (!checkArgs(n, nMaxIter, nThread, nArg, args)) {
    std::fprintf(stderr, "Usage: %s <N> [#Thread] [#MaxIter]\n", args[0]);
    return EXIT_FAILURE;
  }

  cudaDeviceProp prop;
  auto dev = chooseDevice();
  checkCuda(cudaSetDevice(dev));
  checkCuda(cudaGetDeviceProperties(&prop, dev));
  std::printf("     Device[%d]: %s\n", dev, prop.name);

  omp_set_num_threads(nThread);

  auto n2 = (n + 2) * (n + 2);

  // CPU Initialization
  auto uCpu = allocNum(n2);
  auto vCpu = allocNum(n2);
  auto uGpu = allocNum(n2);

  std::fill(uCpu, uCpu + n2, 0);
  std::fill(vCpu, vCpu + n2, 0);

  // GPU Initialization
  Num *d_u, *d_v, *d_res;
  checkCuda(cudaMalloc(&d_u, n2 * sizeof(Num)));
  checkCuda(cudaMalloc(&d_v, n2 * sizeof(Num)));
  checkCuda(cudaMalloc(&d_res, sizeof(Num)));

  initKernel<<<1, 1>>>(d_u, n);
  initKernel<<<1, 1>>>(d_v, n);
  cudaDeviceSynchronize();

  std::printf("             N: %d\n", n);
  std::printf("CPU    #Thread: %d\n", nThread);

  Timer t;
  t.tic();
  auto nIterCpu = jacobiCpu(uCpu, vCpu, n, nMaxIter);
  auto tCpu = t.toc();
  auto resCpu = computeResidual(uCpu, n);

  std::printf("CPU #Iteration: %d\n", nIterCpu);
  std::printf("CPU       Time: %.6f s\n", tCpu);
  std::printf("CPU   Residual: %e\n", resCpu);
  
  t.tic();
  auto nIterGpu = jacobiGpu(uGpu, d_u, d_v, d_res, n, nMaxIter);
  auto tGpu = t.toc();
  auto resGpu = computeResidual(uGpu, n);

  std::printf("GPU #Iteration: %d\n", nIterGpu);
  std::printf("GPU       Time: %.6f s\n", tGpu);
  std::printf("GPU   Residual: %e\n", resGpu);

  Num error = 0;
  for (auto i = 0; i < n2; ++i)
    error = std::max(error, std::fabs(uGpu[i] - uCpu[i]));
  std::printf("         Error: %e\n", error);

  checkCuda(cudaFree(d_u));
  checkCuda(cudaFree(d_v));
  checkCuda(cudaFree(d_res));

  std::free(uCpu);
  std::free(vCpu);
  std::free(uGpu);

  return 0;
}
