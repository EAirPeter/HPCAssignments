#define NDEBUG

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

bool checkArgs(int& n, int& nMaxIter, int& nThread, int nArg, char* args[]) {
  if (nArg < 2 || nArg > 4)
    return false;

  n = std::atoi(args[1]);
  if (n < 1)
    return false;

  nThread = nArg >= 3 ? std::atoi(args[2]) : 0;

  if (!nThread) {
#   pragma omp parallel
    {
#     pragma omp master
      nThread = omp_get_num_threads();
    }
  }

  nMaxIter = nArg >= 4 ? std::atoi(args[3]) : INT_MAX - 1;
  if (nMaxIter <= 0)
    return false;

  return true;
}

using Num = double;

constexpr size_t alignment = 128;

constexpr int blkSize = 32;
constexpr int redBlkSize = 128;
constexpr int redGrdSize = 128;

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

__host__ __device__
constexpr Num sqr(Num x) { return x * x; }

inline Num cpuCalcRes(const Num* __restrict__ u, int n) {
  // |r| = |Au-f|
  // r[i,j] = sum{A[i,j][x,y]*u[x,y]} - 1
  //        = g^2(4u[i,j]-u[i-1,j]-u[i,j-1]-u[i+1,j]-u[i,j+1]) - 1
  auto m = n + 2;
  auto g2 = sqr(n + 1);
  Num res = 0;
# pragma omp parallel for reduction(+: res) schedule(static)
  for (int i = 1; i <= n; ++i) {
    Num tmp = 0;
    for (int j = 1; j <= n; ++j) {
      tmp += sqr(g2 * (4 * u[j + i * m] -
        u[j + (i - 1) * m] - u[j - 1 + i * m] -
        u[j + (i + 1) * m] - u[j + 1 + i * m]) - 1);
    }
    res += tmp;
  }
  return std::sqrt(res);
}

int cpuJacobi(Num*& __restrict__ u,
  Num*& __restrict__ v, int n, int nMaxIter)
{
  u = (Num*) __builtin_assume_aligned(u, alignment);
  v = (Num*) __builtin_assume_aligned(v, alignment);

  auto m = n + 2;
  auto h2 = Num{1} / sqr(n + 1);
  auto res = cpuCalcRes(u, n);
  auto resInit = res;
  auto nIter = 1;

  while (nIter < nMaxIter && res * 1e6 > resInit) {
    // v[i,j] = (h^2*f[i,j]+u[i-1,j]+u[i,j-1]+u[i+1,j]+u[i,j+1])/4
    //        = .25(h^2+u[i-1,j]+u[i,j-1]+u[i+1,j]+u[i,j+1])
#   pragma omp parallel for schedule(static)
    for (int i = 1; i <= n; ++i)
      for (int j = 1; j <= n; ++j) {
        v[j + i * m] = .25 * (h2 +
          u[j + (i - 1) * m] + u[j - 1 + i * m] +
          u[j + (i + 1) * m] + u[j + 1 + i * m]);
      }
    std::swap(u, v);
    res = cpuCalcRes(u, n);
    ++nIter;
  }
  // One more iteration after residual check, for consistency with GPU version
# pragma omp parallel for schedule(static)
  for (int i = 1; i <= n; ++i)
    for (int j = 1; j <= n; ++j) {
      v[j + i * m] = .25 * (h2 +
        u[j + (i - 1) * m] + u[j - 1 + i * m] +
        u[j + (i + 1) * m] + u[j + 1 + i * m]);
    }
  std::swap(u, v);
  return nIter;
}

__global__
void initKernel(Num* u, int m2) {
  auto i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < m2)
    u[i] = 0;
}

__global__
void reductionKernel(Num* __restrict__ y, const Num* __restrict__ x, int n) {
  __shared__ Num shm[redBlkSize];

  auto tid = threadIdx.x;
  auto i = tid + blockIdx.x * redBlkSize * 2;
  auto stride = gridDim.x * redBlkSize * 2;

  Num res = 0;

  while (i < n) {
    res += x[i];
    if (i + redBlkSize < n)
      res += x[i + redBlkSize];
    i += stride;
  }

  shm[tid] = res;
  __syncthreads();

  auto h = redBlkSize / 2;

  while (h > 32) {
    if (tid < h)
      shm[tid] = res += shm[tid + h];
    __syncthreads();
    h >>= 1;
  }

  if (tid < 32) {
    while (h) {
      shm[tid] = res += shm[tid + h];
      __syncthreads();
      h >>= 1;
    }
  }

  if (!tid)
    y[blockIdx.x] = res;
}

Num gpuReduce(Num* __restrict__ d_y, const Num* __restrict__ d_x, int n) {
  Num tmp[redGrdSize] alignas(alignment);
  auto nBlk = std::min(redGrdSize,
    (n + redBlkSize * 2 - 1) / (redBlkSize * 2));
  reductionKernel<<<nBlk, redBlkSize>>>(d_y, d_x, n);
  checkCuda(cudaMemcpy(tmp, d_y, nBlk * sizeof(Num), cudaMemcpyDeviceToHost));
  Num res = 0;
  for (auto i = 0l; i < nBlk; ++i)
    res += tmp[i];
  return res;
}

inline Num gpuCalcRes(Num* __restrict__ d_res,
  const Num* __restrict__ d_tmp, int n2)
{
  auto res = gpuReduce(d_res, d_tmp, n2);
  return std::sqrt(res);
}

__global__
void jacobiKernel(Num* __restrict__ tmp, Num* __restrict__ u,
  int m, Num g2, Num h2)
{
  __shared__ Num shm[blkSize + 2][blkSize + 2];
  auto xOff = blockIdx.x * (blkSize - 2);
  auto yOff = blockIdx.y * (blkSize - 2);

  shm[threadIdx.y][threadIdx.x] = 
    threadIdx.x + xOff < m && threadIdx.y + yOff < m ?
    u[(threadIdx.y + yOff) * m + (threadIdx.x + xOff)] : 0;
  __syncthreads();

  auto f = .25 * (h2 + shm[threadIdx.y][threadIdx.x + 1] +
    shm[threadIdx.y + 2][threadIdx.x + 1] + shm[threadIdx.y + 1][threadIdx.x] +
    shm[threadIdx.y + 1][threadIdx.x + 2]);

  auto r = sqr(g2 * (4 * shm[threadIdx.y + 1][threadIdx.x + 1] -
    shm[threadIdx.y][threadIdx.x + 1] - shm[threadIdx.y + 2][threadIdx.x + 1] -
    shm[threadIdx.y + 1][threadIdx.x] - shm[threadIdx.y + 1][threadIdx.x + 2])
    - 1);

  if (threadIdx.x + 2 < blkSize && threadIdx.y + 2 < blkSize) {
    if (threadIdx.x + xOff + 2 < m && threadIdx.y + yOff + 2 < m) {
      u[(threadIdx.y + yOff + 1) * m + (threadIdx.x + xOff + 1)] = f;
      tmp[(threadIdx.y + yOff) * (m - 2) + (threadIdx.x + xOff)] = r;
    }
  }
}

int gpuJacobi(Num* __restrict__ u, Num* __restrict__ d_u,
  Num* __restrict__ d_tmp, Num* __restrict__ d_res, int n, int nMaxIter)
{
  auto m = n + 2;
  auto n2 = n * n;
  auto m2 = m * m;
  auto g2 = sqr(n + 1);
  auto h2 = Num{1} / sqr(n + 1);
  auto grdSize = (n + blkSize - 3) / (blkSize - 2);
  auto blockDim = dim3(blkSize, blkSize);
  auto gridDim = dim3(grdSize, grdSize);

  jacobiKernel<<<gridDim, blockDim>>>(d_tmp, d_u, m, g2, h2);
  auto res = gpuCalcRes(d_res, d_tmp, n * n);
  auto resInit = res;
  auto nIter = 1;

  while (nIter < nMaxIter && res * 1e6 > resInit) {
    jacobiKernel<<<gridDim, blockDim>>>(d_tmp, d_u, m, g2, h2);
    res = gpuCalcRes(d_res, d_tmp, n2);
    ++nIter;
  }
  checkCuda(cudaMemcpy(u, d_u, m2 * sizeof(Num), cudaMemcpyDeviceToHost));
  return nIter;
}

int main(int nArg, char* args[]) {
  int n, nMaxIter, nThread;
  if (!checkArgs(n, nMaxIter, nThread, nArg, args)) {
    std::fprintf(stderr, "Usage: %s <N> [#Thread] [#MaxIter]\n", args[0]);
    return EXIT_FAILURE;
  }

  std::printf("             N: %d\n", n);
  std::printf("CPU    #Thread: %d\n", nThread);

  cudaDeviceProp prop;
  auto dev = chooseDevice();
  checkCuda(cudaSetDevice(dev));
  checkCuda(cudaGetDeviceProperties(&prop, dev));
  std::printf("CUDA Device[%d]: %s\n", dev, prop.name);

  omp_set_num_threads(nThread);

  auto m = n + 2;
  auto m2 = m * m;

  // CPU Initialization
  auto uCpu = allocNum(m2);
  auto vCpu = allocNum(m2);
  auto uGpu = allocNum(m2);

  std::fill(uCpu, uCpu + m2, 0);
  std::fill(vCpu, vCpu + m2, 0);

  // GPU Initialization
  Num *d_u, *d_tmp, *d_res;
  checkCuda(cudaMalloc(&d_u, m2 * sizeof(Num)));
  checkCuda(cudaMalloc(&d_tmp, n * n * sizeof(Num)));
  checkCuda(cudaMalloc(&d_res, redGrdSize * sizeof(Num)));

  initKernel<<<(m2 + 1023) & ~1023, 1024>>>(d_u, m2);
  cudaDeviceSynchronize();

  std::printf("Init  Residual: %e\n", cpuCalcRes(uCpu, n));

  Timer t;
  t.tic();
  auto nIterCpu = cpuJacobi(uCpu, vCpu, n, nMaxIter);
  auto tCpu = t.toc();
  auto resCpu = cpuCalcRes(uCpu, n);

  std::printf("CPU #Iteration: %d\n", nIterCpu);
  std::printf("CPU       Time: %.6f s\n", tCpu);
  std::printf("CPU   Residual: %e\n", resCpu);
  
  t.tic();
  auto nIterGpu = gpuJacobi(uGpu, d_u, d_tmp, d_res, n, nMaxIter);
  auto tGpu = t.toc();
  auto resGpu = cpuCalcRes(uGpu, n);

  std::printf("GPU #Iteration: %d\n", nIterGpu);
  std::printf("GPU       Time: %.6f s\n", tGpu);
  std::printf("GPU   Residual: %e\n", resGpu);

  Num error = 0;
  for (auto i = 0; i < m2; ++i)
    error = std::max(error, std::fabs(uGpu[i] - uCpu[i]));
  std::printf("CPU-GPU  Error: %e\n", error);

  checkCuda(cudaFree(d_u));
  checkCuda(cudaFree(d_tmp));
  checkCuda(cudaFree(d_res));

  std::free(uCpu);
  std::free(vCpu);
  std::free(uGpu);

  return 0;
}
