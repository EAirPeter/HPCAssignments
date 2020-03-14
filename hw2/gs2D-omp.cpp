#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

#include <omp.h>

#include "utils.h"

namespace {
  int N, N2;
  int nMaxIter;
  int nThread;
  double g2, h2;
}

bool checkArgs(int nArg, char* args[]) {
  if (nArg < 2 || nArg > 4)
    return false;

  N = std::atoi(args[1]);
  if (N < 1)
    return false;

  N2 = (N + 2) * (N + 2);

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

inline double sqr(double x) { return x * x; }

inline double& at(double* __restrict__ u, int i, int j)
{ return u[j + i * (N + 2)]; }

inline double at(const double* __restrict__ u, int i, int j)
{ return u[j + i * (N + 2)]; }

inline double computeResidual(const double* __restrict__ u) {
  // |r| = |Au-f|
  // r[i,j] = sum{A[i,j][x,y]*u[x,y]} - 1
  //        = g^2(4u[i,j]-u[i-1,j]-u[i,j-1]-u[i+1,j]-u[i,j+1]) - 1
  double res = 0;
  for (int i = 1; i <= N; ++i)
    for (int j = 1; j <= N; ++j) {
      res += sqr(g2 * (4 * at(u, i, j) -
        at(u, i - 1, j) - at(u, i, j - 1) -
        at(u, i + 1, j) - at(u, i, j + 1)) - 1);
    }
  return std::sqrt(res);
}

int main(int nArg, char* args[]) {
  if (!checkArgs(nArg, args)) {
    std::fprintf(stderr, "Usage: %s <N> [#Thread] [#MaxIter]\n", args[0]);
    return EXIT_FAILURE;
  }

  double* __restrict__ u = (double* __restrict__) __builtin_assume_aligned(
    aligned_malloc(sizeof(double) * (size_t) N2), MEMORY_ALIGNMENT);
  double* __restrict__ v = (double* __restrict__) __builtin_assume_aligned(
    aligned_malloc(sizeof(double) * (size_t) N2), MEMORY_ALIGNMENT);
  if (!u || !v) {
    std::fprintf(stderr, "Failed to allocate memory for u and tmp_u\n");
    return EXIT_FAILURE;
  }

  std::fill(u, u + N2, 0.);
  std::fill(v, v + N2, 0.);

  g2 = sqr(N + 1);
  h2 = 1. / g2;

  std::printf("         N: %d\n", N);
  std::printf("   #Thread: %d\n", nThread);

  Timer T;
  T.tic();

  double resInit = computeResidual(u);

  int nIter = 0;

  while (nIter < nMaxIter && computeResidual(u) * 1e6 > resInit) {
    // v[i,j] = (h^2*f[i,j]+u[i-1,j]+u[i,j-1]+u[i+1,j]+u[i,j+1])/4
    //        = .25(h^2+u[i-1,j]+u[i,j-1]+u[i+1,j]+u[i,j+1])
# ifdef _OPENMP
#   pragma omp parallel for num_threads(nThread)
# endif
    for (int i = 1; i <= N; ++i)
      for (int j = 2 - (i & 1); j <= N; j += 2) {
        at(v, i, j) = .25 * (h2 +
          at(u, i - 1, j) + at(u, i, j - 1) +
          at(u, i + 1, j) + at(u, i, j + 1));
      }
# ifdef _OPENMP
#   pragma omp parallel for num_threads(nThread)
# endif
    for (int i = 1; i <= N; ++i)
      for (int j = 1 + (i & 1); j <= N; j += 2) {
        at(v, i, j) = .25 * (h2 +
          at(v, i - 1, j) + at(v, i, j - 1) +
          at(v, i + 1, j) + at(v, i, j + 1));
      }
    std::swap(u, v);
    ++nIter;
  }

  double t = T.toc();
  std::printf("#Iteration: %d\n", nIter);
  std::printf("      Time: %.9f s\n", t);

  aligned_free(u);
  aligned_free(v);

  return 0;
}
