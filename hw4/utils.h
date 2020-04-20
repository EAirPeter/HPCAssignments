#pragma once

#include <chrono>

#include <cstdio>
#include <cstdlib>

#define Strify_(s_) # s_
#define Strify(s_) Strify_(s_)

class Timer {
public:
  void tic() {
    using namespace std::chrono;
    tStart = high_resolution_clock::now();
  }

  double toc() {
    using namespace std::chrono;
    auto tEnd = high_resolution_clock::now();
    return duration_cast<nanoseconds>(tEnd - tStart).count() * 1e-9;
  }
private:
  std::chrono::time_point<std::chrono::high_resolution_clock> tStart;
};

#ifdef NDEBUG

#define checkCuda(e_) (e_)

#else

#define checkCuda(e_) (implCheckCuda((e_), Strify(e_)))

inline void implCheckCuda(cudaError_t res, const char* expr) {
  if (res == cudaSuccess)
    return;
  std::fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(res));
  std::fprintf(stderr, "  Expression: %s\n", expr);
  std::exit(EXIT_FAILURE);
}

#endif
