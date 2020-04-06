#include <algorithm>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  if (!n)
    return;

  constexpr long nthrd = 3; // the optimal value on my computer is 3

  constexpr long mask = 64 / sizeof(long) - 1;
  const long chunk = ((n + nthrd - 1) / nthrd + mask) & ~mask;
# pragma omp parallel num_threads(nthrd) default(none) shared(prefix_sum, A, n)
  {
    auto id = omp_get_thread_num();
    auto beg = chunk * id;
    auto end = std::min(n, beg + chunk);

    auto sum = beg ? A[beg - 1] : 0;
    prefix_sum[beg] = sum;
    for (long i = beg + 1; i < end; ++i)
      prefix_sum[i] = sum += A[i - 1];
  }

  for (long i = chunk; i < n; i += chunk) {
    auto off = prefix_sum[i - 1];
    for (long j = i; j < std::min(n, i + chunk); ++j)
      prefix_sum[j] += off;
  }
#if 0
  // This part is a textbook version of parallel scan (using contraction, O(n)
  // work, O(log n) span) with recurrence expanded. But it only achieves very
  // poor performance due to multi-threading overhead.
# pragma omp parallel default(none) shared(A, prefix_sum, n)
  {
#   pragma omp single
    prefix_sum[1] = A[0];
#   pragma omp for
    for (long i = 3; i < n; i += 2)
      prefix_sum[i] = A[i - 2] + A[i - 1];
    long h;
    for (h = 4; h <= n; h <<= 1)
#   pragma omp for
      for (long i = h - 1; i < n; i += h)
        prefix_sum[i] += prefix_sum[i - (h >> 1)];
    for (h >>= 1; h > 2; h >>= 1)
#   pragma omp for
      for (long i = h - 1; i < n - (h >> 1); i += h)
        prefix_sum[i + (h >> 1)] += prefix_sum[i];
#   pragma omp single
    prefix_sum[0] = 0;
#   pragma omp for
    for (long i = 1; i < n - 1; i += 2)
      prefix_sum[i + 1] = prefix_sum[i] + A[i];
  }
#endif
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = i;

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
