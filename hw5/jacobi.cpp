#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

using namespace std;

#include <mpi.h>

#define Strify_(s_) # s_
#define Strify(s_) Strify_(s_)

#ifdef NDEBUG
#define checkMpi(e_) (e_)
#else
#define checkMpi(e_) (implCheckMpi((e_), __LINE__, Strify(e_)))
void implCheckMpi(int res, long line, const char* expr) {
  if (res == MPI_SUCCESS)
    return;
  std::fprintf(stderr, "MPI Runtime Error: %d\n", res);
  std::fprintf(stderr, "  At line %ld: %s\n", line, expr);
  std::exit(EXIT_FAILURE);
}
#endif

constexpr size_t Alignment = 64;
using Num = double;
#define MpiNum MPI_DOUBLE

Num* allocNum(size_t N) {
  if (auto res = std::aligned_alloc(Alignment, N * sizeof(Num)))
    return (Num*) __builtin_assume_aligned(res, Alignment);
  std::fprintf(stderr, "Allocation of %zu bytes failed\n", N * sizeof(Num));
  checkMpi(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
  std::exit(EXIT_FAILURE);
}

constexpr Num sqr(Num x) { return x * x; }

bool checkArgs(int& N, int& lN, int &nMaxIter, int nArg, char* args[], int lId)
{
  if (nArg < 2 || nArg > 3) {
    if (!lId)
      std::fprintf(stderr, "Incorrect command line\n");
    return false;
  }
  if (1 != sscanf(args[1], "%d", &N)) {
    if (!lId)
      std::fprintf(stderr, "Invalid N: %s\n", args[1]);
    return false;
  }
  if (1 != sscanf(args[2], "%d", &nMaxIter)) {
    if (!lId)
      std::fprintf(stderr, "Invalid #MaxIter: %s\n", args[2]);
    return false;
  }
  int np;
  checkMpi(MPI_Comm_size(MPI_COMM_WORLD, &np));
  auto log = __builtin_ctz(np);
  if ((1 << log) != np || (log & 1)) {
    if (!lId)
      std::fprintf(stderr, "#Process=%d must be power of 4\n", np);
    return false;
  }
  auto sqrtnp = 1 << (log >> 1);
  if (N % sqrtnp) {
    if (!lId) {
      std::fprintf(stderr, "N=%d must be multiple of sqrt(#Process)=%d\n",
        N, sqrtnp);
    }
    return false;
  }
  lN = N / sqrtnp;
  return true;
}

Num calcRes(const Num* __restrict__ lu, int lN, Num g2) {
  auto lM = lN + 2;
  Num lres = 0;
  for (auto i = 1; i <= lN; ++i)
    for (auto j = 1; j <= lN; ++j) {
      lres += sqr(g2 * (4 * lu[j + i * lM] -
        lu[j + (i - 1) * lM] - lu[j - 1 + i * lM] -
        lu[j + (i + 1) * lM] - lu[j + 1 + i * lM]) - 1);
    }
  Num res;
  checkMpi(MPI_Allreduce(&lres, &res, 1, MpiNum, MPI_SUM, MPI_COMM_WORLD));
  return std::sqrt(res);
}

constexpr int TagSendL = 101;
constexpr int TagSendR = 102;
constexpr int TagSendD = 103;
constexpr int TagSendU = 104;
constexpr int TagRecvL = TagSendR;
constexpr int TagRecvR = TagSendL;
constexpr int TagRecvD = TagSendU;
constexpr int TagRecvU = TagSendD;

int main(int nArg, char* args[]) {
  checkMpi(MPI_Init(&nArg, &args));
  int lId, nProc, N, lN, nMaxIter;
  checkMpi(MPI_Comm_rank(MPI_COMM_WORLD, &lId));
  checkMpi(MPI_Comm_size(MPI_COMM_WORLD, &nProc));

  if (!checkArgs(N, lN, nMaxIter, nArg, args, lId) && !lId) {
    std::fprintf(stderr, "Usage: %s <N> <#MaxIter>\n", args[0]);
    checkMpi(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
    return EXIT_FAILURE;
  }

  auto K = N / lN;
  auto lx = lId % K;
  auto ly = lId / K;

  auto lM = lN + 2;
  auto lM2 = lM * lM;

  MPI_Datatype MpiCol;
  checkMpi(MPI_Type_vector(lN, 1, lM, MpiNum, &MpiCol));
  checkMpi(MPI_Type_commit(&MpiCol));

  checkMpi(MPI_Barrier(MPI_COMM_WORLD));
  auto tStart = MPI_Wtime();

  auto lu = allocNum(lM2);
  auto lv = allocNum(lM2);

  std::fill(lu, lu + lM2, 0);
  std::fill(lv, lv + lM2, 0);

  auto g2 = sqr(N + 1);
  auto h2 = 1 / g2;
  auto res = calcRes(lu, lN, g2);
  auto resInit = res;
  auto nIter = 1;

  if (!lId)
    std::printf("[Iter 0] Residual: %g\n", res);

  while (nIter <= nMaxIter && res * 1e6 > resInit) {
    // Jacobi Step
    for (auto i = 1; i <= lN; ++i)
      for (auto j = 1; j <= lN; ++j) {
        lv[j + i * lM] = .25 * (h2 +
          lu[j + (i - 1) * lM] + lu[j - 1 + i * lM] +
          lu[j + (i + 1) * lM] + lu[j + 1 + i * lM]);
      }
    // Communication
    //std::printf("[%d] Iter %d Comm L\n", lId, nIter);
    if (lx > 0) {
      checkMpi(MPI_Send(lv + 1 * lM + 1, 1, MpiCol, lId - 1, TagSendL,
        MPI_COMM_WORLD));
      checkMpi(MPI_Recv(lv + 1 * lM, 1, MpiCol, lId - 1, TagRecvL,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }
    //std::printf("[%d] Iter %d Comm R\n", lId, nIter);
    if (lx < K - 1) {
      checkMpi(MPI_Send(lv + 1 * lM + lN, 1, MpiCol, lId + 1, TagSendR,
        MPI_COMM_WORLD));
      checkMpi(MPI_Recv(lv + 1 * lM + lN + 1, 1, MpiCol, lId + 1, TagRecvR,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }
    //std::printf("[%d] Iter %d Comm D\n", lId, nIter);
    if (ly > 0) {
      checkMpi(MPI_Send(lv + lM + 1, lN, MpiNum, lId - K, TagSendD,
        MPI_COMM_WORLD));
      checkMpi(MPI_Recv(lv + 1, lN, MpiNum, lId - K, TagRecvD,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }
    //std::printf("[%d] Iter %d Comm U\n", lId, nIter);
    if (ly < K - 1) {
      checkMpi(MPI_Send(lv + lN * lM + 1, lN, MpiNum, lId + K, TagSendU,
        MPI_COMM_WORLD));
      checkMpi(MPI_Recv(lv + (lN + 1) * lM + 1, lN, MpiNum, lId + K, TagRecvU,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }
    //std::printf("[%d] Iter %d Comm End\n", lId, nIter);
    std::swap(lu, lv);
    if (!(nIter % 10)) {
      res = calcRes(lu, lN, g2);
      if (!lId)
        std::printf("[Iter %d] Residual: %e\n", nIter, res);
    }
    ++nIter;
  }

  res = calcRes(lu, lN, g2);

  std::free(lu);
  std::free(lv);

  checkMpi(MPI_Type_free(&MpiCol));

  auto tEnd = MPI_Wtime();
  auto time = tEnd - tStart;

  if (!lId) {
    std::printf("[Completed] Time: %f s\n", time);
    std::printf("[Completed] Residual: %e\n", res);
  }

  checkMpi(MPI_Finalize());
  return 0;
}
