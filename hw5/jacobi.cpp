#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

#define Strify_(s_) # s_
#define Strify(s_) Strify_(s_)

#ifdef NDEBUG
#define checkMpi(e_) (e_)
#else
#define checkMpi(e_) (implCheckMpi((e_), __LINE__, Strify(e_)))
void implCheckMpi(int res, long line, const char* expr) {
  if (res == MPI_SUCCESS)
    return;
  fprintf(stderr, "MPI Runtime Error: %d\n", res);
  fprintf(stderr, "  At line %ld: %s\n", line, expr);
  exit(EXIT_FAILURE);
}
#endif

bool checkArgs(int& N, int& lN, int &nMaxIter, int nArg, char* args[]) {
  if (nArg < 2 || nArg > 4) {
    fprintf(stderr, "Incorrect command line\n");
    return false;
  }
  if (1 != sscanf(args[1], "%d", &N)) {
    fprintf(stderr, "Invalid N: %s\n", args[1]);
    return false;
  }
  if (1 != sscanf(args[2], "%d", &nMaxIter)) {
    fprintf(stderr, "Invalid #MaxIter: %s\n", args[2]);
    return false;
  }
  int np;
  checkMpi(MPI_Comm_size(MPI_COMM_WORLD, &np));
  auto log = __builtin_ctz(np);
  if ((1 << log) != np || (log & 1)) {
    fprintf(stderr, "#Process=%d must be power of 4\n", np);
    return false;
  }
  auto sqrtnp = 1 << (log >> 1);
  if (N % sqrtnp) {
    fprintf(stderr, "N=%d must be multiple of sqrt(#Process)=%d\n", N, sqrtnp);
    return false;
  }
  lN = N / sqrtnp;
  return true;
}

int main(int nArg, char* args[]) {
  checkMpi(MPI_Init(&nArg, &args));
  int lId, nProc, N, lN, nMaxIter;
  checkMpi(MPI_Comm_rank(MPI_COMM_WORLD, &lId));
  checkMpi(MPI_Comm_size(MPI_COMM_WORLD, &nProc));

  if (!lId && !checkArgs(N, lN, nMaxIter, nArg, args)) {
    fprintf(stderr, "Usage: %s <N> <#MaxIter>\n", args[0]);
    checkMpi(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
    return EXIT_FAILURE;
  }

  checkMpi(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD));
  checkMpi(MPI_Bcast(&lN, 1, MPI_INT, 0, MPI_COMM_WORLD));
  checkMpi(MPI_Bcast(&nMaxIter, 1, MPI_INT, 0, MPI_COMM_WORLD));

  checkMpi(MPI_Finalize());
  return 0;
}
