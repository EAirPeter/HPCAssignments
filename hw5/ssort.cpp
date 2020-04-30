#define NDEBUG

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
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

using Num = int;
#define MpiNum MPI_INT

template<class T>
Num* alloc(size_t N) {
  if (auto res = std::malloc(N * sizeof(T)))
    return (Num*) res;
  std::fprintf(stderr, "Allocation of %zu bytes failed\n", N * sizeof(T));
  checkMpi(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
  std::exit(EXIT_FAILURE);
}

template<class T>
Num* realloc(Num* p, size_t N) {
  if (auto res = std::realloc(p, N * sizeof(T)))
    return (Num*) res;
  std::fprintf(stderr, "Allocation of %zu bytes failed\n", N * sizeof(T));
  checkMpi(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
  std::exit(EXIT_FAILURE);
}

bool checkArgs(int& N, int nArg, char* args[], int id) {
  if (nArg != 2) {
    if (!id)
      std::fprintf(stderr, "Incorrect command line\n");
    return false;
  }
  if (1 != sscanf(args[1], "%d", &N)) {
    if (!id)
      std::fprintf(stderr, "Invalid N: %s\n", args[1]);
    return false;
  }
  return true;
}

int main(int nArg, char* args[]) {
  checkMpi(MPI_Init(&nArg, &args));
  int id, P, N;
  checkMpi(MPI_Comm_rank(MPI_COMM_WORLD, &id));
  checkMpi(MPI_Comm_size(MPI_COMM_WORLD, &P));

  if (!checkArgs(N, nArg, args, id) && !id) {
    std::fprintf(stderr, "Usage: %s <N>\n", args[0]);
    checkMpi(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
    return EXIT_FAILURE;
  }

  // Workspace
  auto sdispl = alloc<int>(P);
  auto scount = alloc<int>(P);
  auto rdispl = alloc<int>(P);
  auto rcount = alloc<int>(P);
  auto spl = alloc<Num>(P - 1);
  auto res = id ? nullptr : alloc<Num>((P - 1) * P);

  // Input
  auto vec = alloc<Num>(N);
  std::mt19937_64 rand(std::random_device{}());
  for (auto i = 0; i < N; ++i)
    vec[i] = (int) rand();

  // Timer Starts
  checkMpi(MPI_Barrier(MPI_COMM_WORLD));
  auto tStart = MPI_Wtime();

  // Local Sort
  std::sort(vec, vec + N);

  // Local Sample
  for (auto i = 0, j = N / P; i < P - 1; ++i, j += N / P)
    spl[i] = vec[j];

  checkMpi(MPI_Gather(spl, P - 1, MpiNum, res, P - 1, MpiNum,
    0, MPI_COMM_WORLD));

  if (!id) {
    auto M = (P - 1) * P;
    // Splitters
    std::sort(res, res + M);
    for (auto i = 0, j = P - 1; i < P - 1; ++i, j += P - 1)
      spl[i] = res[j];
  }

  // Broadcast Splitters
  checkMpi(MPI_Bcast(spl, P - 1, MpiNum, 0, MPI_COMM_WORLD));

  // Bucket Assignment
  sdispl[0] = 0;
  for (auto i = 1; i < P; ++i)
    sdispl[i] = std::lower_bound(vec, vec + N, spl[i - 1]) - vec;
  for (auto i = 1; i < P; ++i)
    scount[i - 1] = sdispl[i] - sdispl[i - 1];
  scount[P - 1] = N - sdispl[P - 1];

  // Element Counts Communication
  checkMpi(MPI_Alltoall(scount, 1, MPI_INT, rcount, 1, MPI_INT,
    MPI_COMM_WORLD));

  auto M = 0;
  for (auto i = 0; i < P; ++i) {
    rdispl[i] = M;
    M += rcount[i];
  }

  // Preallocation
  res = realloc<Num>(res, M);

  // Elements Communication
  checkMpi(MPI_Alltoallv(vec, scount, sdispl, MpiNum,
    res, rcount, rdispl, MpiNum, MPI_COMM_WORLD));

  // Last Sort
  std::sort(res, res + M);

  // Timer Stops
  checkMpi(MPI_Barrier(MPI_COMM_WORLD));
  auto tEnd = MPI_Wtime();

  // Write to file
  static char fname[256];
  std::snprintf(fname, 256, "ssort-n%d-p%d-r%.2d.txt", N, P, id);
  auto fp = std::fopen(fname, "w");
  if (!fp) {
    std::fprintf(stderr, "Failed to open %s for writing\n", fname);
    checkMpi(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
    return EXIT_FAILURE;
  }

  std::fprintf(fp, "[Rank %d] Completed, time: %f\n", id, tEnd - tStart);
  std::fprintf(fp, "[Rank %d] #Number in this bucket: %d\n", id, M);
  for (auto i = 0; i < M; ++i)
    std::fprintf(fp, "%d\n", res[i]);

  std::fclose(fp);

  std::free(vec);

  std::free(sdispl);
  std::free(scount);
  std::free(rdispl);
  std::free(rcount);
  std::free(spl);
  std::free(res);

  checkMpi(MPI_Finalize());
  return 0;
}
