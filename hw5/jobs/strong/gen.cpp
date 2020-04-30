#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

struct Version {
  char prefix[4];
  char exec[28];
};

constexpr Version vers[]{{"bl", "../jacobi"}, {"nb", "../jacobi-nb"}};

constexpr int nIter = 100;
constexpr int N = 25600;
//constexpr int nCpuPerTask = 1;
constexpr int nTaskPerNode = 16;
constexpr int nNode = 16;

int main() {
  auto nMaxTask = nTaskPerNode * nNode;
  ofstream all("all.sh");
  for (auto& ver : vers) {
    for (auto np = 1, k = 1; np <= nMaxTask; np <<= 2, k <<= 1) {
      auto nodes = max(1, np / nTaskPerNode);
      auto tasksPerNode = min(np, nTaskPerNode);
      auto ln = N / k;
      ostringstream oss;
      oss << ver.prefix << "-n" << N << "-ln" << ln;
      oss << "-it" << nIter << "-np" << np;
      auto name = oss.str();
      all << "sbatch " << name << ".sh\n";
      ofstream job(name + ".sh");
      job << "#!/bin/bash\n";
      job << "#\n";
      job << "#SBATCH --job-name=" << name << '\n';
      job << "#SBATCH --nodes=" << nodes << '\n';
      job << "#SBATCH --tasks-per-node=" << tasksPerNode << '\n';
      job << "#SBATCH --cpus-per-task=1\n";
      job << "#SBATCH --time=1:00:00\n";
      job << "#SBATCH --mem=1GB\n";
      //job << "#SBATCH --mail-type=END\n";
      //job << "#SBATCH --mail-user=zl2972@nyu.edu\n";
      job << "#SBATCH --output=" << name << ".out\n";
      job << "\n";
      job << "module purge\n";
      job << "module load openmpi/gnu/4.0.2\n";
      job << "\n";
      job << "mpiexec " << ver.exec << ' ' << N << ' ' << nIter << '\n';
    }
  }
  return 0;
}
