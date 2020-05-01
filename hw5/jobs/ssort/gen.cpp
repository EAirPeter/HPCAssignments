#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

struct Version {
  char prefix[8];
  char exec[24];
};

constexpr Version vers[]{{"ssort", "../../ssort"}};

//constexpr int nCpuPerTask = 1;
constexpr int nTaskPerNode = 14;
constexpr int nNode = 12;

int main() {
  ofstream all("all.sh");
  for (auto& ver : vers) {
    for (auto n = 10000; n <= 1000000; n *= 10) {
      auto p = nTaskPerNode * nNode;
      ostringstream oss;
      oss << ver.prefix << "-n" << n << "-p" << p;
      auto name = oss.str();
      all << "sbatch " << name << ".sh\n";
      ofstream job(name + ".sh");
      job << "#!/bin/bash\n";
      job << "#\n";
      job << "#SBATCH --job-name=" << name << '\n';
      job << "#SBATCH --nodes=" << nNode << '\n';
      job << "#SBATCH --tasks-per-node=" << nTaskPerNode << '\n';
      job << "#SBATCH --cpus-per-task=1\n";
      job << "#SBATCH --time=12:00:00\n";
      job << "#SBATCH --mem=4GB\n";
      //job << "#SBATCH --mail-type=END\n";
      //job << "#SBATCH --mail-user=zl2972@nyu.edu\n";
      job << "#SBATCH --output=" << name << ".out\n";
      job << "\n";
      job << "module purge\n";
      job << "module load openmpi/gnu/4.0.2\n";
      job << "\n";
      job << "mpiexec " << ver.exec << ' ' << n << '\n';
    }
  }
  return 0;
}
