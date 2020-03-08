ROOT_DIR = .

SRCS = val_test01_solved.cpp val_test02_solved.cpp
OMP_SRCS = MMult1.cpp #omp_bug2.c omp_bug3.c omp_bug4.c omp_bug5.c omp_bug6.c

EXES = $(basename $(SRCS))
OMP_EXES = $(basename $(OMP_SRCS))

.DEFAULT_GOAL = rel
.PHONY: rel dbg clean

rel: CXXFLAGS += -O2
rel: CFLAGS += -O2
dbg: CXXFLAGS += -O0 -ggdb
dbg: CFLAGS += -O0 -ggdb

$(OMP_EXES): CXXFLAGS += -fopenmp
$(OMP_EXES): CFLAGS += -fopenmp

MMult1: CXXFLAGS = -fopenmp -march=native -O3

rel: $(EXES) $(OMP_EXES)
dbg: $(EXES) $(OMP_EXES)

clean:
	$(RM) $(EXES)
	$(RM) $(OMP_EXES)
