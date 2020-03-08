ROOT_DIR = .

SRCS = val_test01_solved.cpp val_test02_solved.cpp
OMP_SRCS = MMult1.cpp #omp_bug2.c omp_bug3.c omp_bug4.c omp_bug5.c omp_bug6.c

EXES = $(basename $(SRCS))
OMP_EXES = $(basename $(OMP_SRCS))

CXXFLAGS += -std=c++17
CFLAGS += -std=c11

.DEFAULT_GOAL = rel
.PHONY: rel dbg clean

rel: CXXFLAGS += -O2
rel: CFLAGS += -O2
dbg: CXXFLAGS += -O0 -ggdb
dbg: CFLAGS += -O0 -ggdb

MMult1: CXXFLAGS += -march=native

$(OMP_EXES): CXXFLAGS += -fopenmp
$(OMP_EXES): CFLAGS += -fopenmp

rel: $(EXES) $(OMP_EXES)
dbg: $(EXES) $(OMP_EXES)

clean:
	$(RM) $(EXES)
	$(RM) $(OMP_EXES)
