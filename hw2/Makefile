SRCS = val_test01_solved.cpp val_test02_solved.cpp
OMP_SRCS = omp_solved2.c omp_solved3.c omp_solved4.c omp_solved5.c omp_solved6.c
OMP_SRCS += MMult1.cpp jacobi2D-omp.cpp gs2D-omp.cpp

EXES = $(basename $(SRCS))
OMP_EXES = $(basename $(OMP_SRCS))

.DEFAULT_GOAL = rel
.PHONY: rel dbg clean

rel: CXXFLAGS += -O3 -march=native
rel: CFLAGS += -O3 -march=native
dbg: CXXFLAGS += -O0 -ggdb
dbg: CFLAGS += -O0 -ggdb

$(OMP_EXES): CXXFLAGS += -fopenmp
$(OMP_EXES): CFLAGS += -fopenmp

rel: $(EXES) $(OMP_EXES)
dbg: $(EXES) $(OMP_EXES)

clean:
	$(RM) $(EXES)
	$(RM) $(OMP_EXES)
