CXX = mpic++
CXXFLAGS = -std=c++17 -O3

TARGETS = $(basename $(wildcard *.cpp))

all: $(TARGETS)

%: %.cpp *.h
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

run:
	mpiexec -np 4 ./jacobi 10 1000

clean:
	-$(RM) $(TARGETS) *~

.PHONY: all, clean
