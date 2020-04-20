NVCC = nvcc
NVCCFLAGS = -std=c++14 -O3 -Xcompiler -march=native -Xcompiler -fopenmp

TARGETS = $(basename $(wildcard *.cu))

all : $(TARGETS)

%:%.cu *.h
	$(NVCC) $(NVCCFLAGS) $< $(LIBS) -o $@

clean:
	-$(RM) $(TARGETS) *~

.PHONY: all, clean
