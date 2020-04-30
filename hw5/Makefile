CXX = mpic++
CXXFLAGS = -std=c++17 -O3

TARGETS = $(basename $(wildcard *.cpp))

all: $(TARGETS)

%: %.cpp *.h
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	-$(RM) $(TARGETS)

.PHONY: all, clean
