CXX      := g++
CXXFLAGS := -O3 -std=c++17 -fopenmp
LDFLAGS  := -fopenmp

BASE_SRC := heat2d_explicit_omp_mpilike.cpp
ADAP_SRC := heat2d_explicit_omp_mpilike_adaptive.cpp

BASE_BIN := heat2d_explicit_omp_mpilike
VAL_BIN  := heat2d_adaptive_validate
PERF_BIN := heat2d_adaptive_perf

.PHONY: all baseline validate perf clean

all: baseline validate perf

baseline: $(BASE_BIN)

validate: $(VAL_BIN)

perf: $(PERF_BIN)

$(BASE_BIN): $(BASE_SRC) heat2d_explicit_common.hpp
	$(CXX) $(CXXFLAGS) $(BASE_SRC) -o $@ $(LDFLAGS)

$(VAL_BIN): $(ADAP_SRC) heat2d_explicit_common.hpp
	$(CXX) $(CXXFLAGS) -DHEAT2D_PROFILE_STATS=1 $(ADAP_SRC) -o $@ $(LDFLAGS)

$(PERF_BIN): $(ADAP_SRC) heat2d_explicit_common.hpp
	$(CXX) $(CXXFLAGS) -DHEAT2D_PROFILE_STATS=0 $(ADAP_SRC) -o $@ $(LDFLAGS)

clean:
	rm -f $(BASE_BIN) $(VAL_BIN) $(PERF_BIN)