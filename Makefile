CXX      := g++
CXXFLAGS := -O3 -std=c++17 -fopenmp -Wall -Wextra -Wpedantic
LDFLAGS  := -fopenmp

COMMON   := heat2d_explicit_common.hpp

BASE_SRC := heat2d_explicit_omp_mpilike.cpp
ORCL_SRC := heat2d_explicit_omp_mpilike_oracle.cpp
CAL_SRC  := heat2d_explicit_omp_mpilike_calibrate.cpp
OFF_SRC  := heat2d_explicit_omp_mpilike_adaptive_calibrated.cpp
ONL_SRC  := heat2d_explicit_omp_mpilike_adaptive_online.cpp

BASE_BIN := heat2d_explicit_omp_mpilike
ORCL_BIN := heat2d_dependency_oracle
CAL_BIN  := heat2d_dependency_calibrate
OFF_BIN  := heat2d_adaptive_calibrated
ONL_BIN  := heat2d_adaptive_online

.PHONY: all baseline oracle calibrate calibrated online clean

all: baseline oracle calibrate calibrated online

baseline: $(BASE_BIN)
oracle: $(ORCL_BIN)
calibrate: $(CAL_BIN)
calibrated: $(OFF_BIN)
online: $(ONL_BIN)

$(BASE_BIN): $(BASE_SRC) $(COMMON)
	$(CXX) $(CXXFLAGS) $(BASE_SRC) -o $@ $(LDFLAGS)

$(ORCL_BIN): $(ORCL_SRC) $(COMMON)
	$(CXX) $(CXXFLAGS) $(ORCL_SRC) -o $@ $(LDFLAGS)

$(CAL_BIN): $(CAL_SRC) $(COMMON)
	$(CXX) $(CXXFLAGS) -DHEAT2D_PROFILE_STATS=0 $(CAL_SRC) -o $@ $(LDFLAGS)

$(OFF_BIN): $(OFF_SRC) $(COMMON)
	$(CXX) $(CXXFLAGS) -DHEAT2D_PROFILE_STATS=0 $(OFF_SRC) -o $@ $(LDFLAGS)

$(ONL_BIN): $(ONL_SRC) $(COMMON)
	$(CXX) $(CXXFLAGS) -DHEAT2D_PROFILE_STATS=0 $(ONL_SRC) -o $@ $(LDFLAGS)

clean:
	rm -f $(BASE_BIN) $(ORCL_BIN) $(CAL_BIN) $(OFF_BIN) $(ONL_BIN)
	rm -f heat2d_cost_model.dat output.txt adaptive_stats.txt
