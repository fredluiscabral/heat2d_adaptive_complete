CXX      := g++
CXXFLAGS := -O3 -std=c++17 -fopenmp -Wall -Wextra -Wpedantic
LDFLAGS  := -fopenmp -pthread

COMMON := heat2d_explicit_common.hpp

# Active sources
MPI_BASE_SRC := heat2d_explicit_omp_mpilike.cpp
MPI_CAL_SRC  := heat2d_explicit_omp_mpilike_calibrate.cpp
BW_BASE_SRC  := heat2d_explicit_omp_busywait_nobarrier_nofs.cpp
SM_BASE_SRC  := heat2d_explicit_omp_sem_nobarrier_nofs.cpp
BW_WCAL_SRC  := heat2d_explicit_omp_busywait_nobarrier_nofs_wait_calibrate.cpp
SM_WCAL_SRC  := heat2d_explicit_omp_sem_nobarrier_nofs_wait_calibrate.cpp
BW_ADP_SRC   := heat2d_explicit_omp_busywait_nobarrier_nofs_adaptive.cpp
SM_ADP_SRC   := heat2d_explicit_omp_sem_nobarrier_nofs_adaptive.cpp

# Active binaries
MPI_BASE_BIN := heat2d_explicit_omp_mpilike
MPI_CAL_BIN  := heat2d_dependency_calibrate
BW_BASE_BIN  := heat2d_explicit_omp_busywait_nobarrier_nofs
SM_BASE_BIN  := heat2d_explicit_omp_sem_nobarrier_nofs
BW_WCAL_BIN  := heat2d_wait_calibrate_busywait
SM_WCAL_BIN  := heat2d_wait_calibrate_semaphore
BW_ADP_BIN   := heat2d_adaptive_busywait
SM_ADP_BIN   := heat2d_adaptive_sem
BW_PROF_BIN  := heat2d_adaptive_busywait_profile
SM_PROF_BIN  := heat2d_adaptive_sem_profile

.PHONY: all core baselines calibrators adaptive profiles clean

# Default: only the active research path.
all: core
core: baselines calibrators adaptive
baselines: $(MPI_BASE_BIN) $(BW_BASE_BIN) $(SM_BASE_BIN)
calibrators: $(MPI_CAL_BIN) $(BW_WCAL_BIN) $(SM_WCAL_BIN)
adaptive: $(BW_ADP_BIN) $(SM_ADP_BIN)
profiles: $(BW_PROF_BIN) $(SM_PROF_BIN)

$(MPI_BASE_BIN): $(MPI_BASE_SRC) $(COMMON)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

$(MPI_CAL_BIN): $(MPI_CAL_SRC) $(COMMON)
	$(CXX) $(CXXFLAGS) -DHEAT2D_PROFILE_STATS=0 $< -o $@ $(LDFLAGS)

$(BW_BASE_BIN): $(BW_BASE_SRC) $(COMMON)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

$(SM_BASE_BIN): $(SM_BASE_SRC) $(COMMON)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

$(BW_WCAL_BIN): $(BW_WCAL_SRC) $(COMMON)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

$(SM_WCAL_BIN): $(SM_WCAL_SRC) $(COMMON)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

$(BW_ADP_BIN): $(BW_ADP_SRC) $(COMMON)
	$(CXX) $(CXXFLAGS) -DHEAT2D_PROFILE_STATS=0 $< -o $@ $(LDFLAGS)

$(SM_ADP_BIN): $(SM_ADP_SRC) $(COMMON)
	$(CXX) $(CXXFLAGS) -DHEAT2D_PROFILE_STATS=0 $< -o $@ $(LDFLAGS)

$(BW_PROF_BIN): $(BW_ADP_SRC) $(COMMON)
	$(CXX) $(CXXFLAGS) -DHEAT2D_PROFILE_STATS=1 $< -o $@ $(LDFLAGS)

$(SM_PROF_BIN): $(SM_ADP_SRC) $(COMMON)
	$(CXX) $(CXXFLAGS) -DHEAT2D_PROFILE_STATS=1 $< -o $@ $(LDFLAGS)

clean:
	rm -f $(MPI_BASE_BIN) $(MPI_CAL_BIN)
	rm -f $(BW_BASE_BIN) $(SM_BASE_BIN) $(BW_WCAL_BIN) $(SM_WCAL_BIN)
	rm -f $(BW_ADP_BIN) $(SM_ADP_BIN) $(BW_PROF_BIN) $(SM_PROF_BIN)
	rm -f heat2d_cost_model.dat heat2d_wait_cost_busywait.dat heat2d_wait_cost_semaphore.dat
	rm -f output.txt adaptive_stats.txt
