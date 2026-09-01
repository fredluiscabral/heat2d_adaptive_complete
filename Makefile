CXX      := g++
CXXFLAGS := -O3 -std=c++17 -fopenmp -Wall -Wextra -Wpedantic
LDFLAGS  := -fopenmp -pthread

COMMON := heat2d_explicit_common.hpp

# -----------------------------------------------------------------------------
# Current MPI-like study
# -----------------------------------------------------------------------------
MPI_BASE_SRC := heat2d_explicit_omp_mpilike.cpp
MPI_ORCL_SRC := heat2d_explicit_omp_mpilike_oracle.cpp
MPI_CAL_SRC  := heat2d_explicit_omp_mpilike_calibrate.cpp
MPI_OFF_SRC  := heat2d_explicit_omp_mpilike_adaptive_calibrated.cpp
MPI_ONL_SRC  := heat2d_explicit_omp_mpilike_adaptive_online.cpp
MPI_CMP_SRC  := heat2d_explicit_omp_mpilike_adaptive_compact.cpp

MPI_BASE_BIN := heat2d_explicit_omp_mpilike
MPI_ORCL_BIN := heat2d_dependency_oracle
MPI_CAL_BIN  := heat2d_dependency_calibrate
MPI_OFF_BIN  := heat2d_adaptive_calibrated
MPI_ONL_BIN  := heat2d_adaptive_online
MPI_CMP_BIN  := heat2d_adaptive_compact

# -----------------------------------------------------------------------------
# Busy-wait / semaphore baselines with false-sharing mitigation
# -----------------------------------------------------------------------------
BW_BASE_SRC := heat2d_explicit_omp_busywait_nobarrier_nofs.cpp
SM_BASE_SRC := heat2d_explicit_omp_sem_nobarrier_nofs.cpp
BW_BASE_BIN := heat2d_explicit_omp_busywait_nobarrier_nofs
SM_BASE_BIN := heat2d_explicit_omp_sem_nobarrier_nofs

# -----------------------------------------------------------------------------
# Backend-specific WAIT calibration
# -----------------------------------------------------------------------------
BW_WCAL_SRC := heat2d_explicit_omp_busywait_nobarrier_nofs_wait_calibrate.cpp
SM_WCAL_SRC := heat2d_explicit_omp_sem_nobarrier_nofs_wait_calibrate.cpp
BW_WCAL_BIN := heat2d_wait_calibrate_busywait
SM_WCAL_BIN := heat2d_wait_calibrate_sem

# -----------------------------------------------------------------------------
# Adaptive shared-array variants
# -----------------------------------------------------------------------------
BW_ADP_SRC := heat2d_explicit_omp_busywait_nobarrier_nofs_adaptive.cpp
SM_ADP_SRC := heat2d_explicit_omp_sem_nobarrier_nofs_adaptive.cpp
BW_ADP_BIN := heat2d_adaptive_busywait
SM_ADP_BIN := heat2d_adaptive_sem

BW_PROF_BIN := heat2d_adaptive_busywait_profile
SM_PROF_BIN := heat2d_adaptive_sem_profile

.PHONY: all current baselines calibrators adaptive diagnostics clean

all: current baselines calibrators adaptive diagnostics

current: $(MPI_BASE_BIN) $(MPI_ORCL_BIN) $(MPI_CAL_BIN) $(MPI_OFF_BIN) $(MPI_ONL_BIN) $(MPI_CMP_BIN)
baselines: $(BW_BASE_BIN) $(SM_BASE_BIN)
calibrators: $(BW_WCAL_BIN) $(SM_WCAL_BIN)
adaptive: $(BW_ADP_BIN) $(SM_ADP_BIN)
diagnostics: $(BW_PROF_BIN) $(SM_PROF_BIN)

$(MPI_BASE_BIN): $(MPI_BASE_SRC) $(COMMON)
	$(CXX) $(CXXFLAGS) $(MPI_BASE_SRC) -o $@ $(LDFLAGS)

$(MPI_ORCL_BIN): $(MPI_ORCL_SRC) $(COMMON)
	$(CXX) $(CXXFLAGS) $(MPI_ORCL_SRC) -o $@ $(LDFLAGS)

$(MPI_CAL_BIN): $(MPI_CAL_SRC) $(COMMON)
	$(CXX) $(CXXFLAGS) -DHEAT2D_PROFILE_STATS=0 $(MPI_CAL_SRC) -o $@ $(LDFLAGS)

$(MPI_OFF_BIN): $(MPI_OFF_SRC) $(COMMON)
	$(CXX) $(CXXFLAGS) -DHEAT2D_PROFILE_STATS=0 $(MPI_OFF_SRC) -o $@ $(LDFLAGS)

$(MPI_ONL_BIN): $(MPI_ONL_SRC) $(COMMON)
	$(CXX) $(CXXFLAGS) -DHEAT2D_PROFILE_STATS=0 $(MPI_ONL_SRC) -o $@ $(LDFLAGS)

$(MPI_CMP_BIN): $(MPI_CMP_SRC) $(COMMON)
	$(CXX) $(CXXFLAGS) -DHEAT2D_PROFILE_STATS=0 $(MPI_CMP_SRC) -o $@ $(LDFLAGS)

$(BW_BASE_BIN): $(BW_BASE_SRC) $(COMMON)
	$(CXX) $(CXXFLAGS) $(BW_BASE_SRC) -o $@ $(LDFLAGS)

$(SM_BASE_BIN): $(SM_BASE_SRC) $(COMMON)
	$(CXX) $(CXXFLAGS) $(SM_BASE_SRC) -o $@ $(LDFLAGS)

$(BW_WCAL_BIN): $(BW_WCAL_SRC) $(COMMON)
	$(CXX) $(CXXFLAGS) $(BW_WCAL_SRC) -o $@ $(LDFLAGS)

$(SM_WCAL_BIN): $(SM_WCAL_SRC) $(COMMON)
	$(CXX) $(CXXFLAGS) $(SM_WCAL_SRC) -o $@ $(LDFLAGS)

$(BW_ADP_BIN): $(BW_ADP_SRC) $(COMMON)
	$(CXX) $(CXXFLAGS) -DHEAT2D_PROFILE_STATS=0 $(BW_ADP_SRC) -o $@ $(LDFLAGS)

$(SM_ADP_BIN): $(SM_ADP_SRC) $(COMMON)
	$(CXX) $(CXXFLAGS) -DHEAT2D_PROFILE_STATS=0 $(SM_ADP_SRC) -o $@ $(LDFLAGS)

$(BW_PROF_BIN): $(BW_ADP_SRC) $(COMMON)
	$(CXX) $(CXXFLAGS) -DHEAT2D_PROFILE_STATS=1 $(BW_ADP_SRC) -o $@ $(LDFLAGS)

$(SM_PROF_BIN): $(SM_ADP_SRC) $(COMMON)
	$(CXX) $(CXXFLAGS) -DHEAT2D_PROFILE_STATS=1 $(SM_ADP_SRC) -o $@ $(LDFLAGS)

clean:
	rm -f $(MPI_BASE_BIN) $(MPI_ORCL_BIN) $(MPI_CAL_BIN) $(MPI_OFF_BIN) $(MPI_ONL_BIN) $(MPI_CMP_BIN)
	rm -f $(BW_BASE_BIN) $(SM_BASE_BIN) $(BW_WCAL_BIN) $(SM_WCAL_BIN)
	rm -f $(BW_ADP_BIN) $(SM_ADP_BIN) $(BW_PROF_BIN) $(SM_PROF_BIN)
	rm -f heat2d_cost_model.dat heat2d_wait_cost_busywait.dat heat2d_wait_cost_semaphore.dat
	rm -f output.txt adaptive_stats.txt
