VERSION := V1_$(shell date "+%d_%m_%Y_%H-%M-%S")

# ------------------------------ Build configuration ------------------------------
USE_NUMBER_OF_COLORS ?= 3
USE_NUMBER_OF_DIMS ?= 4
BUILD_MULTI_GPU ?= $(MGPU)
MEASURE_TIMMINGS ?= yes
USE_CUDARNG ?= MRG32k3a           # XORWOW / MRG32k3a
START_LATTICE_PARTITION_BY_X ?= no
GPU_GLOBAL_SET_CACHE_PREFER_L1 ?= yes
USE_MPI_GPU_DIRECT ?= no
GAUGEFIX_AUTO_TUNE_FFT_ALPHA ?= yes
USE_THETA_STOP_GAUGEFIX ?= yes
USE_GPU_FAST_MATH ?= no
USE_CUDA_CUB ?= yes
USE_GAUGE_FIX ?= no
USE_GAUGE_FIX_COV ?= no
DEBUG ?= no

MPI_INC ?= /usr/include/openmpi-x86_64
CUDA_PATH ?= /usr/local/cuda
# Auto-detect GPU arch from nvidia-smi (first GPU). Override with:
#   make GPU_ARCH=sm_90
GPU_CC := $(strip $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | awk 'NR==1{gsub(/\./,""); if ($$1 ~ /^[0-9]+$$/) print $$1}'))
GPU_ARCH ?= $(if $(GPU_CC),sm_$(GPU_CC),sm_120)

# ------------------------------ Toolchain ------------------------------
ifeq ($(strip $(BUILD_MULTI_GPU)),yes)
  CXX ?= mpic++
else
  CXX ?= g++
endif

NVCC ?= $(CUDA_PATH)/bin/nvcc
CUDA_INC_PATH := $(CUDA_PATH)/include
CUDA_LIB_PATH := $(CUDA_PATH)/lib64
CUDA_DRIVER_LIB_PATH ?= $(if $(wildcard /usr/lib/wsl/lib/libcuda.so),/usr/lib/wsl/lib,$(CUDA_LIB_PATH))
OS_ARCH := $(shell uname -m)

# ------------------------------ CUDA version / hash ------------------------------
CUDA_VERSION := $(shell awk '/^#define CUDA_VERSION/{print $$3}' $(CUDA_PATH)/include/cuda.h)
CUDA_VERSION_MAJOR := $(shell expr $(CUDA_VERSION) / 1000)
CUDA_VERSION_MINOR := $(shell expr \( $(CUDA_VERSION) % 1000 \) / 10)
CUDA_VERSION_PATCH := $(shell expr $(CUDA_VERSION) % 10)
HASH := \"cpu_arch=$(OS_ARCH),gpu_arch=$(GPU_ARCH),cuda_version=$(CUDA_VERSION)\"

# ------------------------------ Flags ------------------------------
COMP_CAP := $(GPU_ARCH:sm_%=%0)
is_yes = $(filter yes,$(strip $(1)))

CPPFLAGS := -I$(CUDA_INC_PATH) -I. -I./include
COMMON_DEFS := -DNCOLORS=$(USE_NUMBER_OF_COLORS) -DNDIMS=$(USE_NUMBER_OF_DIMS) -D__COMPUTE_CAPABILITY__=$(COMP_CAP)

COMMON_DEFS += $(if $(call is_yes,$(MEASURE_TIMMINGS)),-DTIMMINGS)
COMMON_DEFS += $(if $(call is_yes,$(USE_THETA_STOP_GAUGEFIX)),-DUSE_THETA_STOP_GAUGEFIX)
COMMON_DEFS += $(if $(call is_yes,$(GPU_GLOBAL_SET_CACHE_PREFER_L1)),-DGLOBAL_SET_CACHE_PREFER_L1)
COMMON_DEFS += $(if $(call is_yes,$(GAUGEFIX_AUTO_TUNE_FFT_ALPHA)),-DGAUGEFIX_AUTOTUNEFFT_ALPHA)
COMMON_DEFS += $(if $(call is_yes,$(USE_CUDA_CUB)),-DUSE_CUDA_CUB)
COMMON_DEFS += $(if $(call is_yes,$(USE_GAUGE_FIX)),-DUSE_GAUGE_FIX)
COMMON_DEFS += $(if $(call is_yes,$(USE_GAUGE_FIX_COV)),-DUSE_GAUGE_FIX_COV)

ifeq ($(strip $(USE_CUDARNG)),XORWOW)
  COMMON_DEFS += -DXORWOW
else
  COMMON_DEFS += -DMRG32k3a
endif

ifeq ($(call is_yes,$(BUILD_MULTI_GPU)),yes)
  COMMON_DEFS += -DMULTI_GPU
  CPPFLAGS += -I$(MPI_INC)
  ifeq ($(call is_yes,$(USE_MPI_MVAPICH)),yes)
    COMMON_DEFS += -DMPI_MVAPICH
  endif
  ifeq ($(call is_yes,$(USE_MPI_OPENMPI)),yes)
    COMMON_DEFS += -DMPI_OPENMPI
  endif
  ifeq ($(call is_yes,$(START_LATTICE_PARTITION_BY_X)),yes)
    COMMON_DEFS += -DSTART_LATTICE_PARTITION_BY_X
  endif
  ifeq ($(call is_yes,$(USE_MPI_GPU_DIRECT)),yes)
    COMMON_DEFS += -DGAUGEFIX_MPI_GPU_DIRECT -DMPI_GPU_DIRECT
  endif
endif

CPPFLAGS += $(COMMON_DEFS)
CXXFLAGS := -O3
NVCCFLAGS := -m64 -O3 -arch=$(GPU_ARCH)

ifeq ($(call is_yes,$(USE_GPU_FAST_MATH)),yes)
  NVCCFLAGS += -use_fast_math
endif
ifeq ($(call is_yes,$(DEBUG)),yes)
  CXXFLAGS += -g
  NVCCFLAGS += -g -G
endif

LDFLAGS := -L$(CUDA_LIB_PATH) -L$(CUDA_DRIVER_LIB_PATH)
LDLIBS := -lcudart -lcuda -lcufft -lcurand -lcublas
EXTRA_NVCCFLAGS ?=
EXTRA_LDFLAGS ?=
EXTRA_CCFLAGS ?=

# ------------------------------ Build outputs ------------------------------
LIBNAME := libSUN.a
SRCDIR := src
INCS := include

ifeq ($(call is_yes,$(BUILD_MULTI_GPU)),yes)
  OBJDIR := obj_mgpu
  LIBDIR := lib_mgpu
  PROJECTNAME := test_mgpu
  MAINOBJ := test_mgpu.o
else
  OBJDIR := obj_sgpu
  LIBDIR := lib_sgpu
  PROJECTNAME := test_sgpu
  MAINOBJ := test_sgpu.o
endif

HEATBATH_OBJ := heatbath_su$(USE_NUMBER_OF_COLORS)_nd$(USE_NUMBER_OF_DIMS).o
HEATBATH_EXE := heatbath_su$(USE_NUMBER_OF_COLORS)_nd$(USE_NUMBER_OF_DIMS)

# ------------------------------ Source layout ------------------------------
GAUGEFIX_OBJS := gauge_fix/gaugefix_fft.o gauge_fix/gaugefix_fft_stdorder.o gauge_fix/gaugefix_ovr.o \
                 gauge_fix/gaugefix_quality.o gauge_fix/gaugefix_quality_cub.o
MONTE_OBJS := monte/monte.o monte/ovr.o
MEAS_OBJS := meas/linkdetsum.o meas/linktrsum.o meas/plaquette.o meas/plaquette_cub.o \
             meas/pl.o meas/plr.o meas/plr3d.o meas/polyakovloop.o meas/linkUF.o meas/wilsonloop.o \
             meas/plaquettefield.o meas/plfield.o meas/chromofield.o
SMEAR_OBJS := smear/ape.o smear/hyp.o smear/multihitsp.o smear/multihit.o smear/stout.o smear/multihitext.o
WL_OBJS := wl/calcop_dg_A0.o wl/wilsonloop_dg_A0.o wl/calcop_dg_33.o wl/wilsonloop_dg.o

OBJS := timer.o random.o constants.o \
        reduction_kernel.o reduction.o reunitarize.o \
        devicemem.o gaugearray.o comm_mpi.o exchange.o \
        alloc.o tune.o io_gauge.o cuda_error_check.o \
        $(MONTE_OBJS) $(MEAS_OBJS) $(SMEAR_OBJS) $(WLEX_OBJS) \
        $(NEW_FIELDS_OBJS) $(MULTI_LEVEL) $(WL_OBJS) $(GAUGEFIX_OBJS)

CUDAOBJS := $(patsubst %.o,$(OBJDIR)/%.o,$(OBJS))
CUDAOBJ := $(OBJDIR)/dlink.o $(CUDAOBJS)
deps := $(MAINOBJ:.o=.d) $(CUDAOBJS:.o=.d)

# ------------------------------ User-facing info ------------------------------
$(info USE_NUMBER_OF_COLORS: $(USE_NUMBER_OF_COLORS))
$(info USE_NUMBER_OF_DIMS: $(USE_NUMBER_OF_DIMS))
$(info BUILD_MULTI_GPU: $(if $(filter yes,$(strip $(BUILD_MULTI_GPU))),yes,no))
$(info USE_CUDARNG: $(USE_CUDARNG))
$(info GLOBAL_SET_CACHE_PREFER_L1: $(GPU_GLOBAL_SET_CACHE_PREFER_L1))
$(info MEASURE_TIMMINGS: $(MEASURE_TIMMINGS))
$(info USE_THETA_STOP_GAUGEFIX: $(USE_THETA_STOP_GAUGEFIX))
$(info GAUGEFIX_AUTOTUNEFFT_ALPHA: $(GAUGEFIX_AUTO_TUNE_FFT_ALPHA))

# ------------------------------ Targets ------------------------------
all: lib $(PROJECTNAME) $(HEATBATH_EXE)
test: $(PROJECTNAME)
lib: $(LIBDIR)/$(LIBNAME)

COMPILE_CPP = $(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EXTRA_CCFLAGS) -MMD -MP -c $< -o $@
COMPILE_CU_DEP = $(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(CPPFLAGS) -M $< -o ${@:.o=.d} -odir $(@D)
COMPILE_CU = $(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(CPPFLAGS) -o $@ -dc $<

$(MAINOBJ): test.cpp
	$(COMPILE_CPP)

$(HEATBATH_OBJ): heatbath.cpp
	$(COMPILE_CPP)

$(PROJECTNAME): $(MAINOBJ) $(LIBDIR)/$(LIBNAME)
	$(CXX) $(CXXFLAGS) -I. -L. -o $@ $+ $(EXTRA_LDFLAGS) -L$(LIBDIR) -lSUN $(LDFLAGS) $(LDLIBS) -lcudadevrt

$(HEATBATH_EXE): $(HEATBATH_OBJ) $(LIBDIR)/$(LIBNAME)
	$(CXX) $(CXXFLAGS) -I. -L. -o $@ $+ $(EXTRA_LDFLAGS) -L$(LIBDIR) -lSUN $(LDFLAGS) $(LDLIBS) -lcudadevrt

$(OBJDIR)/gaugefix_ovr.o: $(SRCDIR)/gaugefix_ovr.cu $(SRCDIR)/gaugefix_ovr.cuh $(SRCDIR)/gaugefix_ovr_device.cuh
	@echo "######################### Compiling: $< #########################"
	@mkdir -p $(@D)
	$(COMPILE_CU_DEP)
	$(COMPILE_CU)

$(OBJDIR)/tune.o: $(SRCDIR)/tune.cpp
	@echo "######################### Compiling: $< #########################"
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DCULQCD_HASH=$(HASH) $(EXTRA_CCFLAGS) -MMD -MP -c $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	@echo "######################### Compiling: $< #########################"
	@mkdir -p $(@D)
	$(COMPILE_CU_DEP)
	$(COMPILE_CU)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@echo "######################### Compiling: $< #########################"
	@mkdir -p $(@D)
	$(COMPILE_CPP)

$(OBJDIR)/dlink.o: $(CUDAOBJS)
	@echo "######################### Creating: $(OBJDIR)/dlink.o #########################"
	$(NVCC) $(NVCCFLAGS) -dlink -o $@ $^ -lcudavert

$(LIBDIR)/$(LIBNAME): directories $(CUDAOBJ)
	@echo "######################### Creating: $(LIBDIR)/$(LIBNAME) #########################"
	@rm -f $(LIBDIR)/$(LIBNAME)
	ar rcs $(LIBDIR)/$(LIBNAME) $(CUDAOBJ)
	ranlib $(LIBDIR)/$(LIBNAME)

directories:
	@mkdir -p $(OBJDIR)
	@mkdir -p $(LIBDIR)
	@echo 
	@echo "GPU_ARCH: $(GPU_ARCH)$(if $(GPU_CC), (auto-detected compute capability=$(GPU_CC)),$(if $(filter sm_120,$(GPU_ARCH)), (fallback default),))"
	@echo "CUDA Toolkit version: $(CUDA_VERSION_MAJOR).$(CUDA_VERSION_MINOR).$(CUDA_VERSION_PATCH) (CUDA_VERSION=$(CUDA_VERSION))"
	@echo 

cleanall:
	@rm -rf $(OBJDIR) $(LIBDIR)
	@rm -f $(PROJECTNAME) $(MAINOBJ) child.o $(HEATBATH_OBJ) $(HEATBATH_EXE) *.d

clean:
	@rm -rf $(LIBDIR)
	@rm -f $(PROJECTNAME) $(MAINOBJ) child.o $(HEATBATH_OBJ) $(HEATBATH_EXE) *.d

pack:
	@echo Generating Package sun_$(VERSION).tar.gz
	@tar cvfz sun_$(VERSION).tar.gz *.cpp *.h $(INCS)/* $(SRCDIR)/* Makefile
	@echo Generated Package sun_$(VERSION).tar.gz

.PHONY: all test lib clean cleanall pack directories $(PROJECTNAME) $(HEATBATH_EXE)

-include $(deps)
