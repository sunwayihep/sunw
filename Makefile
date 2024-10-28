############################################################################################################
VERSION  = V1_$(shell date "+%d_%m_%Y_%T")
VERSION  = V1_$(shell date "+%d_%m_%Y_%H-%M-%S")
STANDARD = c99
############################################################################################################
USE_NUMBER_OF_COLORS = 2 				#Set N of SU(N), needs make cleanall before recompile
BUILD_MULTI_GPU = $(MGPU) 				#no Multi-GPU: make or make MGPU=no, for Multi-GPU: make MGPU=yes / change "$(MGPU)" to no/yes
MEASURE_TIMMINGS = yes 					#report timing, gb/s and gflops
USE_CUDARNG = MRG32k3a 					#XORWOW/MRG32k3a  #CURAND type random generator
START_LATTICE_PARTITION_BY_X = no  		#MULTI_GPU lattice partition priority, no to start by T dimension
GPU_GLOBAL_SET_CACHE_PREFER_L1 = yes
USE_MPI_GPU_DIRECT = no 				#with MVAPICH2 run example with MPI GPU DIRECT -> mpirun -n 4 env MV2_USE_CUDA=1 ./program args 
GAUGEFIX_AUTO_TUNE_FFT_ALPHA = yes 		#yes for auto tune gauge fixing with FFTs
USE_THETA_STOP_GAUGEFIX = yes			#use theta as stop criterium else otherwise uses MILC criterium
USE_GPU_FAST_MATH = no 
USE_CUDA_CUB = yes 
USE_GAUGE_FIX = no
USE_GAUGE_FIX_COV = no

DEBUG = no
#MPI home folder
#MPI_HOME ?= /home/nuno/.mpi/
#MPI_INC ?= $(MPI_INC)/include/
MPI_HOME ?= /usr/lib64/openmpi/bin
MPI_INC ?= /usr/include/openmpi-x86_64 


ifeq ($(strip $(BUILD_MULTI_GPU)), yes)	#SET DEFAULT COMPILER
#GCC ?= $(MPI_HOME)/bin/mpic++
GCC ?= mpic++
else
GCC ?= g++
endif

#SET CUDA PATH
CUDA_PATH ?= /usr/local/cuda-11.7
#SET DEVICE ARQUITECTURE -> NO SUPPORT for SM 1.X!!!!!
GPU_ARCH = sm_80
GENCODE_FLAGS = -arch=$(GPU_ARCH)
#GENCODE_FLAGS = -arch=$(GPU_ARCH) --ptxas-options=-v
############################################################################################################
############################################################################################################
############################################################################################################
# OS Name (Linux or Darwin)
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])
# Flags to detect 32-bit or 64-bit OS platform
OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")
OS_ARCH = $(shell uname -m | sed -e "s/i386/i686/")
# These flags will override any settings
ifeq ($(i386),1)
	OS_SIZE = 32
	OS_ARCH = i686
endif
ifeq ($(x86_64),1)
	OS_SIZE = 64
	OS_ARCH = x86_64
endif
# Flags to detect either a Linux system (linux) or Mac OSX (darwin)
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))
# Location of the CUDA Toolkit binaries and libraries
#CHANGE TO YOUR CUDA PATH
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
ifneq ($(DARWIN),)
  CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
else
  ifeq ($(OS_SIZE),32)
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
  else
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib64
  endif
endif
# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc
# Extra user flags
EXTRA_NVCCFLAGS ?=
EXTRA_LDFLAGS   ?=
# CUDA code generation flags
#NO SUPPORT for SM1.X
#GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
#GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
#GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
#GENCODE_FLAGS   :=  $(GENCODE_SM20) --ptxas-options=-v
#  -Xptxas -dlcm=cg
# --maxrregcount=63
# -use_fast_math -Xptxas -dlcm=cg  
# OS-specific build flags
ifneq ($(DARWIN),) 
      LDFLAGS   := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart -lcuda  -lcufft -lcurand -lcublas
      CCFLAGS   := -arch $(OS_ARCH) 
else
  ifeq ($(OS_SIZE),32)
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart -lcuda  -lcufft -lcurand -lcublas 
 #     CCFLAGS   := -m32 -O3
      CCFLAGS   := -O3
  else
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart  -lcuda  -lcufft -lcurand -lcublas 
#      CCFLAGS   := -m64 -O3
      CCFLAGS   :=  -O3
  endif
endif
# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
      NVCCFLAGS := -m32 -O3
else
      NVCCFLAGS := -m64 -O3 
endif
# Debug build flags
ifeq ($(strip $(DEBUG)), yes)
      CCFLAGS   += -g
      NVCCFLAGS += -g -G
      TARGET := debug
else
      TARGET := release
endif


#
COMP_CAP = $(GPU_ARCH:sm_%=%0)
NVCCFLAGS += -D__COMPUTE_CAPABILITY__=$(COMP_CAP)
CCFLAGS += -D__COMPUTE_CAPABILITY__=$(COMP_CAP) 

INCLUDES := -I$(CUDA_INC_PATH)/ -I.  -I./include/  $(GENERICFLAG) -lcuda -lcudart  -lcufft -lcurand -lcublas 

LIBNAME=libSUN.a
OBJLIB=libSUNobj.a

ifeq ($(strip $(USE_NUMBER_OF_COLORS)),)
#$(error USE_NUMBER_OF_COLORS is not set)
$(warning USE_NUMBER_OF_COLORS is not set. Setting default value: 3)
USE_NUMBER_OF_COLORS = 3
SUNPROPERTIES= -DNCOLORS=$(USE_NUMBER_OF_COLORS)
$(warning USE_NUMBER_OF_COLORS: $(USE_NUMBER_OF_COLORS))
else
SUNPROPERTIES= -DNCOLORS=$(USE_NUMBER_OF_COLORS)
$(warning USE_NUMBER_OF_COLORS: $(USE_NUMBER_OF_COLORS))
endif

ifeq ($(strip $(USE_GPU_FAST_MATH)), yes)
  NVCCFLAGS += -use_fast_math
$(warning Using -use_fast_math)
endif
ifeq ($(strip $(BUILD_MULTI_GPU)), yes)
  INCLUDES += -DMULTI_GPU 
  INCLUDES += -I$(MPI_INC)
  $(warning Compiling to multi-GPU)
ifeq ($(strip $(USE_MPI_MVAPICH)), yes)
	INCLUDES += -DMPI_MVAPICH
endif
ifeq ($(strip $(USE_MPI_OPENMPI)), yes)
	INCLUDES += -DMPI_OPENMPI
endif
ifeq ($(strip $(START_LATTICE_PARTITION_BY_X)), yes)
  INCLUDES += -DSTART_LATTICE_PARTITION_BY_X
$(warning Lattice partition priority starting by X)
else
$(warning Lattice partition priority starting by T)
endif
OBJDIR=obj_mgpu
LIBDIR=lib_mgpu
PROJECTNAME=test_mgpu
MAINOBJ:=test_mgpu.o
else
$(warning Compililing to single-GPU)
OBJDIR=obj_sgpu
LIBDIR=lib_sgpu
PROJECTNAME=test_sgpu
MAINOBJ:=test_sgpu.o
endif
$(warning GLOBAL_SET_CACHE_PREFER_L1: $(GPU_GLOBAL_SET_CACHE_PREFER_L1))
$(warning MEASURE_TIMMINGS: $(MEASURE_TIMMINGS))
$(warning USE_THETA_STOP_GAUGEFIX: $(USE_THETA_STOP_GAUGEFIX))
$(warning GAUGEFIX_AUTOTUNEFFT_ALPHA: $(GAUGEFIX_AUTO_TUNE_FFT_ALPHA))
ifeq ($(strip $(MEASURE_TIMMINGS)), yes)
  INCLUDES += -DTIMMINGS
endif

ifeq ($(strip $(USE_THETA_STOP_GAUGEFIX)), yes)
  INCLUDES += -DUSE_THETA_STOP_GAUGEFIX
endif
ifeq ($(strip $(GPU_GLOBAL_SET_CACHE_PREFER_L1)), yes)
  INCLUDES += -DGLOBAL_SET_CACHE_PREFER_L1
endif

ifeq ($(strip $(GAUGEFIX_AUTO_TUNE_FFT_ALPHA)), yes)
  INCLUDES += -DGAUGEFIX_AUTOTUNEFFT_ALPHA
endif


ifeq ($(strip $(USE_MPI_GPU_DIRECT)), yes)
	ifeq ($(strip $(BUILD_MULTI_GPU)), yes)
 	 INCLUDES += -DGAUGEFIX_MPI_GPU_DIRECT -DMPI_GPU_DIRECT
 	 $(warning USE_MPI_GPU_DIRECT: $(USE_MPI_GPU_DIRECT))
 	 endif
 else
	 ifeq ($(strip $(BUILD_MULTI_GPU)), yes)
 	 $(warning USE_MPI_GPU_DIRECT: no)
 	 endif
endif


ifeq ($(strip $(USE_CUDARNG)), XORWOW)
  INCLUDES += -DXORWOW
$(warning USE_CUDARNG: XORWOW)
else
  INCLUDES += -DMRG32k3a
$(warning USE_CUDARNG: MRG32k3a)
endif

ifeq ($(strip $(USE_CUDA_CUB)), yes)
  INCLUDES += -DUSE_CUDA_CUB
endif




ifeq ($(strip $(USE_GAUGE_FIX)), yes)
  INCLUDES += -DUSE_GAUGE_FIX
endif
ifeq ($(strip $(USE_GAUGE_FIX_COV)), yes)
  INCLUDES += -DUSE_GAUGE_FIX_COV
endif





############################################################################################################
#For tune file
CUDA_VERSION = $(shell awk '/\#define CUDA_VERSION/{print $$3}' $(CUDA_PATH)/include/cuda.h)
HASH = \"cpu_arch=$(strip $(OS_ARCH)),gpu_arch=$(strip $(GPU_ARCH)),cuda_version=$(strip $(CUDA_VERSION))\"
############################################################################################################
############################################################################################################

HEATBATH_OBJ=heatbath_su$(strip $(USE_NUMBER_OF_COLORS)).o
HEATBATH_EXE=heatbath_su$(strip $(USE_NUMBER_OF_COLORS))
# Target rules
all: lib  $(PROJECTNAME) $(HEATBATH_EXE)
############################################################################################################
############################################################################################################
deps = $(MAINOBJ:.o=.d)
test: $(PROJECTNAME)
$(MAINOBJ): test.cpp
	$(VERBOSE)$(GCC) $(CCFLAGS) $(LDFLAGS)  $(EXTRA_CCFLAGS) $(INCLUDES) $(SUNPROPERTIES)  -MMD -MP  -c $< -o $@  

$(HEATBATH_OBJ): heatbath.cpp
	$(VERBOSE)$(GCC) $(CCFLAGS) $(LDFLAGS)  $(EXTRA_CCFLAGS) $(INCLUDES) $(SUNPROPERTIES)  -MMD -MP  -c $< -o $@  
 
$(PROJECTNAME):  $(MAINOBJ) $(LIBDIR)/$(LIBNAME)
	$(VERBOSE)$(GCC) $(CCFLAGS) -I. -L. -o $@ $+ $(SUNPROPERTIES)  $(EXTRA_LDFLAGS) -L$(LIBDIR)/ -lSUN  $(LDFLAGS)  -lcudadevrt

$(HEATBATH_EXE):  $(HEATBATH_OBJ) $(LIBDIR)/$(LIBNAME)
	$(VERBOSE)$(GCC) $(CCFLAGS) -I. -L. -o $@ $+ $(SUNPROPERTIES)  $(EXTRA_LDFLAGS) -L$(LIBDIR)/ -lSUN  $(LDFLAGS)  -lcudadevrt
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
lib: $(LIBDIR)/$(LIBNAME)
############################################################################################################
SRCDIR := src

GAUGEFIX_OBJS :=  gauge_fix/gaugefix_fft.o gauge_fix/gaugefix_fft_stdorder.o gauge_fix/gaugefix_ovr.o \
		gauge_fix/gaugefix_quality.o gauge_fix/gaugefix_quality_cub.o



MONTE_OBJS := monte/monte.o monte/ovr.o

MEAS_OBJS :=  meas/linkdetsum.o  meas/linktrsum.o meas/plaquette.o meas/plaquette_cub.o \
		meas/pl.o  meas/plr.o meas/plr3d.o meas/polyakovloop.o meas/linkUF.o meas/wilsonloop.o \
        meas/plaquettefield.o meas/plfield.o meas/chromofield.o



SMEAR_OBJS := smear/ape.o smear/hyp.o smear/multihitsp.o  smear/multihit.o smear/stout.o smear/multihitext.o

WL_OBJS := wl/calcop_dg_A0.o wl/wilsonloop_dg_A0.o wl/calcop_dg_33.o wl/wilsonloop_dg.o


OBJS := timer.o random.o constants.o texture.o  texture_host.o \
		reduction_kernel.o  reduction.o reunitarize.o \
		devicemem.o gaugearray.o  comm_mpi.o exchange.o \
		alloc.o   tune.o io_gauge.o cuda_error_check.o \
		$(MONTE_OBJS) $(MEAS_OBJS) $(SMEAR_OBJS) $(WLEX_OBJS) \
		$(NEW_FIELDS_OBJS) $(MULTI_LEVEL) $(WL_OBJS) $(GAUGEFIX_OBJS)
 #$(GAUGEFIX_OBJS) $(FIELDS_OBJS) 
		
INCS:= include
CUDAOBJS  := $(patsubst %.o,$(OBJDIR)/%.o,$(OBJS))
deps += $(CUDAOBJS:.o=.d)
############################################################################################################
$(OBJDIR)/gaugefix_ovr.o: $(SRCDIR)/gaugefix_ovr.cu $(SRCDIR)/gaugefix_ovr.cuh $(SRCDIR)/gaugefix_ovr_device.cuh
	@echo "######################### Compiling: "$<" #########################"
	$(VERBOSE)$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $(EXTRA_NVCCFLAGS) $(INCLUDES) $(SUNPROPERTIES) -M $< -o ${@:.o=.d} -odir $(@D)
	$(VERBOSE)$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $(EXTRA_NVCCFLAGS) $(INCLUDES) $(SUNPROPERTIES) -o $@ -dc $<
############################################################################################################
$(OBJDIR)/tune.o: $(SRCDIR)/tune.cpp
	@echo "######################### Compiling: "$<" #########################"
	$(VERBOSE)$(GCC) $(CCFLAGS) $(LDFLAGS) -DCULQCD_HASH=$(HASH) $(EXTRA_CCFLAGS) $(INCLUDES) $(SUNPROPERTIES)  -MMD -MP   -c $< -o $@  
############################################################################################################
############################################################################################################
$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	@echo "######################### Compiling: "$<" #########################"
	$(VERBOSE)mkdir -p $(dir $(abspath $@ ) )
	$(VERBOSE)$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $(EXTRA_NVCCFLAGS) $(INCLUDES) $(SUNPROPERTIES) -M $< -o ${@:.o=.d} -odir $(@D)
	$(VERBOSE)$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $(EXTRA_NVCCFLAGS) $(INCLUDES) $(SUNPROPERTIES) -o $@ -dc $<

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@echo "######################### Compiling: "$<" #########################"
	$(VERBOSE)$(GCC) $(CCFLAGS) $(LDFLAGS)  $(EXTRA_CCFLAGS) $(INCLUDES) $(SUNPROPERTIES)  -MMD -MP   -c $< -o $@ 
############################################################################################################
############################################################################################################
CUDAOBJ=$(OBJDIR)/dlink.o $(CUDAOBJS)
$(OBJDIR)/dlink.o: $(CUDAOBJS)
	@echo ""######################### Creating: "$(OBJDIR)/dlink.o" #########################"
	$(VERBOSE)$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -dlink -o $@ $^ -lcudavert
############################################################################################################
############################################################################################################
$(LIBDIR)/$(LIBNAME):  directories $(CUDAOBJ) 
	@echo ""######################### Creating: "$(LIBDIR)/$(LIBNAME)" #########################"
	$(VERBOSE)rm -f  $(LIBDIR)/$(LIBNAME)
	ar rcs $(LIBDIR)/$(LIBNAME)  $(CUDAOBJ)
	ranlib $(LIBDIR)/$(LIBNAME)
############################################################################################################
############################################################################################################
directories:
	$(VERBOSE)mkdir -p $(OBJDIR)
	$(VERBOSE)mkdir -p $(LIBDIR)
cleanall:
	$(VERBOSE)rm -r -f $(OBJDIR)
	$(VERBOSE)rm -r -f $(LIBDIR)
	$(VERBOSE)rm -f $(PROJECTNAME) $(MAINOBJ) child.o $(HEATBATH_OBJ) $(HEATBATH_EXE) *.d
clean:
	$(VERBOSE)rm -r -f $(LIBDIR)
	$(VERBOSE)rm -f $(PROJECTNAME) $(MAINOBJ) child.o $(HEATBATH_OBJ) $(HEATBATH_EXE) *.d

pack: 
	@echo Generating Package sun_$(VERSION).tar.gz
	@tar cvfz sun_$(VERSION).tar.gz *.cpp *.h $(INCS)/*  $(SRCDIR)/* Makefile
	@echo Generated Package sun_$(VERSION).tar.gz

.PHONY : clean cleanall pack directories lib $(PROJECTNAME) $(HEATBATH_EXE)

-include $(deps)
