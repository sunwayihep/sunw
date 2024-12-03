#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <execinfo.h> // for backtrace
#include <map>
#include <string>
#include <unistd.h> // for getpagesize()

#include <cuda.h>
#include <cuda_runtime.h>

#include <comm_mpi.h>
#include <cuda_common.h>
#include <cuda_error_check.h>
#include <modes.h>

// CODE FROM QUDA LIBRARY WITH A FEW MODIFICATIONS

namespace CULQCD {

enum AllocType {
  DEVICE_PTR,
  DEVICE_PINNED_PTR,
  HOST_PTR,
  PINNED_PTR,
  MAPPED_PTR,
  MANAGED_PTR,
  N_ALLOC_TYPE
};

class MemAlloc {

public:
  std::string func;
  std::string file;
  int line;
  size_t size;
  size_t base_size;

  MemAlloc() : line(-1), size(0), base_size(0) {}

  MemAlloc(std::string func, std::string file, int line)
      : func(func), file(file), line(line), size(0), base_size(0) {}

  MemAlloc &operator=(const MemAlloc &a) {
    if (&a != this) {
      func = a.func;
      file = a.file;
      line = a.line;
      size = a.size;
      base_size = a.base_size;
    }
    return *this;
  }
};

static std::map<void *, MemAlloc> alloc[N_ALLOC_TYPE];
static long total_bytes[N_ALLOC_TYPE] = {0};
static long max_total_bytes[N_ALLOC_TYPE] = {0};
static long total_host_bytes, max_total_host_bytes;
static long total_pinned_bytes, max_total_pinned_bytes;

long dev_allocated_peak() { return max_total_bytes[DEVICE_PTR]; }

long pinned_allocated_peak() { return max_total_bytes[PINNED_PTR]; }

long mapped_allocated_peak() { return max_total_bytes[MAPPED_PTR]; }

long host_allocated_peak() { return max_total_bytes[HOST_PTR]; }

long managed_allocated_peak() { return max_total_bytes[MANAGED_PTR]; }

static void print_trace(void) {
  void *array[10];
  size_t size;
  char **strings;
  size = backtrace(array, 10);
  strings = backtrace_symbols(array, size);
  printfCULQCD("Obtained %zd stack frames.\n", size);
  for (size_t i = 0; i < size; i++)
    printfCULQCD("%s\n", strings[i]);
  free(strings);
}

static void print_alloc_header() {
  printfCULQCD("Type        Pointer          Size             Location\n");
  printfCULQCD("----------------------------------------------------------\n");
}

static void print_alloc(AllocType type) {
  const char *type_str[] = {"Device", "Device Pinned", "Host",
                            "Pinned", "Mapped",        "Managed"};
  std::map<void *, MemAlloc>::iterator entry;

  for (entry = alloc[type].begin(); entry != alloc[type].end(); entry++) {
    void *ptr = entry->first;
    MemAlloc a = entry->second;
    printfCULQCD("%s  %15p  %15lu  %s(), %s:%d\n", type_str[type], ptr,
                 (unsigned long)a.base_size, a.func.c_str(), a.file.c_str(),
                 a.line);
  }
}

static void track_malloc(const AllocType &type, const MemAlloc &a, void *ptr) {
  total_bytes[type] += a.base_size;
  if (total_bytes[type] > max_total_bytes[type]) {
    max_total_bytes[type] = total_bytes[type];
  }
  if (type != DEVICE_PTR && type != DEVICE_PINNED_PTR) {
    total_host_bytes += a.base_size;
    if (total_host_bytes > max_total_host_bytes) {
      max_total_host_bytes = total_host_bytes;
    }
  }
  if (type == PINNED_PTR || type == MAPPED_PTR) {
    total_pinned_bytes += a.base_size;
    if (total_pinned_bytes > max_total_pinned_bytes) {
      max_total_pinned_bytes = total_pinned_bytes;
    }
  }
  alloc[type][ptr] = a;
}

static void track_free(const AllocType &type, void *ptr) {
  size_t size = alloc[type][ptr].base_size;
  total_bytes[type] -= size;
  if (type != DEVICE_PTR && type != DEVICE_PINNED_PTR) {
    total_host_bytes -= size;
  }
  if (type == PINNED_PTR || type == MAPPED_PTR) {
    total_pinned_bytes -= size;
  }
  alloc[type].erase(ptr);
}

/**
 * Under CUDA 4.0, cudaHostRegister seems to require that both the
 * beginning and end of the buffer be aligned on page boundaries.
 * This local function takes care of the alignment and gets called
 * by pinned_malloc_() and mapped_malloc_()
 */
static void *aligned_malloc(MemAlloc &a, size_t size) {
  void *ptr = nullptr;

  a.size = size;

#if (CUDA_VERSION > 4000) &&                                                   \
    0 // we need to manually align to page boundaries to allow us to bind a
      // texture to mapped memory
  a.base_size = size;
  ptr = malloc(size);
  if (!ptr) {
#else
  static int page_size = 2 * getpagesize();
  a.base_size = ((size + page_size - 1) / page_size) *
                page_size; // round up to the nearest multiple of page_size
  int align = posix_memalign(&ptr, page_size, a.base_size);
  if (!ptr || align != 0) {
#endif
    printfCULQCD("ERROR: Failed to allocate aligned host memory of size %zu "
                 "(%s:%d in %s())\n",
                 size, a.file.c_str(), a.line, a.func.c_str());
    errorCULQCD("Aborting");
  }
  return ptr;
}

/**
 * Perform a standard cudaMalloc() with error-checking.  This
 * function should only be called via the dev_malloc() macro,
 * defined in alloc.h
 */
void *dev_malloc_(const char *func, const char *file, int line, size_t size) {
  MemAlloc a(func, file, line);
  void *ptr;

  a.size = a.base_size = size;

  cudaError_t err = cudaMalloc(&ptr, size);
  if (err != cudaSuccess) {
    printfCULQCD(
        "ERROR: Failed to allocate device memory of size %zu (%s:%d in %s())\n",
        size, file, line, func);
    errorCULQCD("Aborting");
  }
  track_malloc(DEVICE_PTR, a, ptr);

  // #ifdef HOST_DEBUG
  err = cudaMemset(ptr, 0, size);
  if (err != cudaSuccess) {
    printfCULQCD(
        "ERROR: Failed to set device memory of size %zu (%s:%d in %s())\n",
        size, file, line, func);
    errorCULQCD("Aborting");
  }
  // #endif

  return ptr;
}

/**
 * Perform a cuMemAlloc with error-checking.  This function is to
 * guarantee a unique memory allocation on the device, since
 * cudaMalloc can be redirected (as is the case with QDPJIT).  This
 * should only be called via the dev_pinned_malloc() macro,
 * defined in alloc.h.
 */
void *dev_pinned_malloc_(const char *func, const char *file, int line,
                         size_t size) {
  // if (!comm_peer2peer_present()) return dev_malloc_(func, file, line, size);

  MemAlloc a(func, file, line);
  void *ptr;

  a.size = a.base_size = size;

  CUresult err = cuMemAlloc((CUdeviceptr *)&ptr, size);
  if (err != CUDA_SUCCESS) {
    printfCULQCD(
        "ERROR: Failed to allocate device memory of size %zu (%s:%d in %s())\n",
        size, file, line, func);
    errorCULQCD("Aborting");
  }
  track_malloc(DEVICE_PINNED_PTR, a, ptr);
  // #ifdef HOST_DEBUG
  cudaError_t err1 = cudaMemset(ptr, 0, size);
  if (err1 != cudaSuccess) {
    printfCULQCD(
        "ERROR: Failed to set device memory of size %zu (%s:%d in %s())\n",
        size, file, line, func);
    errorCULQCD("Aborting");
  }
  // #endif
  return ptr;
}

/**
 * Perform a standard malloc() with error-checking.  This function
 * should only be called via the safe_malloc() macro, defined in
 * alloc.h
 */
void *safe_malloc_(const char *func, const char *file, int line, size_t size) {
  MemAlloc a(func, file, line);
  a.size = a.base_size = size;

  void *ptr = malloc(size);
  if (!ptr) {
    printfCULQCD(
        "ERROR: Failed to allocate host memory of size %zu (%s:%d in %s())\n",
        size, file, line, func);
    errorCULQCD("Aborting");
  }
  track_malloc(HOST_PTR, a, ptr);
  // #ifdef HOST_DEBUG
  memset(ptr, 0, size);
  // #endif
  return ptr;
}

/**
 * Allocate page-locked ("pinned") host memory.  This function
 * should only be called via the pinned_malloc() macro, defined in
 * alloc.h
 *
 * Note that we do not rely on cudaHostAlloc(), since buffers
 * allocated in this way have been observed to cause problems when
 * shared with MPI via GPU Direct on some systems.
 */
void *pinned_malloc_(const char *func, const char *file, int line,
                     size_t size) {
  MemAlloc a(func, file, line);
  void *ptr = aligned_malloc(a, size);

  cudaError_t err = cudaHostRegister(ptr, a.base_size, cudaHostRegisterDefault);
  if (err != cudaSuccess) {
    printfCULQCD(
        "ERROR: Failed to register pinned memory of size %zu (%s:%d in %s())\n",
        size, file, line, func);
    errorCULQCD("Aborting");
  }
  track_malloc(PINNED_PTR, a, ptr);
  memset(ptr, 0, a.base_size);
  return ptr;
}

#define HOST_ALLOC // this needs to be set presently on P9

/**
 * Allocate page-locked ("pinned") host memory, and map it into the
 * GPU address space.  This function should only be called via the
 * mapped_malloc() macro, defined in alloc.h
 */
void *mapped_malloc_(const char *func, const char *file, int line,
                     size_t size) {
  MemAlloc a(func, file, line);

#ifdef HOST_ALLOC
  void *ptr;
  cudaError_t err = cudaHostAlloc(
      &ptr, size, cudaHostRegisterMapped | cudaHostRegisterPortable);
  if (err != cudaSuccess) {
    printfCULQCD("ERROR: cudaHostAlloc failed of size %zu (%s:%d in %s())\n",
                 size, file, line, func);
    errorCULQCD("Aborting");
  }
#else
  void *ptr = aligned_malloc(a, size);
  cudaError_t err = cudaHostRegister(ptr, a.base_size, cudaHostRegisterMapped);
  if (err != cudaSuccess) {
    printfCULQCD("ERROR: Failed to register host-mapped memory of size %zu "
                 "(%s:%d in %s())\n",
                 size, file, line, func);
    errorCULQCD("Aborting");
  }
#endif
  track_malloc(MAPPED_PTR, a, ptr);
  memset(ptr, 0, a.base_size);
  return ptr;
}

bool use_managed_memory() {
  static bool managed = false;
  static bool init = false;

  if (!init) {
    char *enable_managed_memory = getenv("CULQCD_ENABLE_MANAGED_MEMORY");
    if (enable_managed_memory && strcmp(enable_managed_memory, "1") == 0) {
      printfCULQCD("Warning: Using managed memory for CUDA allocations");
      managed = true;

      cudaDeviceProp deviceProp;
      int dev;
      cudaSafeCall(cudaGetDevice(&dev));
      cudaGetDeviceProperties(&deviceProp, dev);

      if (deviceProp.major < 6)
        printfCULQCD("Warning: Using managed memory on pre-Pascal architecture "
                     "is limited\n");
    }

    init = true;
  }

  return managed;
}

bool is_prefetch_enabled() {
  static bool prefetch = false;
  static bool init = false;

  if (!init) {
    if (use_managed_memory()) {
      char *enable_managed_prefetch = getenv("CULQCD_ENABLE_MANAGED_PREFETCH");
      if (enable_managed_prefetch &&
          strcmp(enable_managed_prefetch, "1") == 0) {
        printfCULQCD("Warning: Enabling prefetch support for managed memory");
        prefetch = true;
      }
    }

    init = true;
  }

  return prefetch;
}

/**
 * Perform a standard cudaMallocManaged() with error-checking.  This
 * function should only be called via the managed_malloc() macro
 */
void *managed_malloc_(const char *func, const char *file, int line,
                      size_t size) {
  MemAlloc a(func, file, line);
  void *ptr;

  a.size = a.base_size = size;

  cudaError_t err = cudaMallocManaged(&ptr, size);
  if (err != cudaSuccess) {
    errorCULQCD(
        "Failed to allocate managed memory of size %zu (%s:%d in %s())\n", size,
        file, line, func);
  }
  track_malloc(MANAGED_PTR, a, ptr);
#ifdef HOST_DEBUG
  // #ifdef HOST_DEBUG
  cudaError_t err1 = cudaMemset(ptr, 0, size);
  if (err1 != cudaSuccess) {
    printfCULQCD(
        "ERROR: Failed to set managed memory of size %zu (%s:%d in %s())\n",
        size, file, line, func);
    errorCULQCD("Aborting\n");
  }
#endif
  return ptr;
}

/**
 * Free device memory allocated with device_malloc().  This function
 * should only be called via the device_free() macro, defined in
 * malloc_quda.h
 */
void managed_free_(const char *func, const char *file, int line, void *ptr) {
  if (!ptr) {
    errorCULQCD("Attempt to free NULL managed pointer (%s:%d in %s())\n", file,
                line, func);
  }
  if (!alloc[MANAGED_PTR].count(ptr)) {
    errorCULQCD("Attempt to free invalid managed pointer (%s:%d in %s())\n",
                file, line, func);
  }
  cudaError_t err = cudaFree(ptr);
  if (err != cudaSuccess) {
    errorCULQCD("Failed to free device memory (%s:%d in %s())\n", file, line,
                func);
  }
  track_free(MANAGED_PTR, ptr);
}

/**
 * Free device memory allocated with dev_malloc().  This function
 * should only be called via the dev_free() macro, defined in
 * alloc.h
 */
void dev_free_(const char *func, const char *file, int line, void *ptr) {
  if (!ptr) {
    printfCULQCD("ERROR: Attempt to free NULL device pointer (%s:%d in %s())\n",
                 file, line, func);
    errorCULQCD("Aborting");
  }
  if (!alloc[DEVICE_PTR].count(ptr)) {
    printfCULQCD(
        "ERROR: Attempt to free invalid device pointer (%s:%d in %s())\n", file,
        line, func);
    errorCULQCD("Aborting");
  }
  cudaError_t err = cudaFree(ptr);
  if (err != cudaSuccess) {
    printfCULQCD("ERROR: Failed to free device memory (%s:%d in %s())\n", file,
                 line, func);
    errorCULQCD("Aborting");
  }
  track_free(DEVICE_PTR, ptr);
}

/**
 * Free device memory allocated with dev_pinned malloc().  This
 * function should only be called via the dev_pinned_free()
 * macro, defined in alloc.h
 */
void dev_pinned_free_(const char *func, const char *file, int line, void *ptr) {
  /*if (!comm_peer2peer_present()) {
    dev_free_(func, file, line, ptr);
    return;
  }*/

  if (!ptr) {
    printfCULQCD("ERROR: Attempt to free NULL device pointer (%s:%d in %s())\n",
                 file, line, func);
    errorCULQCD("Aborting");
  }
  if (!alloc[DEVICE_PINNED_PTR].count(ptr)) {
    printfCULQCD(
        "ERROR: Attempt to free invalid device pointer (%s:%d in %s())\n", file,
        line, func);
    errorCULQCD("Aborting");
  }
  CUresult err = cuMemFree((CUdeviceptr)ptr);
  if (err != CUDA_SUCCESS) {
    printfCULQCD("ERROR: Failed to free device memory (%s:%d in %s())\n", file,
                 line, func);
    errorCULQCD("Aborting");
  }
  track_free(DEVICE_PINNED_PTR, ptr);
}

/**
 * Free host memory allocated with safe_malloc(), pinned_malloc(),
 * or mapped_malloc().  This function should only be called via the
 * host_free() macro, defined in alloc.h
 */
void host_free_(const char *func, const char *file, int line, void *ptr) {
  if (!ptr) {
    printfCULQCD("ERROR: Attempt to free NULL host pointer (%s:%d in %s())\n",
                 file, line, func);
    errorCULQCD("Aborting");
  }
  if (alloc[HOST_PTR].count(ptr)) {
    track_free(HOST_PTR, ptr);
    free(ptr);
  } else if (alloc[PINNED_PTR].count(ptr)) {
    cudaError_t err = cudaHostUnregister(ptr);
    if (err != cudaSuccess) {
      printfCULQCD(
          "ERROR: Failed to unregister pinned memory (%s:%d in %s())\n", file,
          line, func);
      errorCULQCD("Aborting");
    }
    track_free(PINNED_PTR, ptr);
    free(ptr);
  } else if (alloc[MAPPED_PTR].count(ptr)) {
#ifdef HOST_ALLOC
    cudaError_t err = cudaFreeHost(ptr);
    if (err != cudaSuccess) {
      printfCULQCD("ERROR: Failed to free host memory (%s:%d in %s())\n", file,
                   line, func);
      errorCULQCD("Aborting");
    }
#else
    cudaError_t err = cudaHostUnregister(ptr);
    if (err != cudaSuccess) {
      printfCULQCD(
          "ERROR: Failed to unregister host-mapped memory (%s:%d in %s())\n",
          file, line, func);
      errorCULQCD("Aborting");
    }
    free(ptr);
#endif
    track_free(MAPPED_PTR, ptr);
  } else {
    printfCULQCD(
        "ERROR: Attempt to free invalid host pointer (%s:%d in %s())\n", file,
        line, func);
    print_trace();
    errorCULQCD("Aborting");
  }
}

void printPeakMemUsage() {
  printfCULQCD("----------------------------------------------------------\n");
  printfCULQCD("Device memory used = %.1f MB\n",
               max_total_bytes[DEVICE_PTR] / (double)(1 << 20));
  printfCULQCD("Managed memory used = %.1f MB\n",
               max_total_bytes[MANAGED_PTR] / (double)(1 << 20));
  printfCULQCD("Pinned device memory used = %.1f MB\n",
               max_total_bytes[DEVICE_PINNED_PTR] / (double)(1 << 20));
  printfCULQCD("Page-locked host memory used = %.1f MB\n",
               max_total_pinned_bytes / (double)(1 << 20));
  printfCULQCD("Total host memory used >= %.1f MB\n",
               max_total_host_bytes / (double)(1 << 20));
  printfCULQCD("----------------------------------------------------------\n");
}

void assertAllMemFree() {
  if (!alloc[DEVICE_PTR].empty() || !alloc[DEVICE_PINNED_PTR].empty() ||
      !alloc[HOST_PTR].empty() || !alloc[PINNED_PTR].empty() ||
      !alloc[MAPPED_PTR].empty() || !alloc[MANAGED_PTR].empty()) {
    printfCULQCD("The following internal memory allocations were not freed.");
    printfCULQCD("\n");
    print_alloc_header();
    print_alloc(DEVICE_PTR);
    print_alloc(MANAGED_PTR);
    print_alloc(DEVICE_PINNED_PTR);
    print_alloc(HOST_PTR);
    print_alloc(PINNED_PTR);
    print_alloc(MAPPED_PTR);
    printfCULQCD("\n");
  } else
    printfCULQCD("All Memory Free!\n");
}

FieldLocation get_pointer_location(const void *ptr) {

  CUpointer_attribute attribute[] = {CU_POINTER_ATTRIBUTE_MEMORY_TYPE};
  CUmemorytype mem_type;
  void *data[] = {&mem_type};
  CUresult error = cuPointerGetAttributes(1, attribute, data,
                                          reinterpret_cast<CUdeviceptr>(ptr));
  if (error != CUDA_SUCCESS) {
    const char *string;
    cuGetErrorString(error, &string);
    errorCULQCD("cuPointerGetAttributes failed with error %s", string);
  }

  // catch pointers that have not been created in CUDA
  if (mem_type == 0)
    mem_type = CU_MEMORYTYPE_HOST;

  switch (mem_type) {
  case CU_MEMORYTYPE_DEVICE:
  case CU_MEMORYTYPE_UNIFIED:
    return CUDA_FIELD_LOCATION;
  case CU_MEMORYTYPE_HOST:
    return CPU_FIELD_LOCATION;
  default:
    errorCULQCD("Unknown memory type %d", mem_type);
    return INVALID_FIELD_LOCATION;
  }
}

static void FreeMemoryType(AllocType type) {
  const char *type_str[] = {"Device", "Device Pinned", "Host",
                            "Pinned", "Mapped",        "Managed"};
  std::map<void *, MemAlloc>::iterator entry;
  std::map<void *, MemAlloc>::iterator next_entry;

  for (entry = alloc[type].begin(), next_entry = entry;
       entry != alloc[type].end(); entry = next_entry) {
    void *ptr = entry->first;
    MemAlloc a = entry->second;
    printfCULQCD("%s  %15p  %15lu  %s(), %s:%d\n", type_str[type], ptr,
                 (unsigned long)a.base_size, a.func.c_str(), a.file.c_str(),
                 a.line);
    next_entry++;
    switch (type) {
    case DEVICE_PTR:
      dev_free_(a.func.c_str(), a.file.c_str(), a.line, ptr);
      break;
    case MANAGED_PTR:
      managed_free_(a.func.c_str(), a.file.c_str(), a.line, ptr);
      break;
    case DEVICE_PINNED_PTR:
      dev_pinned_free_(a.func.c_str(), a.file.c_str(), a.line, ptr);
      break;
    case HOST_PTR:
      host_free_(a.func.c_str(), a.file.c_str(), a.line, ptr);
      break;
    case PINNED_PTR:
      host_free_(a.func.c_str(), a.file.c_str(), a.line, ptr);
      break;
    case MAPPED_PTR:
      host_free_(a.func.c_str(), a.file.c_str(), a.line, ptr);
      break;
    }
  }
}

void FreeAllMemory() {
  if (!alloc[DEVICE_PTR].empty()) {
    printfCULQCD(
        "Releasing DEVICE internal memory allocations not freed by user:\n");
    print_alloc_header();
    FreeMemoryType(DEVICE_PTR);
    printfCULQCD("\n");
  }
  if (!alloc[MANAGED_PTR].empty()) {
    printfCULQCD(
        "Releasing MANAGED internal memory allocations not freed by user:\n");
    print_alloc_header();
    FreeMemoryType(MANAGED_PTR);
    printfCULQCD("\n");
  }
  if (!alloc[DEVICE_PINNED_PTR].empty()) {
    printfCULQCD("Releasing DEVICE_PINNED internal memory allocations not "
                 "freed by user:\n");
    print_alloc_header();
    FreeMemoryType(DEVICE_PINNED_PTR);
    printfCULQCD("\n");
  }
  if (!alloc[PINNED_PTR].empty()) {
    printfCULQCD(
        "Releasing PINNED internal memory allocations not freed by user:\n");
    print_alloc_header();
    FreeMemoryType(PINNED_PTR);
    printfCULQCD("\n");
  }
  if (!alloc[MAPPED_PTR].empty()) {
    printfCULQCD(
        "Releasing MAPPED internal memory allocations not freed by user:\n");
    print_alloc_header();
    FreeMemoryType(MAPPED_PTR);
    printfCULQCD("\n");
  }
  if (!alloc[HOST_PTR].empty()) {
    printfCULQCD(
        "Releasing HOST internal memory allocations not freed by user:\n");
    print_alloc_header();
    FreeMemoryType(HOST_PTR);
    printfCULQCD("\n");
  }
  assertAllMemFree();
}
} // namespace CULQCD
