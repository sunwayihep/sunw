
#ifndef MODES_H
#define MODES_H


#include <climits>

namespace CULQCD{

typedef enum ReadMode_S {
    Host,
    Device
} ReadMode;

typedef enum CopyMode_s {
    HostToHost,
    HostToDevice,
    DeviceToHost,
    DeviceToDevice
} CopyMode;

typedef enum ArrayType_s {
    SOA = 0,
    SOA12 = 1,
    SOA12A = 2,
    SOA8 = 3
} ArrayType;


typedef enum Verbosity_s {
 SILENT,
 SUMMARIZE,
 VERBOSE,
 DEBUG_VERBOSE
} Verbosity;


typedef enum TuneMode_S {
    TUNE_NO,
    TUNE_YES
} TuneMode;


typedef enum OrderMode_S {
    NORMAL,
    EVEN_ODD
} OrderMode;



typedef enum HaloMode_S {
    NO_HALOS,
    HALOS
} HaloMode;


#define CULQCD_INVALID_ENUM INT_MIN

typedef enum MemoryType_s {
CULQCD_MEMORY_DEVICE,
CULQCD_MEMORY_PINNED,
CULQCD_MEMORY_MAPPED,
CULQCD_MEMORY_INVALID = CULQCD_INVALID_ENUM
} MemoryType;


// Where the field is stored
typedef enum FieldLocation_s {
CPU_FIELD_LOCATION = 1,
CUDA_FIELD_LOCATION = 2,
INVALID_FIELD_LOCATION = CULQCD_INVALID_ENUM
} FieldLocation;



}
#endif
