
#ifndef GAUGEARRAY_H
#define GAUGEARRAY_H

//#include <stdio.h>
//#include <string.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <sstream>
#include <cstdlib>
#include <string>
#include <iomanip>

#include <cuda_common.h>
#include <complex.h>
#include <matrixsun.h>
#include <reconstruct_12p_8p.h>
#include <constants.h>
#include <modes.h>
#include <index.h>
#include <comm_mpi.h>
#include <random.h>
#include <devicemem.h>

#include <alloc.h>
#include <cuda_error_check.h>

using namespace std;

namespace CULQCD {

/*!
  \brief convert to string
*/
template <class Real> string ToString(Real number) {
  stringstream ss; // create a stringstream
  ss << number; // add number to the stream
  return ss.str(); // return a string with the contents of the stream
}

inline cudaMemcpyKind copytype(ReadMode cp0, ReadMode cp1) {
  cudaMemcpyKind cptype;
  switch (cp0) {
  case Host:
    switch (cp1) {
    case Host:
      cptype = cudaMemcpyHostToHost;
      break;
    case Device:
      cptype = cudaMemcpyHostToDevice;
      break;
    }
    break;
  case Device:
    switch (cp1) {
    case Host:
      cptype = cudaMemcpyDeviceToHost;
      break;
    case Device:
      cptype = cudaMemcpyDeviceToDevice;
      break;
    }
    break;
  }
  return cptype;
}

/**
    @brief Container to store the gauge field as well as the some other
   parameters like even/odd store, Host or Device allocation...
*/
template <class Real> class _gauge {
private:
  /** @brief In SU(3) there are 3 options SOA(18 real parameters)/SOA12(12 real
   * parameters)/SOA8(8 real parameters) */
  ArrayType atype;
  /** @brief Host/Device memory array allocation*/
  ReadMode mode;
  /** @brief If array is stored in Even/Odd order*/
  bool evenodd;
  /** @brief Number of links */
  int size;
  /** @brief array container for the gauge field*/
  complex *array;
  bool border;

  complex *backup;
  ReadMode backup_mode;

  complex *tempcopy;
  ReadMode tempcopy_mode;

public:
  _gauge(void) {
    size = 0;
    evenodd = false;
    atype = SOA;
    array = NULL;
    backup = NULL;
    tempcopy = NULL;
    border = false;
  }
  _gauge(ArrayType typein, ReadMode modein, bool evenoddarray = false,
         bool borderin = false) {
    mode = modein;
    atype = typein;
    size = 0;
    evenodd = evenoddarray;
    array = NULL;
    backup = NULL;
    tempcopy = NULL;
    border = borderin;
#if (NCOLORS > 3)
    if (atype == SOA12 || atype == SOA12A || atype == SOA8) {
      if (atype == SOA12)
        cout << "<SOA12> ";
      if (atype == SOA12A)
        cout << "<SOA12A> ";
      if (atype == SOA8)
        cout << "<SOA8> ";
      cout << "This option is only for SU(3)" << endl;
      cout << "Set option to <SOA> type" << endl;
      atype = SOA;
      // exit(0);
    }
#endif
  }
  _gauge(ArrayType typein, ReadMode modein, int sizein,
         bool evenoddarray = false, bool borderin = false) {
    mode = modein;
    atype = typein;
    // The size is set only in Allocate(int) function
    size = 0;
    evenodd = evenoddarray;
    array = NULL;
    backup = NULL;
    tempcopy = NULL;
    border = borderin;
#if (NCOLORS > 3)
    if (atype == SOA12 || atype == SOA12A || atype == SOA8) {
      if (atype == SOA12)
        cout << "<SOA12> ";
      if (atype == SOA12A)
        cout << "<SOA12A> ";
      if (atype == SOA8)
        cout << "<SOA8> ";
      cout << "This option is only for SU(3)" << endl;
      cout << "Set option to <SOA> type" << endl;
      atype = SOA;
      // exit(0);
    }
#endif
    Allocate(sizein, border);
  }

  /*! @brief Return the number of SU(Nc) complex parameters used to store in
   * memory*/
  unsigned int getNumElems() const {
    unsigned int num = NCOLORS * NCOLORS;
    switch (atype) {
    case SOA:
      num = NCOLORS * NCOLORS;
      break;
    case SOA12:
      num = 6;
      break;
    case SOA12A:
      num = 6;
      break;
    case SOA8:
      num = 4;
      break;
    }
    return num;
  }
  /*! @brief Return the number of SU(Nc) real parameters used to store in
   * memory*/
  unsigned int getNumParams() const {
    unsigned int num = NCOLORS * NCOLORS * 2;
    switch (atype) {
    case SOA:
      num = NCOLORS * NCOLORS * 2;
      break;
    case SOA12:
      num = 12;
      break;
    case SOA12A:
      num = 12;
      break;
    case SOA8:
      num = 8;
      break;
    }
    return num;
  }
  /**
      @brief Return the number of floating-point operations in the case of
     reconstruction
      for read and write operations
  */
  unsigned int getNumFlop(bool read) const {
    unsigned int num = NCOLORS * NCOLORS;
    switch (atype) {
    case SOA:
      num = 0;
      break;
    case SOA12:
      if (read)
        num = 42;
      else
        num = 0;
      break;
    case SOA12A:
      num = 0;
      break;
    case SOA8:
      if (read)
        num = 102;
      else
        num = 2;
      break;
    }
    return num;
  }
  /*! @brief Set array type (SOA/SOA12/SOA8), mode(Host/Device) and store
   * order*/
  void Set(ArrayType typein, ReadMode modein, bool evenoddarray = false) {
    mode = modein;
    atype = typein;
    size = 0;
    evenodd = evenoddarray;
#if (NCOLORS > 3)
    if (atype == SOA12 || atype == SOA12A || atype == SOA8) {
      if (atype == SOA12)
        cout << "<SOA12> ";
      if (atype == SOA12A)
        cout << "<SOA12A> ";
      if (atype == SOA8)
        cout << "<SOA8> ";
      cout << "This option is only for SU(3)" << endl;
      cout << "Set option to <SOA> type" << endl;
      atype = SOA;
      // exit(0);
    }
#endif
  }
  /*! @brief Prints gauge array details*/
  void Details();
  /*! @brief Retrieve/Set array type, SOA/SOA12/SOA8*/
  M_HOSTDEVICE ArrayType Type() const {
    return atype;
  };
  /*! @brief Retrieve/Set where array is allocated, Host/Device*/
  M_HOSTDEVICE ReadMode Mode() const { return mode; }
  /*! @brief Retrieve/Set if the array are stored in even/odd way*/
  M_HOSTDEVICE bool EvenOdd() const { return evenodd; }
  /*! @brief Retrieve/Set the array size. Careful when using as set after array
   * allocation*/
  M_HOSTDEVICE int Size() const { return size; }
  /*! @brief Retrieve the array memory pointer*/
  M_HOSTDEVICE complex *GetPtr() { return array; }

  /*! @brief Set array type, SOA/SOA12/SOA8*/
  void SetType(ArrayType typein) { atype = typein; }
  /*! @brief Set where array is allocated, Host/Device*/
  void SetMode(ReadMode modein) { mode = modein; }
  /*! @brief Retrieve the array memory pointer*/
  complex *Get() { return array; }
  /*! @brief Retrieve a SU(Nc) matrix from array in position k*/
  M_HOSTDEVICE msun Get(int k);
  /*! @brief Set a new array pointer*/
  void Set(complex *newarray) { array = newarray; }
  /*! @brief Set a new array pointer*/
  void SetPtr(complex *newarray) { array = newarray; }
  /*! @brief Set SU(Nc) matrix in array position k*/
  M_HOSTDEVICE void Set(msun A, int k);
  /*! @brief Allocate array */
  void Allocate(int sizein, bool borderin = false);
  /*! @brief Free memory allocated */
  void Release();
  /*! @brief set allocated memory to zero */
  void Clean();
  bool Border() { return border; }
  // void Copy(_gauge<Real> &from_gauge);
  void Copy(_gauge<Real> &from_gauge);

  /*! @brief Backup gauge array. To use only in tune */
  void Backup() {
    printfCULQCD("Backup gauge array...\n");
    if (backup != NULL) {
      if (backup_mode == Device)
        dev_free(backup);
      else
        host_free(backup);
      backup = NULL;
    }
    if (mode == Device) {
      size_t mfree, mtotal;
      int gpuid = -1;
      cudaSafeCall(cudaGetDevice(&gpuid));
      cudaSafeCall(cudaMemGetInfo(&mfree, &mtotal));
      printfCULQCD("Device memory free: %.2f MB of %.2f MB.\n",
                   mfree / (float)(1048576), mtotal / (float)(1048576));
      printfCULQCD("Memory size required to backup array: %.2f MB.\n",
                   Bytes() / (float)(1048576));

      if (mfree > Bytes()) {
        printfCULQCD("Backup array in Device...\n");
        backup_mode = Device;
        backup = (complex *)dev_malloc(Bytes());
      } else {
        printfCULQCD("Backup array in Host...\n");
        backup_mode = Host;
        backup = (complex *)safe_malloc(Bytes());
      }
    } else {
      backup_mode = Host;
      backup = (complex *)safe_malloc(Bytes());
    }
    cudaMemcpyKind cptype = copytype(mode, backup_mode);
    cudaSafeCall(cudaMemcpy(backup, array, Bytes(), cptype));
  }

  /*! @brief Restore the backup gauge array. To use only in tune */
  void Restore() {
    if (backup != NULL) {
      printfCULQCD("Restore gauge array and release allocated memory...\n");
      cudaMemcpyKind cptype = copytype(backup_mode, mode);
      cudaSafeCall(cudaMemcpy(array, backup, Bytes(), cptype));
      if (backup_mode == Device)
        dev_free(backup);
      else
        host_free(backup);
      backup = NULL;
    } else {
      errorCULQCD("Restoring a gauge array without made a prior Backup...\n");
    }
  }

  /*! @brief Backup gauge array to host memory*/
  void BackupCopyToHost() {
    printfCULQCD("Backup gauge array to Host memory...\n");
    if (tempcopy != NULL) {
      if (tempcopy_mode == Device)
        dev_free(tempcopy);
      else
        host_free(tempcopy);
      tempcopy = NULL;
    }
    tempcopy_mode = Host;
    tempcopy = (complex *)safe_malloc(Bytes());
    cudaMemcpyKind cptype = copytype(mode, tempcopy_mode);
    cudaSafeCall(cudaMemcpy(tempcopy, array, Bytes(), cptype));
  }

  /*! @brief Backup gauge array. */
  void BackupCopy() {
    printfCULQCD("Backup gauge array...\n");
    if (tempcopy != NULL) {
      if (tempcopy_mode == Device)
        dev_free(tempcopy);
      else
        host_free(tempcopy);
      tempcopy = NULL;
    }
    if (mode == Device) {
      size_t mfree, mtotal;
      int gpuid = -1;
      cudaSafeCall(cudaGetDevice(&gpuid));
      cudaSafeCall(cudaMemGetInfo(&mfree, &mtotal));
      printfCULQCD("Device memory free: %.2f MB of %.2f MB.\n",
                   mfree / (float)(1048576), mtotal / (float)(1048576));
      printfCULQCD("Memory size required to backup array: %.2f MB.\n",
                   Bytes() / (float)(1048576));

      if (mfree > Bytes()) {
        printfCULQCD("Backup array in Device...\n");
        tempcopy_mode = Device;
        tempcopy = (complex *)dev_malloc(Bytes());
      } else {
        printfCULQCD("Backup array in Host...\n");
        tempcopy_mode = Host;
        tempcopy = (complex *)safe_malloc(Bytes());
      }
    } else {
      tempcopy_mode = Host;
      tempcopy = (complex *)safe_malloc(Bytes());
    }
    cudaMemcpyKind cptype = copytype(mode, tempcopy_mode);
    cudaSafeCall(cudaMemcpy(tempcopy, array, Bytes(), cptype));
  }

  /*! @brief Restore the backup gauge array. To use only in tune */
  void RestoreCopy() {
    if (tempcopy != NULL) {
      printfCULQCD("Restore gauge array and release allocated memory...\n");
      cudaMemcpyKind cptype = copytype(tempcopy_mode, mode);
      cudaSafeCall(cudaMemcpy(array, tempcopy, Bytes(), cptype));
      if (tempcopy_mode == Device)
        dev_free(tempcopy);
      else
        host_free(tempcopy);
      tempcopy = NULL;
    } else {
      errorCULQCD("Restoring a gauge array without made a prior Backup...\n");
    }
  }

  string ToStringArrayType() const {
    string namearraytype = "";
    switch (atype) {
    case SOA:
      namearraytype = "SOA";
      break;
    case SOA12:
      namearraytype = "SOA12";
      break;
    case SOA12A:
      namearraytype = "SOA12A";
      break;
    case SOA8:
      namearraytype = "SOA8";
      break;
    }
    if (evenodd)
      namearraytype += "_EO";
    return namearraytype;
  }

  /*! @brief Initialize array with a COLD start, identity SU(Nc) matrix*/
  void Init();
  /*! @brief Initialize array with a HOT start, random SU(Nc) matrix*/
  void Init(RNG randstates);

  float GetMemMB() const {
    float arraysize;
    switch (atype) {
    case SOA:
      arraysize = NCOLORS * NCOLORS * size * sizeof(complex) / (float)(1048576);
      break;
    case SOA12:
      arraysize = 6 * size * sizeof(complex) / (float)(1048576);
      break;
    case SOA12A:
      arraysize = 6 * size * sizeof(complex) / (float)(1048576);
      break;
    case SOA8:
      arraysize = 4 * size * sizeof(complex) / (float)(1048576);
      break;
    }
    return arraysize;
  }
  size_t Bytes() const {
    size_t arraysize;
    switch (atype) {
    case SOA:
      arraysize = NCOLORS * NCOLORS * size * sizeof(complex);
      break;
    case SOA12:
      arraysize = 6 * size * sizeof(complex);
      break;
    case SOA12A:
      arraysize = 6 * size * sizeof(complex);
      break;
    case SOA8:
      arraysize = 4 * size * sizeof(complex);
      break;
    }
    return arraysize;
  }
};

#define gauge _gauge<Real>
typedef _gauge<float> gauges;
typedef _gauge<double> gauged;

void GaugeCopy(_gauge<double> arrayin, _gauge<float> &arrayout);
void GaugeCopy(_gauge<float> arrayin, _gauge<double> &arrayout);

/**
   @brief Prints gauge field details, e.g., size in memory, Host/Device
   allocation.
*/
template <class Real> void _gauge<Real>::Details() {
  string smode = "";
  switch (mode) {
  case Host:
    smode = "Host";
    break;
  case Device:
    smode = "Device";
    break;
  }
  string arrayorder = "normal lattice order";
  if (EvenOdd())
    arrayorder = "even/odd lattice order";
  float arraysize;
  switch (atype) {
  case SOA:
    arraysize = NCOLORS * NCOLORS * size * sizeof(complex) / (float)(1048576);
    COUT << "Array stored in " << arrayorder << " allocated in the <" << smode
         << "> with type: <SOA> and size: <" << arraysize << " MB>."
         << std::endl;
    break;
  case SOA12:
    arraysize = 6 * size * sizeof(complex) / (float)(1048576);
    COUT << "Array stored in " << arrayorder << " allocated in the <" << smode
         << "> with type: <SOA12> and size: <" << arraysize << " MB>."
         << std::endl;
    break;
  case SOA12A:
    arraysize = 6 * size * sizeof(complex) / (float)(1048576);
    COUT << "Array stored in " << arrayorder << " allocated in the <" << smode
         << "> with type: <SOA12A> and size: <" << arraysize << " MB>."
         << std::endl;
    break;
  case SOA8:
    arraysize = 4 * size * sizeof(complex) / (float)(1048576);
    COUT << "Array stored in " << arrayorder << " allocated in the <" << smode
         << "> with type: <SOA8> and size: <" << arraysize << " MB>."
         << std::endl;
    break;
  }
}

/**
    @brief Retrieve a SU(Nc) matrix in a given lattice site.
    @param k lattice site.
    @return Returns a SU(Nc) matrix.
*/
template <class Real> M_HOSTDEVICE msun _gauge<Real>::Get(int k) {
  msun m;
  switch (atype) {
  case SOA:
    for (int i = 0; i < NCOLORS; i++)
      for (int j = 0; j < NCOLORS; j++)
        m.e[i][j] = array[k + (j + i * NCOLORS) * size];
    break;
  case SOA12:
    for (int i = 0; i < NCOLORS - 1; i++)
      for (int j = 0; j < NCOLORS; j++)
        m.e[i][j] = array[k + (j + i * NCOLORS) * size];
    m.e[2][0] = ~(m.e[0][1] * m.e[1][2] - m.e[0][2] * m.e[1][1]);
    m.e[2][1] = ~(m.e[0][2] * m.e[1][0] - m.e[0][0] * m.e[1][2]);
    m.e[2][2] = ~(m.e[0][0] * m.e[1][1] - m.e[0][1] * m.e[1][0]);
    break;
  case SOA12A:
    m.e[0][0] = array[k];
    m.e[0][1] = array[k + size];
    m.e[0][2] = array[k + 2 * size];
    m.e[1][1] = array[k + 3 * size];
    m.e[1][2] = array[k + 4 * size];
    m.e[2][2] = array[k + 5 * size];
    m.e[1][0] = complex::make_complex(-m.e[0][1].real(), m.e[0][1].imag());
    m.e[2][0] = complex::make_complex(-m.e[0][2].real(), m.e[0][2].imag());
    m.e[2][1] = complex::make_complex(-m.e[1][2].real(), m.e[1][2].imag());
    break;
  case SOA8:
    m.e[0][1] = array[k];
    m.e[0][2] = array[k + size];
    m.e[1][0] = array[k + 2 * size];
    complex theta = array[k + 3 * size];
    reconstruct8p<Real>(m, theta);
    break;
  }
  return m;
}

/**
    @brief Set a SU(Nc) matrix in a given lattice site.
    @param A SU(Nc) matrix.
    @param k lattice site.
*/
template <class Real> M_HOSTDEVICE void _gauge<Real>::Set(msun A, int k) {
  switch (atype) {
  case SOA:
    for (int i = 0; i < NCOLORS; i++)
      for (int j = 0; j < NCOLORS; j++)
        array[k + (j + i * NCOLORS) * size] = A.e[i][j];
    break;
  case SOA12:
    for (int i = 0; i < NCOLORS - 1; i++)
      for (int j = 0; j < NCOLORS; j++)
        array[k + (j + i * NCOLORS) * size] = A.e[i][j];
    break;
  case SOA12A:
    array[k] = A.e[0][0];
    array[k + size] = A.e[0][1];
    array[k + 2 * size] = A.e[0][2];
    array[k + 3 * size] = A.e[1][1];
    array[k + 4 * size] = A.e[1][2];
    array[k + 5 * size] = A.e[2][2];
    break;
  case SOA8:
    array[k] = A.e[0][1];
    complex theta;
    theta.real() = A.e[0][0].phase();
    theta.imag() = A.e[2][0].phase();
    array[k + size] = A.e[0][2];
    array[k + 2 * size] = A.e[1][0];
    array[k + 3 * size] = theta;
    break;
  }
}

/**
    @brief Allocate gauge field.
    @param sizein number of gauge field links.
*/
template <class Real> void _gauge<Real>::Allocate(int sizein, bool borderin) {
  if (size == 0 && sizein > 0) {
    size = sizein;
    border = borderin;
    string namearraytype = "";
    size_t arraysize;
    switch (atype) {
    case SOA:
      arraysize = size * NCOLORS * NCOLORS * sizeof(complex);
      namearraytype = "SOA";
      break;
    case SOA12:
      arraysize = size * 6 * sizeof(complex);
      namearraytype = "SOA12";
      break;
    case SOA12A:
      arraysize = size * 6 * sizeof(complex);
      namearraytype = "SOA12A";
      break;
    case SOA8:
      arraysize = size * 4 * sizeof(complex);
      namearraytype = "SOA8";
      break;
    }
    switch (mode) {
    case Host:
      array = (complex *)safe_malloc(arraysize);
      // array =(complex *) malloc( arraysize );
      if (array == NULL) {
        printf("ERROR: Can't malloc array\n");
        exit(0);
      }
      memset(array, 0, arraysize);
      // COUT << "Allocated <" << namearraytype << "> array in the HOST with
      // size: " << arraysize/(float)(1048576) << " MB" << std::endl;
      break;
    case Device:
      array = (complex *)dev_malloc(arraysize);
      // cudaSafeCall(cudaMalloc((void **)&array, arraysize));
      // cudaSafeCall(cudaMemset( array , 0 , arraysize ));
      // COUT << "Allocated <" << namearraytype << "> array in the DEVICE with
      // size: " << arraysize/(float)(1048576) << " MB" << std::endl;
      break;
    }
  } else
    COUT << "Array already Allocated!!!!" << std::endl;
}

/**
    @brief Release allocate memory.
*/
template <class Real> void _gauge<Real>::Release() {
  if (size > 0) {
    switch (mode) {
    case Host:
      host_free(array);
      // COUT << "Free <" << namearraytype << "> array in the HOST with size: "
      // << arraysize/(float)(1048576) << " MB" << std::endl;
      break;
    case Device:
      // CUDA_SAFE_DEVICE_SYNC( );
      dev_free(array);
      // cudaSafeCall(cudaFree(array));
      // COUT << "Free <" << namearraytype << "> array in the DEVICE with size:
      // " << arraysize/(float)(1048576) << " MB" << std::endl;
      break;
    }
    size = 0;
    array = NULL;
  } else if (size == 0)
    COUT << "Array not yet allocated!!!" << std::endl;

  if (backup != NULL) {
    if (backup_mode == Device)
      dev_free(backup);
    else
      host_free(backup);
  }
  backup = NULL;

  if (tempcopy != NULL) {
    if (tempcopy_mode == Device)
      dev_free(tempcopy);
    else
      host_free(tempcopy);
  }
  tempcopy = NULL;
}

/**
    @brief Clean all values in allocated memory.
*/
template <class Real> void _gauge<Real>::Clean() {
  if (size > 0) {
    string namearraytype = "";
    size_t arraysize;
    switch (atype) {
    case SOA:
      arraysize = size * NCOLORS * NCOLORS * sizeof(complex);
      namearraytype = "SOA";
      break;
    case SOA12:
      arraysize = size * 6 * sizeof(complex);
      namearraytype = "SOA12";
      break;
    case SOA12A:
      arraysize = size * 6 * sizeof(complex);
      namearraytype = "SOA12A";
      break;
    case SOA8:
      arraysize = size * 4 * sizeof(complex);
      namearraytype = "SOA8";
      break;
    }
    switch (mode) {
    case Host:
      memset(array, 0, arraysize);
      if (getVerbosity() >= DEBUG_VERBOSE)
        COUT << "Clean <" << namearraytype << "> array in the HOST with size: "
             << arraysize / (float)(1048576) << " MB" << std::endl;
      break;
    case Device:
      cudaSafeCall(cudaMemset(array, 0, arraysize));
      if (getVerbosity() >= DEBUG_VERBOSE)
        COUT << "Clean <" << namearraytype
             << "> array in the DEVICE with size: "
             << arraysize / (float)(1048576) << " MB" << std::endl;
      break;
    }
  } else
    COUT << "Empty array, cannot clean..." << std::endl;
}

/**
   @brief Initialize gauge field with a COLD start
   @param array gauge field
*/
template <class Real> void ColdStart(gauge array);

/**
   @brief Initialize gauge field with a HOT start
   @param array gauge field
   @param randstates current state of RNG
*/
template <class Real> void HotStart(gauge array, RNG randstates);

template <class Real> void HotStart00(gauge array, RNG randstates);

/**
    @brief Initialize gauge field with a COLD start.
*/
template <class Real> void _gauge<Real>::Init() {
  switch (mode) {
  case Host:
    for (int i = 0; i < size; i++)
      Set(msun::identity(), i);
    break;
  case Device:
    ColdStart<Real>(*this);
    break;
  }
}
/**
    @brief Initialize gauge field with a HOT start.
    @param randstates current CUDA RNG state array.
*/
template <class Real> void _gauge<Real>::Init(RNG randstates) {
  switch (mode) {
  case Host:
    cout << "Not yet implemented.\nUsing cold start." << endl;
    Init();
    break;
  case Device:
    HotStart<Real>(*this, randstates);
    break;
  }
}
}

#endif
