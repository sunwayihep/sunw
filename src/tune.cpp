

// #include <stdio.h>
// #include <string.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include <cfloat> // for FLT_MAX
#include <ctime>
#include <fcntl.h>
#include <fstream>
#include <map>
#include <sys/stat.h> // for stat()
#include <typeinfo>
#include <unistd.h>

#include <comm_mpi.h>
#include <complex.h>
#include <constants.h>
#include <cuda_common.h>
#include <index.h>
#include <matrixsun.h>
#include <modes.h>
#include <random.h>
#include <reconstruct_12p_8p.h>

#include <alloc.h>
#include <comm_mpi.h>
#include <tune.h>

#include <culqcd_version.h>

using namespace std;

namespace CULQCD {

std::string GetCurrentWorkingDir() {
  std::string cwd("\0", FILENAME_MAX + 1);
  return getcwd(&cwd[0], cwd.capacity());
}

// CODE FROM QUDA LIBRARY WITH A FEW MODIFICATIONS

typedef std::map<TuneKey, TuneParam> map;

static const std::string CULQCD_hash = CULQCD_HASH; // defined in lib/Makefile
static std::string resource_path;
static map tunecache;
static map::iterator it;
static size_t initial_cache_size = 0;

#define STR_(x) #x
#define STR(x) STR_(x)
static const std::string CULQCD_version = STR(CULQCD_VERSION_MAJOR) "." STR(
    CULQCD_VERSION_MINOR) "." STR(CULQCD_VERSION_SUBMINOR);
#undef STR
#undef STR_

/**
 * Deserialize tunecache from an istream, useful for reading a file or receiving
 * from other nodes.
 */
static void deserializeTuneCache(std::istream &in) {
  std::string line;
  std::stringstream ls;

  TuneKey key;
  TuneParam param;

  std::string v;
  std::string n;
  std::string t;
  std::string a;

  while (in.good()) {
    getline(in, line);
    if (!line.length())
      continue; // skip blank lines (e.g., at end of file)
    ls.clear();
    ls.str(line);
    ls >> v >> n >> t >> a >> param.block.x >> param.block.y >> param.block.z;
    sprintf(key.volume, "%s", v.c_str());
    sprintf(key.name, "%s", n.c_str());
    sprintf(key.atype, "%s", t.c_str());
    sprintf(key.aux, "%s", a.c_str());
    ls >> param.grid.x >> param.grid.y >> param.grid.z >> param.shared_bytes;
    ls.ignore(1);               // throw away tab before comment
    getline(ls, param.comment); // assume anything remaining on the line is a
                                // comment
    param.comment += "\n"; // our convention is to include the newline, since
                           // ctime() likes to do this
    tunecache[key] = param;
  }
}

/**
 * Serialize tunecache to an ostream, useful for writing to a file or sending to
 * other nodes.
 */
static void serializeTuneCache(std::ostream &out) {
  map::iterator entry;

  for (entry = tunecache.begin(); entry != tunecache.end(); entry++) {
    TuneKey key = entry->first;
    TuneParam param = entry->second;

    out << key.volume << "\t" << key.name << "\t" << key.atype << "\t"
        << key.aux << "\t";
    out << param.block.x << "\t" << param.block.y << "\t" << param.block.z
        << "\t";
    out << param.grid.x << "\t" << param.grid.y << "\t" << param.grid.z << "\t";
    out << param.shared_bytes << "\t"
        << param.comment; // param.comment ends with a newline
  }
}

/**
 * Distribute the tunecache from node 0 to all other nodes.
 */
static void broadcastTuneCache() {
#ifdef MULTI_GPU

  std::stringstream serialized;
  size_t size;

  if (comm_rank() == 0) {
    serializeTuneCache(serialized);
    size = serialized.str().length();
  }

  comm_broadcast(&size, sizeof(size_t));

  if (size > 0) {
    if (comm_rank() == 0) {
      comm_broadcast(const_cast<char *>(serialized.str().c_str()), size);
    } else {
      char *serstr = new char[size + 1];
      comm_broadcast(serstr, size);
      serstr[size] = '\0'; // null-terminate
      serialized.str(serstr);
      deserializeTuneCache(serialized);
      delete[] serstr;
    }
  }
#endif
}

/*
 * Read tunecache from disk.
 */
void loadTuneCache(Verbosity verbosity) {
  char *path;
  struct stat pstat;
  std::string cache_path, line, token;
  std::ifstream cache_file;
  std::stringstream ls;

  path = getenv("CULQCD_RESOURCE_PATH");
  if (!path) {
    printfCULQCD(
        "warning: Environment variable CULQCD_RESOURCE_PATH is not set.\n");
    // printfCULQCD("warning: Caching of tuned parameters will be disabled.");
    resource_path = GetCurrentWorkingDir();
    printfCULQCD("warning: using default file in current directory: %s\n",
                 resource_path.c_str());
    // return;
  } else if (stat(path, &pstat) || !S_ISDIR(pstat.st_mode)) {
    printfCULQCD("warning: The path \"%s\" specified by CULQCD_RESOURCE_PATH "
                 "does not exist or is not a directory.\n",
                 path);
    // printfCULQCD("warning: Caching of tuned parameters will be disabled.");
    resource_path = GetCurrentWorkingDir();
    printfCULQCD("warning: using default file in current directory: %s\n",
                 resource_path.c_str());
    // return;
  } else {
    resource_path = path;
  }

#ifdef MULTI_GPU
  if (mynode() == 0)
#endif
  {
    cache_path = resource_path;
    cache_path += "/tunecache.csv";
    cache_file.open(cache_path.c_str());

    if (cache_file) {

      if (!cache_file.good())
        errorCULQCD("Bad format in %s", cache_path.c_str());
      getline(cache_file, line);
      ls.str(line);
      ls >> token;
      if (token.compare("tunecache"))
        errorCULQCD("Bad format in %s", cache_path.c_str());
      ls >> token;
      if (token.compare(CULQCD_version))
        errorCULQCD("Cache file %s does not match current CULQCD version. "
                    "\nPlease delete this file or set the CULQCD_RESOURCE_PATH "
                    "environment variable to point to a new path.",
                    cache_path.c_str());
      ls >> token;
      if (token.compare(CULQCD_hash))
        errorCULQCD("Cache file %s does not match current CULQCD build. "
                    "\nPlease delete this file or set the CULQCD_RESOURCE_PATH "
                    "environment variable to point to a new path.",
                    cache_path.c_str());

      if (!cache_file.good())
        errorCULQCD("Bad format in %s", cache_path.c_str());
      getline(cache_file, line); // eat the blank line

      if (!cache_file.good())
        errorCULQCD("Bad format in %s", cache_path.c_str());
      getline(cache_file, line); // eat the description line

      deserializeTuneCache(cache_file);

      cache_file.close();
      initial_cache_size = tunecache.size();

      if (verbosity >= SUMMARIZE) {
        printfCULQCD("Loaded %d sets of cached parameters from %s\n",
                     static_cast<int>(initial_cache_size), cache_path.c_str());
      }

    } else {
      printfCULQCD("warning: Cache file not found.  All kernels will be "
                   "re-tuned (if tuning is enabled).");
    }

    // #ifdef MULTI_GPU
  }
  // #endif

  broadcastTuneCache();
}

/**
 * Write tunecache to disk.
 */
void saveTuneCache(Verbosity verbosity) {
  time_t now;
  int lock_handle;
  std::string lock_path, cache_path;
  std::ofstream cache_file;

  if (resource_path.empty())
    return;

    // FIXME: We should really check to see if any nodes have tuned a kernel
    // that was not also tuned on node 0, since as things
    //       stand, the corresponding launch parameters would never get cached
    //       to
    // disk in this situation.  This will come up if we
    //       ever support different subvolumes per GPU (as might be convenient
    //       for
    // lattice volumes that don't divide evenly).

#ifdef MULTI_GPU
  if (mynode() == 0)
#endif
  {

    if (tunecache.size() == initial_cache_size)
      return;

    // Acquire lock.  Note that this is only robust if the filesystem supports
    // flock() semantics, which is true for
    // NFS on recent versions of linux but not Lustre by default (unless the
    // filesystem was mounted with "-o flock").
    lock_path = resource_path + "/tunecache.lock";
    lock_handle = open(lock_path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0666);
    if (lock_handle == -1) {
      printfCULQCD("warning: Unable to lock cache file.  Tuned launch "
                   "parameters will not be cached to disk.  "
                   "If you are certain that no other instances of CULQCD are "
                   "accessing this filesystem, "
                   "please manually remove %s",
                   lock_path.c_str());
      return;
    }
    char msg[] = "If no instances of applications using CULQCD are running,\n"
                 "this lock file shouldn't be here and is safe to delete.";
    int stat = write(lock_handle, msg,
                     sizeof(msg)); // check status to avoid compiler warning
    if (stat == -1)
      printfCULQCD(
          "warning: Unable to write to lock file for some bizarre reason");

    cache_path = resource_path + "/tunecache.csv";
    cache_file.open(cache_path.c_str());

    if (verbosity >= SUMMARIZE) {
      printfCULQCD("Saving %d sets of cached parameters to %s\n",
                   static_cast<int>(tunecache.size()), cache_path.c_str());
    }

    time(&now);
    cache_file << "tunecache\t" << CULQCD_version << "\t" << CULQCD_hash
               << "\t# Last updated " << ctime(&now) << std::endl;
    cache_file << "volume\tname\tarray_type\taux\tblock.x\tblock.y\tblock."
                  "z\tgrid.x\tgrid.y\tgrid.z\tshared_bytes\tcomment"
               << std::endl;
    serializeTuneCache(cache_file);
    cache_file.close();

    // Release lock.
    close(lock_handle);
    remove(lock_path.c_str());

    initial_cache_size = tunecache.size();

    // #ifdef MULTI_GPU
  }
  // #endif
}

// static TimeProfile launchTimer("tuneLaunch");

//  static int tally = 0;

/**
 * Return the optimal launch parameters for a given kernel, either
 * by retrieving them from tunecache or autotuning on the spot.
 */
TuneParam &tuneLaunch(Tunable &tunable, TuneMode enabled, Verbosity verbosity) {

  const TuneKey key = tunable.tuneKey();
  static TuneParam param;
  // first check if we have the tuned value and return if we have it
  it = tunecache.find(key);
  if (enabled == TUNE_YES && it != tunecache.end()) {
    TuneParam param = it->second;
    tunable.checkLaunchParam(it->second);
    return it->second;
  }
  // We must switch off the global sum when tuning in case of process divergence
  // bool reduceState = globalReduce;
  bool reduceState = true;
  // globalReduce = false;

  static bool tuning = false;           // tuning in progress?
  static const Tunable *active_tunable; // for error checking
  if (enabled == TUNE_NO) {
    tunable.defaultTuneParam(param);
    tunable.checkLaunchParam(param);
  } else if (!tuning) {
    TuneParam best_param;
    cudaError_t error;
    cudaEvent_t start, end;
    float elapsed_time, best_time;
    time_t now;

    tuning = true;
    active_tunable = &tunable;
    best_time = FLT_MAX;

    if (verbosity >= DEBUG_VERBOSE)
      printfCULQCD("PreTune %s\n", key.name);
    tunable.preTune();

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    if (verbosity >= DEBUG_VERBOSE) {
      printfCULQCD("Tuning %s with %s at vol=%s for array_type %s\n", key.name,
                   key.aux, key.volume, key.atype);
    }

    tunable.initTuneParam(param);
    while (tuning) {
      cudaDeviceSynchronize();
      cudaGetLastError(); // clear error counter
      tunable.checkLaunchParam(param);
      cudaEventRecord(start, 0);
      for (int i = 0; i < tunable.tuningIter(); i++) {
        if (verbosity >= DEBUG_VERBOSE) {
          printfCULQCD("About to call tunable.apply\n");
        }
        tunable.apply(0); // calls tuneLaunch() again, which simply returns the
                          // currently active param
      }
      cudaEventRecord(end, 0);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&elapsed_time, start, end);
      cudaDeviceSynchronize();
      error = cudaGetLastError();

      { // check that error state is cleared
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
          errorCULQCD("Failed to clear error state %s\n",
                      cudaGetErrorString(error));
      }

      elapsed_time /= (1e3 * tunable.tuningIter());
      if ((elapsed_time < best_time) && (error == cudaSuccess)) {
        best_time = elapsed_time;
        best_param = param;
      }
      if ((verbosity >= DEBUG_VERBOSE)) {
        if (error == cudaSuccess) {
          printfCULQCD("    %s gives %s\n", tunable.paramString(param).c_str(),
                       tunable.perfString(elapsed_time).c_str());
        } else {
          printfCULQCD("    %s gives %s\n", tunable.paramString(param).c_str(),
                       cudaGetErrorString(error));
        }
      }
      tuning = tunable.advanceTuneParam(param);
    }

    if (best_time == FLT_MAX) {
      errorCULQCD(
          "Auto-tuning failed for %s with %s at vol=%s for array_type %s\n",
          key.name, key.aux, key.volume, key.atype);
    }
    if (verbosity >= VERBOSE) {
      printfCULQCD("Tuned %s giving %s for %s with %s for array_type %s\n",
                   tunable.paramString(best_param).c_str(),
                   tunable.perfString(best_time).c_str(), key.name, key.aux,
                   key.atype);
    }
    time(&now);
    best_param.comment = "# " + tunable.perfString(best_time) + ", tuned ";
    best_param.comment += ctime(&now); // includes a newline

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    if (verbosity >= DEBUG_VERBOSE)
      printfCULQCD("PostTune %s\n", key.name);
    tunable.postTune();
    param = best_param;
    tunecache[key] = best_param;

  } else if (&tunable != active_tunable) {
    errorCULQCD("Unexpected call to tuneLaunch() in %s::apply()",
                typeid(tunable).name());
  }
  // restore the original reduction state
  // globalReduce = reduceState;
  return param;
}
} // namespace CULQCD
