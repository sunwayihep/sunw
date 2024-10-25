//////////////////////////////////////////////////////////////////////////////
// Timer.cpp
// =========
// High Resolution Timer.
// This timer is able to measure the elapsed time with 1 micro-second accuracy
// in both Windows, Linux and Unix system
//
//  AUTHOR: Song Ho Ahn (song.ahn@gmail.com)
// CREATED: 2003-01-13
// UPDATED: 2006-01-13
//
// Copyright (c) 2003 Song Ho Ahn
//////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#ifdef MULTI_GPU
#include <mpi.h>
#endif
#include <timer.h>

/*! \brief Get current date/time, format is YYYY-MM-DD.HH:mm:ss */
const std::string currentDateTime() {
  time_t now = time(0);
  struct tm tstruct;
  char buf[80];
  tstruct = *localtime(&now);
  strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

  return buf;
}

/*! \brief constructor*/
Timer::Timer() {

#ifdef MULTI_GPU
  startTimeInSec = 0;
  endTimeInSec = 0;
#elif defined(WIN32)
  QueryPerformanceFrequency(&frequency);
  startCount.QuadPart = 0;
  endCount.QuadPart = 0;
  startTimeInMicroSec = 0;
  endTimeInMicroSec = 0;
#else
  startCount.tv_sec = startCount.tv_usec = 0;
  endCount.tv_sec = endCount.tv_usec = 0;
  startTimeInMicroSec = 0;
  endTimeInMicroSec = 0;
#endif

  stopped = 0;
}

/*! \brief destructor*/
Timer::~Timer() {}

/*! \brief start timer. @a startCount will be set at this point.*/
void Timer::start() {
  stopped = 0; // reset stop flag
#ifdef MULTI_GPU
  MPI_Barrier(MPI_COMM_WORLD);
  startTimeInSec = MPI_Wtime();
#elif defined(WIN32)
  QueryPerformanceCounter(&startCount);
#else
  gettimeofday(&startCount, NULL);
#endif
}

/*! \brief stop the timer. @a endCount will be set at this point. */
void Timer::stop() {
  stopped = 1; // set timer stopped flag

#ifdef MULTI_GPU
  MPI_Barrier(MPI_COMM_WORLD);
  endTimeInSec = MPI_Wtime();
#elif defined(WIN32)
  QueryPerformanceCounter(&endCount);
#else
  gettimeofday(&endCount, NULL);
#endif
}

/*! \brief compute elapsed time in micro-second resolution. other @a
 * getElapsedTime will call this first, then convert to correspond resolution.
 */
double Timer::getElapsedTimeInMicroSec() {
#ifdef MULTI_GPU
  return this->getElapsedTimeInSec() * 1000000.0;
#elif defined(WIN32)
  if (!stopped)
    QueryPerformanceCounter(&endCount);

  startTimeInMicroSec = startCount.QuadPart * (1000000.0 / frequency.QuadPart);
  endTimeInMicroSec = endCount.QuadPart * (1000000.0 / frequency.QuadPart);
  return endTimeInMicroSec - startTimeInMicroSec;
#else
  if (!stopped)
    gettimeofday(&endCount, NULL);

  startTimeInMicroSec = (startCount.tv_sec * 1000000.0) + startCount.tv_usec;
  endTimeInMicroSec = (endCount.tv_sec * 1000000.0) + endCount.tv_usec;
  return endTimeInMicroSec - startTimeInMicroSec;
#endif
}

/*! \brief divide @a elapsedTimeInMicroSec by 1000 */
double Timer::getElapsedTimeInMilliSec() {
#ifdef MULTI_GPU
  return this->getElapsedTimeInSec() * 1000.0;
#else
  return this->getElapsedTimeInMicroSec() / 1000;
#endif
}

/*! \brief divide @a elapsedTimeInMicroSec by 1000000 */
double Timer::getElapsedTimeInSec() {
#ifdef MULTI_GPU
  if (!stopped)
    endTimeInSec = MPI_Wtime();
  return endTimeInSec - startTimeInSec;
#else
  return this->getElapsedTimeInMicroSec() * 0.000001;
#endif
}

/*! \brief same as @a getElapsedTimeInSec() */
double Timer::getElapsedTime() { return this->getElapsedTimeInSec(); }
