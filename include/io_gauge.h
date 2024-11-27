
#ifndef IO_GAUGE_H
#define IO_GAUGE_H

#include <gaugearray.h>
#include <string>

namespace CULQCD {

template <class Real> void Save_Gauge(gauge arrayin, std::string filename);

template <class Real> void Read_Gauge(gauge arrayin, std::string filename);

template <class Real, class RealSaveConf>
void SaveBin_Gauge(gauge arrayin, std::string filename,
                   bool withheader = false);

template <class Real, class RealSaveConf>
void ReadBin_Gauge(gauge arrayin, std::string filename,
                   bool withheader = false);

// ONLY SINGLE GPU THIS FUNCTION!!!!!!!
template <class Real> void Read_GaugeOLD(gauge arrayin, std::string filename);
} // namespace CULQCD

#endif
