
#ifndef EXCHANGE_H
#define EXCHANGE_H


#include <gaugearray.h>
#include <constants.h>



namespace CULQCD{
#ifdef MULTI_GPU





template<class Real>
void AllocateTempBuffersAndStreams();
void FreeTempBuffersAndStreams();



template<class Real>
void  Exchange_gauge_border_links_gauge(gauge _pgauge, int dir, int parity, bool all_radius_border=false);



template<class Real>
void  Exchange_gauge_topborder_links_gauge(gauge _pgauge, int dir, int parity, bool all_radius_border=false);



template<class Real>
void  EndExchange_gauge_fix_links_gauge(gauge _pgauge, int parity);

template<class Real>
void  StartExchange_gauge_fix_links_gauge(gauge _pgauge, int parity);


//G(x) array
template<class Real>
void  Exchange_gauge_border_links(gauge _pgauge, int dir, int parity, bool all_radius_border);

#endif
}

#endif
