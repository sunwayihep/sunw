
#ifndef CULQCD_H
#define CULQCD_H

#include <cuda_common.h> /*CUDA Macros*/
#include <complex.h>     /* complex numbers and related operations*/
#include <matrixsun.h>   /*SU(N) matrix and related operations*/
#include <gaugearray.h>  /* gauge field container*/
#include <random.h>      /* CURAND RNG container and rng initialization*/
#include <constants.h>   /*Host and Device constants*/
#include <devicemem.h>   /*Device memory details*/
#include <reunitarize.h> /*reunitarize gauge field*/
#include <texture_host.h>
#include <alloc.h>
#include <exchange.h>
#include <timer.h>
#include <comm_mpi.h> /*MPI setup...*/
#include <io_gauge.h> /* Read and save gauge configurations */

#include <gaugefix/gaugefix.h> /*Gauge fixing, Coulomb and Landau*/
#include <meas/plaquette.h>
#include <meas/linktrsum.h>  /*mean link trace*/
#include <meas/linkdetsum.h> /*mean link determinant*/
#include <meas/linkUF.h>
#include <monte/monte.h>
#include <monte/ovr.h>
#include <meas/polyakovloop.h>
#include <meas/pl.h>
#include <meas/wilsonloop.h>
#include <smear/smear.h>

#include <reunitlink.h>

#include <meas/multilevel.h>
#include <meas/chromofield.h>
#include <meas/wloopex.h>

#endif
