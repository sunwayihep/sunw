
#ifndef CULQCD_H
#define CULQCD_H

#include <alloc.h>
#include <comm_mpi.h>    /*MPI setup...*/
#include <complex.h>     /* complex numbers and related operations*/
#include <constants.h>   /*Host and Device constants*/
#include <cuda_common.h> /*CUDA Macros*/
#include <devicemem.h>   /*Device memory details*/
#include <exchange.h>
#include <gaugearray.h>  /* gauge field container*/
#include <io_gauge.h>    /* Read and save gauge configurations */
#include <matrixsun.h>   /*SU(N) matrix and related operations*/
#include <random.h>      /* CURAND RNG container and rng initialization*/
#include <reunitarize.h> /*reunitarize gauge field*/
#include <texture_host.h>
#include <timer.h>

#include <gaugefix/gaugefix.h> /*Gauge fixing, Coulomb and Landau*/
#include <meas/linkUF.h>
#include <meas/linkdetsum.h> /*mean link determinant*/
#include <meas/linktrsum.h>  /*mean link trace*/
#include <meas/pl.h>
#include <meas/plaquette.h>
#include <meas/polyakovloop.h>
#include <meas/wilsonloop.h>
#include <monte/monte.h>
#include <monte/ovr.h>
#include <smear/smear.h>

#include <reunitlink.h>

#include <meas/chromofield.h>
#include <meas/multilevel.h>
#include <meas/wloopex.h>

#endif
