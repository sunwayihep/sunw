
#ifndef GAUGEFIX_OVR_DEVICE_CUH
#define GAUGEFIX_OVR_DEVICE_CUH

template <int DIR, int NTPVOL, class Real>
__forceinline__ __device__ void
GaugeFixHit_AtomicAdd(msun &link, Real relax_boost, int tid) {
  // Container for the four real parameters of SU(2) subgroup in shared memory
  //__shared__ Real elems[NTPVOL * 4];
  Real *elems = SharedMem<Real>();
  // initialize shared memory
  if (threadIdx.x < NTPVOL * 4)
    elems[threadIdx.x] = 0.0;
  __syncthreads();
  // Loop over all SU(2) subroups of SU(N)
  // #pragma unroll
  for (int block = 0; block < TOTAL_SUB_BLOCKS; block++) {
    int p, q;
    // Get the two indices for the SU(N) matrix
    IndexBlock(block, p, q);
    Real asq = 1.0;
    if (threadIdx.x < NTPVOL * 4)
      asq = -1.0;
    // FOR COULOMB AND LANDAU!!!!!!!!
    // if(nu0<DIR){
    // In terms of thread index
    if (threadIdx.x < NTPVOL * DIR ||
        (threadIdx.x >= NTPVOL * 4 && threadIdx.x < NTPVOL * (DIR + 4))) {
      // Retrieve the four SU(2) parameters...
      CudaAtomicAdd(elems + tid,
                    link.e[p][p].real() + link.e[q][q].real()); // a0
      CudaAtomicAdd(elems + tid + NTPVOL,
                    (link.e[p][q].imag() + link.e[q][p].imag()) * asq); // a1
      CudaAtomicAdd(elems + tid + NTPVOL * 2,
                    (link.e[p][q].real() - link.e[q][p].real()) * asq); // a2
      CudaAtomicAdd(elems + tid + NTPVOL * 3,
                    (link.e[p][p].imag() - link.e[q][q].imag()) * asq); // a3
    } // FLOP per lattice site = DIR * 2 * (4 + 7) = DIR * 22
    __syncthreads();
    if (threadIdx.x < NTPVOL) {
      // Over-relaxation boost
      asq = elems[threadIdx.x + NTPVOL] * elems[threadIdx.x + NTPVOL];
      asq += elems[threadIdx.x + NTPVOL * 2] * elems[threadIdx.x + NTPVOL * 2];
      asq += elems[threadIdx.x + NTPVOL * 3] * elems[threadIdx.x + NTPVOL * 3];
      Real a0sq = elems[threadIdx.x] * elems[threadIdx.x];
      Real x = (relax_boost * a0sq + asq) / (a0sq + asq);
      Real r = 1.0 / sqrt((a0sq + x * x * asq));
      elems[threadIdx.x] *= r;
      elems[threadIdx.x + NTPVOL] *= x * r;
      elems[threadIdx.x + NTPVOL * 2] *= x * r;
      elems[threadIdx.x + NTPVOL * 3] *= x * r;
    } // FLOP per lattice site = 22
    __syncthreads();
    //_____________
    if (threadIdx.x < NTPVOL * 4) {
      complex m0;
      // Do SU(2) hit on all upward links
      // left multiply an su3_matrix by an su2 matrix
      // link <- u * link
      // #pragma unroll
      for (int j = 0; j < NCOLORS; j++) {
        m0 = link.e[p][j];
        link.e[p][j] =
            complex::make_complex(elems[tid], elems[tid + NTPVOL * 3]) * m0 +
            complex::make_complex(elems[tid + NTPVOL * 2],
                                  elems[tid + NTPVOL]) *
                link.e[q][j];
        link.e[q][j] =
            complex::make_complex(-elems[tid + NTPVOL * 2],
                                  elems[tid + NTPVOL]) *
                m0 +
            complex::make_complex(elems[tid], -elems[tid + NTPVOL * 3]) *
                link.e[q][j];
      }
    } else {
      complex m0;
      // Do SU(2) hit on all downward links
      // right multiply an su3_matrix by an su2 matrix
      // link <- link * u_adj
      // #pragma unroll
      for (int j = 0; j < NCOLORS; j++) {
        m0 = link.e[j][p];
        link.e[j][p] =
            complex::make_complex(elems[tid], -elems[tid + NTPVOL * 3]) * m0 +
            complex::make_complex(elems[tid + NTPVOL * 2],
                                  -elems[tid + NTPVOL]) *
                link.e[j][q];
        link.e[j][q] =
            complex::make_complex(-elems[tid + NTPVOL * 2],
                                  -elems[tid + NTPVOL]) *
                m0 +
            complex::make_complex(elems[tid], elems[tid + NTPVOL * 3]) *
                link.e[j][q];
      }
    }
    //_____________ //FLOP per lattice site = 8 * NCOLORS * 2 * (2*6+2) =
    //NCOLORS * 224
    if (block < TOTAL_SUB_BLOCKS - 1) {
      __syncthreads();
      // reset shared memory SU(2) elements
      if (threadIdx.x < NTPVOL * 4)
        elems[threadIdx.x] = 0.0;
      __syncthreads();
    }
  } // FLOP per lattice site = (block < NCOLORS * ( NCOLORS - 1) / 2) * (22 + 28
    // DIR + 224 NCOLORS)
  // write updated link to global memory
}

template <int DIR, int NTPVOL, class Real>
__forceinline__ __device__ void
GaugeFixHit_NoAtomicAdd(msun &link, Real relax_boost, int tid) {
  // Container for the four real parameters of SU(2) subgroup in shared memory
  //__shared__ Real elems[NTPVOL * 4 * 8];
  Real *elems = SharedMem<Real>();
  // Loop over all SU(2) subroups of SU(N)
  // #pragma unroll
  for (int block = 0; block < TOTAL_SUB_BLOCKS; block++) {
    int p, q;
    // Get the two indices for the SU(N) matrix
    IndexBlock(block, p, q);
    Real asq = 1.0;
    if (threadIdx.x < NTPVOL * 4)
      asq = -1.0;
    if (threadIdx.x < NTPVOL * DIR ||
        (threadIdx.x >= NTPVOL * 4 && threadIdx.x < NTPVOL * (DIR + 4))) {
      elems[threadIdx.x] = link.e[p][p].real() + link.e[q][q].real();
      elems[threadIdx.x + NTPVOL * 8] =
          (link.e[p][q].imag() + link.e[q][p].imag()) * asq;
      elems[threadIdx.x + NTPVOL * 8 * 2] =
          (link.e[p][q].real() - link.e[q][p].real()) * asq;
      elems[threadIdx.x + NTPVOL * 8 * 3] =
          (link.e[p][p].imag() - link.e[q][q].imag()) * asq;
    } // FLOP per lattice site = DIR * 2 * 7 = DIR * 14
    __syncthreads();
    if (threadIdx.x < NTPVOL) {
      Real a0, a1, a2, a3;
      a0 = 0.0;
      a1 = 0.0;
      a2 = 0.0;
      a3 = 0.0;
#pragma unroll
      for (int i = 0; i < DIR; i++) {
        a0 += elems[tid + i * NTPVOL] + elems[tid + (i + 4) * NTPVOL];
        a1 += elems[tid + i * NTPVOL + NTPVOL * 8] +
              elems[tid + (i + 4) * NTPVOL + NTPVOL * 8];
        a2 += elems[tid + i * NTPVOL + NTPVOL * 8 * 2] +
              elems[tid + (i + 4) * NTPVOL + NTPVOL * 8 * 2];
        a3 += elems[tid + i * NTPVOL + NTPVOL * 8 * 3] +
              elems[tid + (i + 4) * NTPVOL + NTPVOL * 8 * 3];
      }
      // Over-relaxation boost
      asq = a1 * a1 + a2 * a2 + a3 * a3;
      Real a0sq = a0 * a0;
      Real x = (relax_boost * a0sq + asq) / (a0sq + asq);
      Real r = rsqrt((a0sq + x * x * asq));
      elems[threadIdx.x] = a0 * r;
      elems[threadIdx.x + NTPVOL] = a1 * x * r;
      elems[threadIdx.x + NTPVOL * 2] = a2 * x * r;
      elems[threadIdx.x + NTPVOL * 3] = a3 * x * r;
    } // FLOP per lattice site = 22 + 8 * DIR
    __syncthreads();
    //_____________
    if (threadIdx.x < NTPVOL * 4) {
      complex m0;
      // Do SU(2) hit on all upward links
      // left multiply an su3_matrix by an su2 matrix
      // link <- u * link
      // #pragma unroll
      for (int j = 0; j < NCOLORS; j++) {
        m0 = link.e[p][j];
        link.e[p][j] =
            complex::make_complex(elems[tid], elems[tid + NTPVOL * 3]) * m0 +
            complex::make_complex(elems[tid + NTPVOL * 2],
                                  elems[tid + NTPVOL]) *
                link.e[q][j];
        link.e[q][j] =
            complex::make_complex(-elems[tid + NTPVOL * 2],
                                  elems[tid + NTPVOL]) *
                m0 +
            complex::make_complex(elems[tid], -elems[tid + NTPVOL * 3]) *
                link.e[q][j];
      }
    } else {
      complex m0;
      // Do SU(2) hit on all downward links
      // right multiply an su3_matrix by an su2 matrix
      // link <- link * u_adj
      // #pragma unroll
      for (int j = 0; j < NCOLORS; j++) {
        m0 = link.e[j][p];
        link.e[j][p] =
            complex::make_complex(elems[tid], -elems[tid + NTPVOL * 3]) * m0 +
            complex::make_complex(elems[tid + NTPVOL * 2],
                                  -elems[tid + NTPVOL]) *
                link.e[j][q];
        link.e[j][q] =
            complex::make_complex(-elems[tid + NTPVOL * 2],
                                  -elems[tid + NTPVOL]) *
                m0 +
            complex::make_complex(elems[tid], elems[tid + NTPVOL * 3]) *
                link.e[j][q];
      }
    }
    //_____________ //FLOP per lattice site = 8 * NCOLORS * 2 * (2*6+2) =
    //NCOLORS * 224
    if (block < TOTAL_SUB_BLOCKS - 1)
      __syncthreads();
  } // FLOP per lattice site = (block < NCOLORS * ( NCOLORS - 1) / 2) * (22 + 28
    // DIR + 224 NCOLORS)
  // write updated link to global memory
}

template <int DIR, int NTPVOL, class Real>
__forceinline__ __device__ void
GaugeFixHit_NoAtomicAdd_LessSM(msun &link, Real relax_boost, int tid) {
  // Container for the four real parameters of SU(2) subgroup in shared memory
  //__shared__ Real elems[NTPVOL * 4 * 8];
  Real *elems = SharedMem<Real>();
  // Loop over all SU(2) subroups of SU(N)
  // #pragma unroll
  for (int block = 0; block < TOTAL_SUB_BLOCKS; block++) {
    int p, q;
    // Get the two indices for the SU(N) matrix
    IndexBlock(block, p, q);
    if (threadIdx.x < NTPVOL) {
      elems[tid] = link.e[p][p].real() + link.e[q][q].real();
      elems[tid + NTPVOL] = -(link.e[p][q].imag() + link.e[q][p].imag());
      elems[tid + NTPVOL * 2] = -(link.e[p][q].real() - link.e[q][p].real());
      elems[tid + NTPVOL * 3] = -(link.e[p][p].imag() - link.e[q][q].imag());
    }
    __syncthreads();
    if (threadIdx.x < NTPVOL * 2 && threadIdx.x >= NTPVOL) {
      elems[tid] += link.e[p][p].real() + link.e[q][q].real();
      elems[tid + NTPVOL] -= (link.e[p][q].imag() + link.e[q][p].imag());
      elems[tid + NTPVOL * 2] -= (link.e[p][q].real() - link.e[q][p].real());
      elems[tid + NTPVOL * 3] -= (link.e[p][p].imag() - link.e[q][q].imag());
    }
    __syncthreads();
    if (threadIdx.x < NTPVOL * 3 && threadIdx.x >= NTPVOL * 2) {
      elems[tid] += link.e[p][p].real() + link.e[q][q].real();
      elems[tid + NTPVOL] -= (link.e[p][q].imag() + link.e[q][p].imag());
      elems[tid + NTPVOL * 2] -= (link.e[p][q].real() - link.e[q][p].real());
      elems[tid + NTPVOL * 3] -= (link.e[p][p].imag() - link.e[q][q].imag());
    }
    if (DIR == 4) {
      __syncthreads();
      if (threadIdx.x < NTPVOL * 4 && threadIdx.x >= NTPVOL * 3) {
        elems[tid] += link.e[p][p].real() + link.e[q][q].real();
        elems[tid + NTPVOL] -= (link.e[p][q].imag() + link.e[q][p].imag());
        elems[tid + NTPVOL * 2] -= (link.e[p][q].real() - link.e[q][p].real());
        elems[tid + NTPVOL * 3] -= (link.e[p][p].imag() - link.e[q][q].imag());
      }
    }
    __syncthreads();
    if (threadIdx.x < NTPVOL * 5 && threadIdx.x >= NTPVOL * 4) {
      elems[tid] += link.e[p][p].real() + link.e[q][q].real();
      elems[tid + NTPVOL] += (link.e[p][q].imag() + link.e[q][p].imag());
      elems[tid + NTPVOL * 2] += (link.e[p][q].real() - link.e[q][p].real());
      elems[tid + NTPVOL * 3] += (link.e[p][p].imag() - link.e[q][q].imag());
    }
    __syncthreads();
    if (threadIdx.x < NTPVOL * 6 && threadIdx.x >= NTPVOL * 5) {
      elems[tid] += link.e[p][p].real() + link.e[q][q].real();
      elems[tid + NTPVOL] += (link.e[p][q].imag() + link.e[q][p].imag());
      elems[tid + NTPVOL * 2] += (link.e[p][q].real() - link.e[q][p].real());
      elems[tid + NTPVOL * 3] += (link.e[p][p].imag() - link.e[q][q].imag());
    }
    __syncthreads();
    if (threadIdx.x < NTPVOL * 7 && threadIdx.x >= NTPVOL * 6) {
      elems[tid] += link.e[p][p].real() + link.e[q][q].real();
      elems[tid + NTPVOL] += (link.e[p][q].imag() + link.e[q][p].imag());
      elems[tid + NTPVOL * 2] += (link.e[p][q].real() - link.e[q][p].real());
      elems[tid + NTPVOL * 3] += (link.e[p][p].imag() - link.e[q][q].imag());
    }
    if (DIR == 4) {
      __syncthreads();
      if (threadIdx.x < NTPVOL * 8 && threadIdx.x >= NTPVOL * 7) {
        elems[tid] += link.e[p][p].real() + link.e[q][q].real();
        elems[tid + NTPVOL] += (link.e[p][q].imag() + link.e[q][p].imag());
        elems[tid + NTPVOL * 2] += (link.e[p][q].real() - link.e[q][p].real());
        elems[tid + NTPVOL * 3] += (link.e[p][p].imag() - link.e[q][q].imag());
      }
    }
    // FLOP per lattice site = 4 + (2*DIR - 1) * 8 = DIR * 16 - 4
    __syncthreads();
    if (threadIdx.x < NTPVOL) {
      // Over-relaxation boost
      Real asq = elems[threadIdx.x + NTPVOL] * elems[threadIdx.x + NTPVOL];
      asq += elems[threadIdx.x + NTPVOL * 2] * elems[threadIdx.x + NTPVOL * 2];
      asq += elems[threadIdx.x + NTPVOL * 3] * elems[threadIdx.x + NTPVOL * 3];
      Real a0sq = elems[threadIdx.x] * elems[threadIdx.x];
      Real x = (relax_boost * a0sq + asq) / (a0sq + asq);
      Real r = rsqrt((a0sq + x * x * asq));
      elems[threadIdx.x] *= r;
      elems[threadIdx.x + NTPVOL] *= x * r;
      elems[threadIdx.x + NTPVOL * 2] *= x * r;
      elems[threadIdx.x + NTPVOL * 3] *= x * r;
    } // FLOP per lattice site = 22
    __syncthreads();
    //_____________
    if (threadIdx.x < NTPVOL * 4) {
      complex m0;
      // Do SU(2) hit on all upward links
      // left multiply an su3_matrix by an su2 matrix
      // link <- u * link
      // #pragma unroll
      for (int j = 0; j < NCOLORS; j++) {
        m0 = link.e[p][j];
        link.e[p][j] =
            complex::make_complex(elems[tid], elems[tid + NTPVOL * 3]) * m0 +
            complex::make_complex(elems[tid + NTPVOL * 2],
                                  elems[tid + NTPVOL]) *
                link.e[q][j];
        link.e[q][j] =
            complex::make_complex(-elems[tid + NTPVOL * 2],
                                  elems[tid + NTPVOL]) *
                m0 +
            complex::make_complex(elems[tid], -elems[tid + NTPVOL * 3]) *
                link.e[q][j];
      }
    } else {
      complex m0;
      // Do SU(2) hit on all downward links
      // right multiply an su3_matrix by an su2 matrix
      // link <- link * u_adj
      // #pragma unroll
      for (int j = 0; j < NCOLORS; j++) {
        m0 = link.e[j][p];
        link.e[j][p] =
            complex::make_complex(elems[tid], -elems[tid + NTPVOL * 3]) * m0 +
            complex::make_complex(elems[tid + NTPVOL * 2],
                                  -elems[tid + NTPVOL]) *
                link.e[j][q];
        link.e[j][q] =
            complex::make_complex(-elems[tid + NTPVOL * 2],
                                  -elems[tid + NTPVOL]) *
                m0 +
            complex::make_complex(elems[tid], elems[tid + NTPVOL * 3]) *
                link.e[j][q];
      }
    }
    //_____________ //FLOP per lattice site = 8 * NCOLORS * 2 * (2*6+2) =
    //NCOLORS * 224
    if (block < TOTAL_SUB_BLOCKS - 1)
      __syncthreads();
  } // FLOP per lattice site = (block < NCOLORS * ( NCOLORS - 1) / 2) * (22 + 28
    // DIR + 224 NCOLORS)
  // write updated link to global memory
}

template <int DIR, int NTPVOL, class Real>
__forceinline__ __device__ void
GaugeFixHit_AtomicAdd(msun &link, msun &link1, Real relax_boost, int tid) {
  // Container for the four real parameters of SU(2) subgroup in shared memory
  //__shared__ Real elems[NTPVOL * 4];
  Real *elems = SharedMem<Real>();
  // initialize shared memory
  if (threadIdx.x < NTPVOL * 4)
    elems[threadIdx.x] = 0.0;
  __syncthreads();
  // Loop over all SU(2) subroups of SU(N)
  // #pragma unroll
  for (int block = 0; block < TOTAL_SUB_BLOCKS; block++) {
    int p, q;
    // Get the two indices for the SU(N) matrix
    IndexBlock(block, p, q);
    if (threadIdx.x < NTPVOL * DIR) {
      CudaAtomicAdd(elems + tid, link.e[p][p].real() + link.e[q][q].real() +
                                     link1.e[p][p].real() +
                                     link1.e[q][q].real());
      CudaAtomicAdd(elems + tid + NTPVOL,
                    (link1.e[p][q].imag() + link1.e[q][p].imag()) -
                        (link.e[p][q].imag() + link.e[q][p].imag()));
      CudaAtomicAdd(elems + tid + NTPVOL * 2,
                    (link1.e[p][q].real() - link1.e[q][p].real()) -
                        (link.e[p][q].real() - link.e[q][p].real()));
      CudaAtomicAdd(elems + tid + NTPVOL * 3,
                    (link1.e[p][p].imag() - link1.e[q][q].imag()) -
                        (link.e[p][p].imag() - link.e[q][q].imag()));
    } // FLOP per lattice site = DIR * 2 * 7 = DIR * 14
    __syncthreads();
    if (threadIdx.x < NTPVOL) {
      // Over-relaxation boost
      Real asq = elems[threadIdx.x + NTPVOL] * elems[threadIdx.x + NTPVOL];
      asq += elems[threadIdx.x + NTPVOL * 2] * elems[threadIdx.x + NTPVOL * 2];
      asq += elems[threadIdx.x + NTPVOL * 3] * elems[threadIdx.x + NTPVOL * 3];
      Real a0sq = elems[threadIdx.x] * elems[threadIdx.x];
      Real x = (relax_boost * a0sq + asq) / (a0sq + asq);
      Real r = rsqrt((a0sq + x * x * asq));
      elems[threadIdx.x] *= r;
      elems[threadIdx.x + NTPVOL] *= x * r;
      elems[threadIdx.x + NTPVOL * 2] *= x * r;
      elems[threadIdx.x + NTPVOL * 3] *= x * r;
    } // FLOP per lattice site = 22
    __syncthreads();
    complex m0;
    // Do SU(2) hit on all upward links
    // left multiply an su3_matrix by an su2 matrix
    // link <- u * link
    // #pragma unroll
    for (int j = 0; j < NCOLORS; j++) {
      m0 = link.e[p][j];
      link.e[p][j] =
          complex::make_complex(elems[tid], elems[tid + NTPVOL * 3]) * m0 +
          complex::make_complex(elems[tid + NTPVOL * 2], elems[tid + NTPVOL]) *
              link.e[q][j];
      link.e[q][j] =
          complex::make_complex(-elems[tid + NTPVOL * 2], elems[tid + NTPVOL]) *
              m0 +
          complex::make_complex(elems[tid], -elems[tid + NTPVOL * 3]) *
              link.e[q][j];
    }
    // Do SU(2) hit on all downward links
    // right multiply an su3_matrix by an su2 matrix
    // link <- link * u_adj
    // #pragma unroll
    for (int j = 0; j < NCOLORS; j++) {
      m0 = link1.e[j][p];
      link1.e[j][p] =
          complex::make_complex(elems[tid], -elems[tid + NTPVOL * 3]) * m0 +
          complex::make_complex(elems[tid + NTPVOL * 2], -elems[tid + NTPVOL]) *
              link1.e[j][q];
      link1.e[j][q] =
          complex::make_complex(-elems[tid + NTPVOL * 2],
                                -elems[tid + NTPVOL]) *
              m0 +
          complex::make_complex(elems[tid], elems[tid + NTPVOL * 3]) *
              link1.e[j][q];
    }
    //_____________ //FLOP per lattice site = 8 * NCOLORS * 2 * (2*6+2) =
    //NCOLORS * 224
    if (block < TOTAL_SUB_BLOCKS - 1) {
      __syncthreads();
      // reset shared memory SU(2) elements
      if (threadIdx.x < NTPVOL * 4)
        elems[threadIdx.x] = 0.0;
      __syncthreads();
    }
  } // FLOP per lattice site = (block < NCOLORS * ( NCOLORS - 1) / 2) * (22 + 28
    // DIR + 224 NCOLORS)
}

template <int DIR, int NTPVOL, class Real>
__forceinline__ __device__ void
GaugeFixHit_NoAtomicAdd(msun &link, msun &link1, Real relax_boost, int tid) {
  Real *elems = SharedMem<Real>();
  // Loop over all SU(2) subroups of SU(N)
  // #pragma unroll
  for (int block = 0; block < TOTAL_SUB_BLOCKS; block++) {
    int p, q;
    // Get the two indices for the SU(N) matrix
    IndexBlock(block, p, q);
    if (threadIdx.x < NTPVOL * DIR) {
      elems[threadIdx.x] = link.e[p][p].real() + link.e[q][q].real() +
                           link1.e[p][p].real() + link1.e[q][q].real();
      elems[threadIdx.x + NTPVOL * 4] =
          (link1.e[p][q].imag() + link1.e[q][p].imag()) -
          (link.e[p][q].imag() + link.e[q][p].imag());
      elems[threadIdx.x + NTPVOL * 4 * 2] =
          (link1.e[p][q].real() - link1.e[q][p].real()) -
          (link.e[p][q].real() - link.e[q][p].real());
      elems[threadIdx.x + NTPVOL * 4 * 3] =
          (link1.e[p][p].imag() - link1.e[q][q].imag()) -
          (link.e[p][p].imag() - link.e[q][q].imag());
    }
    __syncthreads();
    if (threadIdx.x < NTPVOL) {
      Real a0, a1, a2, a3;
      a0 = 0.0;
      a1 = 0.0;
      a2 = 0.0;
      a3 = 0.0;
#pragma unroll
      for (int i = 0; i < DIR; i++) {
        a0 += elems[tid + i * NTPVOL];
        a1 += elems[tid + i * NTPVOL + NTPVOL * 4];
        a2 += elems[tid + i * NTPVOL + NTPVOL * 4 * 2];
        a3 += elems[tid + i * NTPVOL + NTPVOL * 4 * 3];
      }
      // Over-relaxation boost
      Real asq = a1 * a1 + a2 * a2 + a3 * a3;
      Real a0sq = a0 * a0;
      Real x = (relax_boost * a0sq + asq) / (a0sq + asq);
      Real r = rsqrt((a0sq + x * x * asq));
      elems[threadIdx.x] = a0 * r;
      elems[threadIdx.x + NTPVOL] = a1 * x * r;
      elems[threadIdx.x + NTPVOL * 2] = a2 * x * r;
      elems[threadIdx.x + NTPVOL * 3] = a3 * x * r;
    }
    __syncthreads();
    complex m0;
    // Do SU(2) hit on all upward links
    // left multiply an su3_matrix by an su2 matrix
    // link <- u * link
    // #pragma unroll
    for (int j = 0; j < NCOLORS; j++) {
      m0 = link.e[p][j];
      link.e[p][j] =
          complex::make_complex(elems[tid], elems[tid + NTPVOL * 3]) * m0 +
          complex::make_complex(elems[tid + NTPVOL * 2], elems[tid + NTPVOL]) *
              link.e[q][j];
      link.e[q][j] =
          complex::make_complex(-elems[tid + NTPVOL * 2], elems[tid + NTPVOL]) *
              m0 +
          complex::make_complex(elems[tid], -elems[tid + NTPVOL * 3]) *
              link.e[q][j];
    }
    // Do SU(2) hit on all downward links
    // right multiply an su3_matrix by an su2 matrix
    // link <- link * u_adj
    // #pragma unroll
    for (int j = 0; j < NCOLORS; j++) {
      m0 = link1.e[j][p];
      link1.e[j][p] =
          complex::make_complex(elems[tid], -elems[tid + NTPVOL * 3]) * m0 +
          complex::make_complex(elems[tid + NTPVOL * 2], -elems[tid + NTPVOL]) *
              link1.e[j][q];
      link1.e[j][q] =
          complex::make_complex(-elems[tid + NTPVOL * 2],
                                -elems[tid + NTPVOL]) *
              m0 +
          complex::make_complex(elems[tid], elems[tid + NTPVOL * 3]) *
              link1.e[j][q];
    }
    if (block < TOTAL_SUB_BLOCKS - 1)
      __syncthreads();
  }
}

template <int DIR, int NTPVOL, class Real>
__forceinline__ __device__ void
GaugeFixHit_NoAtomicAdd_LessSM(msun &link, msun &link1, Real relax_boost,
                               int tid) {
  Real *elems = SharedMem<Real>();
  // Loop over all SU(2) subroups of SU(N)
  // #pragma unroll
  for (int block = 0; block < TOTAL_SUB_BLOCKS; block++) {
    int p, q;
    // Get the two indices for the SU(N) matrix
    IndexBlock(block, p, q);
    if (threadIdx.x < NTPVOL) {
      elems[tid] = link.e[p][p].real() + link.e[q][q].real() +
                   link1.e[p][p].real() + link1.e[q][q].real();
      elems[tid + NTPVOL] = (link1.e[p][q].imag() + link1.e[q][p].imag()) -
                            (link.e[p][q].imag() + link.e[q][p].imag());
      elems[tid + NTPVOL * 2] = (link1.e[p][q].real() - link1.e[q][p].real()) -
                                (link.e[p][q].real() - link.e[q][p].real());
      elems[tid + NTPVOL * 3] = (link1.e[p][p].imag() - link1.e[q][q].imag()) -
                                (link.e[p][p].imag() - link.e[q][q].imag());
    }
    __syncthreads();
    if (threadIdx.x < NTPVOL * 2 && threadIdx.x >= NTPVOL) {
      elems[tid] += link.e[p][p].real() + link.e[q][q].real() +
                    link1.e[p][p].real() + link1.e[q][q].real();
      elems[tid + NTPVOL] += (link1.e[p][q].imag() + link1.e[q][p].imag()) -
                             (link.e[p][q].imag() + link.e[q][p].imag());
      elems[tid + NTPVOL * 2] += (link1.e[p][q].real() - link1.e[q][p].real()) -
                                 (link.e[p][q].real() - link.e[q][p].real());
      elems[tid + NTPVOL * 3] += (link1.e[p][p].imag() - link1.e[q][q].imag()) -
                                 (link.e[p][p].imag() - link.e[q][q].imag());
    }
    __syncthreads();
    if (threadIdx.x < NTPVOL * 3 && threadIdx.x >= NTPVOL * 2) {
      elems[tid] += link.e[p][p].real() + link.e[q][q].real() +
                    link1.e[p][p].real() + link1.e[q][q].real();
      elems[tid + NTPVOL] += (link1.e[p][q].imag() + link1.e[q][p].imag()) -
                             (link.e[p][q].imag() + link.e[q][p].imag());
      elems[tid + NTPVOL * 2] += (link1.e[p][q].real() - link1.e[q][p].real()) -
                                 (link.e[p][q].real() - link.e[q][p].real());
      elems[tid + NTPVOL * 3] += (link1.e[p][p].imag() - link1.e[q][q].imag()) -
                                 (link.e[p][p].imag() - link.e[q][q].imag());
    }
    if (DIR == 4) {
      __syncthreads();
      if (threadIdx.x < NTPVOL * 4 && threadIdx.x >= NTPVOL * 3) {
        elems[tid] += link.e[p][p].real() + link.e[q][q].real() +
                      link1.e[p][p].real() + link1.e[q][q].real();
        elems[tid + NTPVOL] += (link1.e[p][q].imag() + link1.e[q][p].imag()) -
                               (link.e[p][q].imag() + link.e[q][p].imag());
        elems[tid + NTPVOL * 2] +=
            (link1.e[p][q].real() - link1.e[q][p].real()) -
            (link.e[p][q].real() - link.e[q][p].real());
        elems[tid + NTPVOL * 3] +=
            (link1.e[p][p].imag() - link1.e[q][q].imag()) -
            (link.e[p][p].imag() - link.e[q][q].imag());
      }
    }
    __syncthreads();
    if (threadIdx.x < NTPVOL) {
      // Over-relaxation boost
      Real asq = elems[threadIdx.x + NTPVOL] * elems[threadIdx.x + NTPVOL];
      asq += elems[threadIdx.x + NTPVOL * 2] * elems[threadIdx.x + NTPVOL * 2];
      asq += elems[threadIdx.x + NTPVOL * 3] * elems[threadIdx.x + NTPVOL * 3];
      Real a0sq = elems[threadIdx.x] * elems[threadIdx.x];
      Real x = (relax_boost * a0sq + asq) / (a0sq + asq);
      Real r = rsqrt((a0sq + x * x * asq));
      elems[threadIdx.x] *= r;
      elems[threadIdx.x + NTPVOL] *= x * r;
      elems[threadIdx.x + NTPVOL * 2] *= x * r;
      elems[threadIdx.x + NTPVOL * 3] *= x * r;
    } // FLOP per lattice site = 22
    __syncthreads();
    complex m0;
    // Do SU(2) hit on all upward links
    // left multiply an su3_matrix by an su2 matrix
    // link <- u * link
    // #pragma unroll
    for (int j = 0; j < NCOLORS; j++) {
      m0 = link.e[p][j];
      link.e[p][j] =
          complex::make_complex(elems[tid], elems[tid + NTPVOL * 3]) * m0 +
          complex::make_complex(elems[tid + NTPVOL * 2], elems[tid + NTPVOL]) *
              link.e[q][j];
      link.e[q][j] =
          complex::make_complex(-elems[tid + NTPVOL * 2], elems[tid + NTPVOL]) *
              m0 +
          complex::make_complex(elems[tid], -elems[tid + NTPVOL * 3]) *
              link.e[q][j];
    }
    // Do SU(2) hit on all downward links
    // right multiply an su3_matrix by an su2 matrix
    // link <- link * u_adj
    // #pragma unroll
    for (int j = 0; j < NCOLORS; j++) {
      m0 = link1.e[j][p];
      link1.e[j][p] =
          complex::make_complex(elems[tid], -elems[tid + NTPVOL * 3]) * m0 +
          complex::make_complex(elems[tid + NTPVOL * 2], -elems[tid + NTPVOL]) *
              link1.e[j][q];
      link1.e[j][q] =
          complex::make_complex(-elems[tid + NTPVOL * 2],
                                -elems[tid + NTPVOL]) *
              m0 +
          complex::make_complex(elems[tid], elems[tid + NTPVOL * 3]) *
              link1.e[j][q];
    }
    if (block < TOTAL_SUB_BLOCKS - 1)
      __syncthreads();
  }
}

#endif
