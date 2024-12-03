## Introduction

This package aims to SU(N) lattice gauge theory simulation in arbitrary space-time dimensions.

Supported features:
- **Gauge action**: Wilson
- **Gauge configuration generation algorithm**: [pseudo-heatbath](https://www.sciencedirect.com/science/article/abs/pii/0370269382906967)
- **Number of colors**: N=2,3,4,...
- **Space-time dimensions**: 2,3,4,...

## Quick start

First, setup the C++ and CUDA compiler environment, and then modify the parameters for the number of colors and space-time dimensions in the `Makefile`, then

```sh
make cleanall
make
```

Ensure that you run `make cleanall` each time you modify the `USE_NUMBER_OF_COLORS` and `USE_NUMBER_OF_DIMS` variables in `Makefile`.

The generated executable will be named `heatbath_su3_nd4` for a 4D SU(3) lattice gauge theory. To generate gauge configurations with lattice size $L^3\times L$, we can run

```sh
./heatbath_su3_nd4 12 12 12 24 6.0 10
```

where **6.0** is the coupling $\beta$ in the gauge action, **10** refers to the number of gauge configurations being generated.

## Development

Contributions to the codebase are very welcome.
Before committing, ensure the code is formatted using `clang-format-all.sh`. It is recommended to use a version of `clang-format` that is 18 or higher.

## Acknowledgement

This package utilizes core functionalities from the [sun](https://github.com/nmrcardoso/sun) codebase.