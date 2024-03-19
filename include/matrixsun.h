
#ifndef MATRIXSUN_H
#define MATRIXSUN_H


#include <stdio.h>
#include <string.h>
#include <iostream>
#include <iomanip>


#include <complex.h>

#include <cuda_common.h>


namespace CULQCD{


#ifndef NCOLORS
#warning Number of colors not defined! -> Using default value: 3
#define NCOLORS 3
#endif
#ifndef TOTAL_SUB_BLOCKS
#define TOTAL_SUB_BLOCKS ((NCOLORS) * ( (NCOLORS) - 1) / 2)
#endif


/**
@brief Holds SU(Nc) matrix and many SU(Nc) matrix operations
*/
template <class Real,  int Ncolors>
  class _matrixsun {
 public:
  //-------------------------------------------------------------------------------
  // data members
  _complex<Real> e[Ncolors][Ncolors];   
  //-------------------------------------------------------------------------------	
  /*M_HOSTDEVICE _matrixsun(){
    #pragma unroll 
    for(int j = 0; j < Ncolors; j++)
    #pragma unroll 
    for(int i = 0; i < Ncolors; i++)
    e[i][j] = (Real)0.0;

    }*/
  //-------------------------------------------------------------------------------
  M_HOSTDEVICE _complex<Real>& operator()(const int i, const int j){
	return e[i][j];
  }
  //-------------------------------------------------------------------------------
  M_HOSTDEVICE _complex<Real> operator()(const int i, const int j) const{
	return e[i][j];
  }
  //-------------------------------------------------------------------------------
  M_HOSTDEVICE _matrixsun<Real,Ncolors> operator=(const _matrixsun<Real,Ncolors> REF(A))  {
	if(Ncolors == 3){
		e[0][0] = A.e[0][0]; e[0][1] = A.e[0][1]; e[0][2] = A.e[0][2];
		e[1][0] = A.e[1][0]; e[1][1] = A.e[1][1]; e[1][2] = A.e[1][2];
		e[2][0] = A.e[2][0]; e[2][1] = A.e[2][1]; e[2][2] = A.e[2][2];
	}
	else{
		#pragma unroll 
		for(int i = 0; i < Ncolors; i++)
		#pragma unroll 
		for(int j = 0; j < Ncolors; j++)
			e[i][j] = A.e[i][j];
	}
	return *this;
  }
  //-------------------------------------------------------------------------------
  // SUM's	
  M_HOSTDEVICE _matrixsun<Real,Ncolors> operator+(const _matrixsun<Real,Ncolors> REF(A)) const {
    _matrixsun<Real,Ncolors> res;
	if(Ncolors == 3){
		res.e[0][0] = e[0][0] + A.e[0][0]; 
		res.e[0][1] = e[0][1] + A.e[0][1]; 
		res.e[0][2] = e[0][2] + A.e[0][2];
		res.e[1][0] = e[1][0] + A.e[1][0]; 
		res.e[1][1] = e[1][1] + A.e[1][1]; 
		res.e[1][2] = e[1][2] + A.e[1][2];
		res.e[2][0] = e[2][0] + A.e[2][0]; 
		res.e[2][1] = e[2][1] + A.e[2][1]; 
		res.e[2][2] = e[2][2] + A.e[2][2];
	}
	else{
		#pragma unroll 
		for(int i = 0; i < Ncolors; i++)
		#pragma unroll 
		for(int j = 0; j < Ncolors; j++)
			res.e[i][j] = e[i][j] + A.e[i][j];
	}
	return res;
  }	
  M_HOSTDEVICE _matrixsun<Real,Ncolors> operator+=(const _matrixsun<Real,Ncolors> REF(A))  {
	if(Ncolors == 3){
		e[0][0] += A.e[0][0]; e[0][1] += A.e[0][1]; e[0][2] += A.e[0][2];
		e[1][0] += A.e[1][0]; e[1][1] += A.e[1][1]; e[1][2] += A.e[1][2];
		e[2][0] += A.e[2][0]; e[2][1] += A.e[2][1]; e[2][2] += A.e[2][2];
	}
	else{	
		#pragma unroll 
		for(int i = 0; i < Ncolors; i++)
		#pragma unroll 
		for(int j = 0; j < Ncolors; j++)
			e[i][j] += A.e[i][j];
	}
	return *this;
  }
  M_HOSTDEVICE _matrixsun<Real,Ncolors> operator+(const complex REF(A)) const {
    _matrixsun<Real,Ncolors> res;
	if(Ncolors == 3){
		res.e[0][0] = e[0][0] + A; res.e[0][1] = e[0][1] + A; res.e[0][2] = e[0][2] + A;
		res.e[1][0] = e[1][0] + A; res.e[1][1] = e[1][1] + A; res.e[1][2] = e[1][2] + A;
		res.e[2][0] = e[2][0] + A, res.e[2][1] = e[2][1] + A; res.e[2][2] = e[2][2] + A;
	}
	else{
		#pragma unroll 
		for(int i = 0; i < Ncolors; i++)
		#pragma unroll 
		for(int j = 0; j < Ncolors; j++)
			res.e[i][j] = e[i][j] + A;
	}
	return res;
  }	
  M_HOSTDEVICE _matrixsun<Real,Ncolors> operator+=(const complex REF(A))  {
	if(Ncolors == 3){
		e[0][0] += A; e[0][1] += A; e[0][2] += A;
		e[1][0] += A; e[1][1] += A; e[1][2] += A;
		e[2][0] += A, e[2][1] += A; e[2][2] += A;
	}
	else{
		#pragma unroll 
		for(int i = 0; i < Ncolors; i++)
		#pragma unroll 
		for(int j = 0; j < Ncolors; j++)
			e[i][j] += A;
	}
	return *this;
  }
  M_HOSTDEVICE _matrixsun<Real,Ncolors> operator+(const Real REF(A)) const {
    _matrixsun<Real,Ncolors> res;
	if(Ncolors == 3){
		res.e[0][0] = e[0][0] + A; res.e[0][1] = e[0][1] + A; res.e[0][2] = e[0][2] + A;
		res.e[1][0] = e[1][0] + A; res.e[1][1] = e[1][1] + A; res.e[1][2] = e[1][2] + A;
		res.e[2][0] = e[2][0] + A, res.e[2][1] = e[2][1] + A; res.e[2][2] = e[2][2] + A;
	}
	else{
		#pragma unroll 
		for(int i = 0; i < Ncolors; i++)
		#pragma unroll 
		for(int j = 0; j < Ncolors; j++)
			res.e[i][j] = e[i][j] + A;
	}
	return res;
  }	
  M_HOSTDEVICE _matrixsun<Real,Ncolors> operator+=(const Real REF(A))  {
	if(Ncolors == 3){
		e[0][0] += A; e[0][1] += A; e[0][2] += A;
		e[1][0] += A; e[1][1] += A; e[1][2] += A;
		e[2][0] += A, e[2][1] += A; e[2][2] += A;
	}
	else{
		#pragma unroll 
		for(int i = 0; i < Ncolors; i++)
		#pragma unroll 
		for(int j = 0; j < Ncolors; j++)
			e[i][j] += A;
	}
	return *this;
  }
  //-------------------------------------------------------------------------------
  // DIF's	
  M_HOSTDEVICE _matrixsun<Real,Ncolors> operator-(const _matrixsun<Real,Ncolors> REF(A)) const {
    _matrixsun<Real,Ncolors> res;
	if(Ncolors == 3){
		res.e[0][0] = e[0][0] - A.e[0][0];
		res.e[0][1] = e[0][1] - A.e[0][1]; 
		res.e[0][2] = e[0][2] - A.e[0][2];
		res.e[1][0] = e[1][0] - A.e[1][0]; 
		res.e[1][1] = e[1][1] - A.e[1][1]; 
		res.e[1][2] = e[1][2] - A.e[1][2];
		res.e[2][0] = e[2][0] - A.e[2][0]; 
		res.e[2][1] = e[2][1] - A.e[2][1]; 
		res.e[2][2] = e[2][2] - A.e[2][2];
	}
	else{
		#pragma unroll 
		for(int i = 0; i < Ncolors; i++)
		#pragma unroll 
		for(int j = 0; j < Ncolors; j++)
			res.e[i][j] = e[i][j] - A.e[i][j];
	}
	return res;
  }	
  M_HOSTDEVICE _matrixsun<Real,Ncolors> operator-=(const _matrixsun<Real,Ncolors> REF(A))  {
	if(Ncolors == 3){
		e[0][0] -= A.e[0][0]; e[0][1] -= A.e[0][1]; e[0][2] -= A.e[0][2];
		e[1][0] -= A.e[1][0]; e[1][1] -= A.e[1][1]; e[1][2] -= A.e[1][2];
		e[2][0] -= A.e[2][0]; e[2][1] -= A.e[2][1]; e[2][2] -= A.e[2][2];
	}
	else{
		#pragma unroll 
		for(int i = 0; i < Ncolors; i++)
		#pragma unroll 
		for(int j = 0; j < Ncolors; j++)
			e[i][j] -= A.e[i][j];
	}
	return *this;
  }
  M_HOSTDEVICE _matrixsun<Real,Ncolors> operator-(const complex REF(A)) const {
	_matrixsun<Real,Ncolors> res;
	if(Ncolors == 3){
		res.e[0][0] = e[0][0]- A; res.e[0][1] = e[0][1]- A; res.e[0][2] = e[0][2]- A;
		res.e[1][0] = e[1][0]- A; res.e[1][1] = e[1][1]- A; res.e[1][2] = e[1][2]- A;
		res.e[2][0] = e[2][0]- A, res.e[2][1] = e[2][1]- A; res.e[2][2] = e[2][2]- A;
	}
	else{
		#pragma unroll 
		for(int i = 0; i < Ncolors; i++)
		#pragma unroll 
		for(int j = 0; j < Ncolors; j++)
			res.e[i][j] = e[i][j] - A;
	}
	return res;
  }	
  M_HOSTDEVICE _matrixsun<Real,Ncolors> operator-=(const complex REF(A))  {
	if(Ncolors == 3){
		e[0][0] -= A; e[0][1] -= A; e[0][2] -= A;
		e[1][0] -= A; e[1][1] -= A; e[1][2] -= A;
		e[2][0] -= A, e[2][1] -= A; e[2][2] -= A;
	}
	else{
		#pragma unroll 
		for(int i = 0; i < Ncolors; i++)
		#pragma unroll 
		for(int j = 0; j < Ncolors; j++)
			e[i][j] -= A;
	}
	return *this;
  }
  M_HOSTDEVICE _matrixsun<Real,Ncolors> operator-(const Real REF(A)) const {
    _matrixsun<Real,Ncolors> res;
	if(Ncolors == 3){
		res.e[0][0] = e[0][0]- A; res.e[0][1] = e[0][1]- A; res.e[0][2] = e[0][2]- A;
		res.e[1][0] = e[1][0]- A; res.e[1][1] = e[1][1]- A; res.e[1][2] = e[1][2]- A;
		res.e[2][0] = e[2][0]- A, res.e[2][1] = e[2][1]- A; res.e[2][2] = e[2][2]- A;
	}
	else{
		#pragma unroll 
		for(int i = 0; i < Ncolors; i++)
		#pragma unroll 
		for(int j = 0; j < Ncolors; j++)
			res.e[i][j] = e[i][j] - A;
	}
	return res;
  }	
  M_HOSTDEVICE _matrixsun<Real,Ncolors> operator-=(const Real REF(A))  {
	if(Ncolors == 3){
		e[0][0] -= A; e[0][1] -= A; e[0][2] -= A;
		e[1][0] -= A; e[1][1] -= A; e[1][2] -= A;
		e[2][0] -= A, e[2][1] -= A; e[2][2] -= A;
	}
	else{
		#pragma unroll 
		for(int i = 0; i < Ncolors; i++)
		#pragma unroll 
		for(int j = 0; j < Ncolors; j++)
			e[i][j] -= A;
	}
	return *this;
  }

  M_HOSTDEVICE bool operator==(const _matrixsun<Real,Ncolors> REF(A)) const {
	bool eqmatrix = true;
		for(int i = 0; i < Ncolors; i++)
		for(int j = 0; j < Ncolors; j++)
			if(e[i][j] != A.e[i][j]){
				eqmatrix = false;
				break;
			}
	return eqmatrix;
  }
  M_HOSTDEVICE bool operator!=(const _matrixsun<Real,Ncolors> REF(A)) const {
	bool eqmatrix = false;
		for(int i = 0; i < Ncolors; i++)
		for(int j = 0; j < Ncolors; j++)
			if(e[i][j] != A.e[i][j]){
				eqmatrix = true;
				break;
			}
	return eqmatrix;
  }


  //-------------------------------------------------------------------------------
  // MULTIPLICATION's
  M_HOSTDEVICE _matrixsun<Real,Ncolors> operator*(const _matrixsun<Real,Ncolors> REF(A)) const {	
	_matrixsun<Real,Ncolors> res;
	if(Ncolors == 3){
		res.e[0][0] = e[0][0] * A.e[0][0] + e[0][1] * A.e[1][0] + e[0][2] * A.e[2][0];
		res.e[0][1] = e[0][0] * A.e[0][1] + e[0][1] * A.e[1][1] + e[0][2] * A.e[2][1]; 
		res.e[0][2] = e[0][0] * A.e[0][2] + e[0][1] * A.e[1][2] + e[0][2] * A.e[2][2];
		res.e[1][0] = e[1][0] * A.e[0][0] + e[1][1] * A.e[1][0] + e[1][2] * A.e[2][0]; 
		res.e[1][1] = e[1][0] * A.e[0][1] + e[1][1] * A.e[1][1] + e[1][2] * A.e[2][1]; 
		res.e[1][2] = e[1][0] * A.e[0][2] + e[1][1] * A.e[1][2] + e[1][2] * A.e[2][2];
		res.e[2][0] = e[2][0] * A.e[0][0] + e[2][1] * A.e[1][0] + e[2][2] * A.e[2][0]; 
		res.e[2][1] = e[2][0] * A.e[0][1] + e[2][1] * A.e[1][1] + e[2][2] * A.e[2][1];
		res.e[2][2] = e[2][0] * A.e[0][2] + e[2][1] * A.e[1][2] + e[2][2] * A.e[2][2];
	}
	else{
		#pragma unroll 
		for(int i = 0; i < Ncolors; i++)	
		#pragma unroll 
		for(int j = 0; j < Ncolors; j++){
			res.e[i][j] = e[i][0]  * A.e[0][j];
			#pragma unroll 
			for(int k = 1; k < Ncolors; k++)   res.e[i][j]  += e[i][k]  * A.e[k][j];
		}
	}
	return res;
  }
  M_HOSTDEVICE _matrixsun<Real,Ncolors> operator*=(const _matrixsun<Real,Ncolors> REF(A))  {
	_matrixsun<Real,Ncolors> res;
	if(Ncolors == 3){
		res.e[0][0] = e[0][0] * A.e[0][0] + e[0][1] * A.e[1][0] + e[0][2] * A.e[2][0];
		res.e[0][1] = e[0][0] * A.e[0][1] + e[0][1] * A.e[1][1] + e[0][2] * A.e[2][1]; 
		res.e[0][2] = e[0][0] * A.e[0][2] + e[0][1] * A.e[1][2] + e[0][2] * A.e[2][2];
		res.e[1][0] = e[1][0] * A.e[0][0] + e[1][1] * A.e[1][0] + e[1][2] * A.e[2][0]; 
		res.e[1][1] = e[1][0] * A.e[0][1] + e[1][1] * A.e[1][1] + e[1][2] * A.e[2][1]; 
		res.e[1][2] = e[1][0] * A.e[0][2] + e[1][1] * A.e[1][2] + e[1][2] * A.e[2][2];
		res.e[2][0] = e[2][0] * A.e[0][0] + e[2][1] * A.e[1][0] + e[2][2] * A.e[2][0]; 
		res.e[2][1] = e[2][0] * A.e[0][1] + e[2][1] * A.e[1][1] + e[2][2] * A.e[2][1];
		res.e[2][2] = e[2][0] * A.e[0][2] + e[2][1] * A.e[1][2] + e[2][2] * A.e[2][2];
	}
	else{
		#pragma unroll 
		for(int i = 0; i < Ncolors; i++)	
		#pragma unroll 
		for(int j = 0; j < Ncolors; j++){
			res.e[i][j] = e[i][0]  * A.e[0][j];
			#pragma unroll 
			for(int k = 1; k < Ncolors; k++)   res.e[i][j]  += e[i][k]  * A.e[k][j];
		}
	}
	*this=res;
	return *this;
  }
  M_HOSTDEVICE _matrixsun<Real,Ncolors> operator*(const complex REF(A)) const {
    _matrixsun<Real,Ncolors> res;	
	if(Ncolors == 3){
		res.e[0][0] = e[0][0] * A; res.e[0][1] = e[0][1] * A; res.e[0][2] = e[0][2] * A;
		res.e[1][0] = e[1][0] * A; res.e[1][1] = e[1][1] * A; res.e[1][2] = e[1][2] * A;
		res.e[2][0] = e[2][0] * A, res.e[2][1] = e[2][1] * A; res.e[2][2] = e[2][2] * A;
	}
	else{
		#pragma unroll 
		for(int i = 0; i < Ncolors; i++)	
		#pragma unroll 
		for(int j = 0; j < Ncolors; j++)
			res.e[i][j]  = e[i][j]  * A;
	}
	return res;
  }
  M_HOSTDEVICE _matrixsun<Real,Ncolors> operator*=(const complex REF(A))  {
	if(Ncolors == 3){
		e[0][0] *= A; e[0][1] *= A; e[0][2] *= A;
		e[1][0] *= A; e[1][1] *= A; e[1][2] *= A;
		e[2][0] *= A, e[2][1] *= A; e[2][2] *= A;
	}
	else{
		#pragma unroll 
		for(int i = 0; i < Ncolors; i++)	
		#pragma unroll 
		for(int j = 0; j < Ncolors; j++)
			e[i][j]  *= A;
	}
	return *this;
  }
  M_HOSTDEVICE _matrixsun<Real,Ncolors> operator*(const Real REF(A)) const {
    _matrixsun<Real,Ncolors> res;	
	if(Ncolors == 3){
		res.e[0][0] = e[0][0] * A; res.e[0][1] = e[0][1] * A; res.e[0][2] = e[0][2] * A;
		res.e[1][0] = e[1][0] * A; res.e[1][1] = e[1][1] * A; res.e[1][2] = e[1][2] * A;
		res.e[2][0] = e[2][0] * A, res.e[2][1] = e[2][1] * A; res.e[2][2] = e[2][2] * A;
	}
	else{
		#pragma unroll 
		for(int i = 0; i < Ncolors; i++)	
		#pragma unroll 
		for(int j = 0; j < Ncolors; j++)
			res.e[i][j]  = e[i][j]  * A;
	}
	return res;
  }
  M_HOSTDEVICE _matrixsun<Real,Ncolors> operator*=(const Real REF(A))  {
	if(Ncolors == 3){
		e[0][0] *= A; e[0][1] *= A; e[0][2] *= A;
		e[1][0] *= A; e[1][1] *= A; e[1][2] *= A;
		e[2][0] *= A, e[2][1] *= A; e[2][2] *= A;
	}
	else{
		#pragma unroll 
		for(int i = 0; i < Ncolors; i++)	
		#pragma unroll 
		for(int j = 0; j < Ncolors; j++)
			e[i][j]  *= A;
	}
	return *this;
  }
  //-------------------------------------------------------------------------------
  //DIVISION's

  //NOT DONE YET: MATRIX X MATRIX 

  M_HOSTDEVICE _matrixsun<Real,Ncolors> operator/(const complex REF(A)) const {
	_matrixsun<Real,Ncolors> res;	
	if(Ncolors == 3){
		res.e[0][0] = e[0][0] / A; res.e[0][1] = e[0][1] / A; res.e[0][2] = e[0][2] / A;
		res.e[1][0] = e[1][0] / A; res.e[1][1] = e[1][1] / A; res.e[1][2] = e[1][2] / A;
		res.e[2][0] = e[2][0] / A, res.e[2][1] = e[2][1] / A; res.e[2][2] = e[2][2] / A;
	}
	else{
		#pragma unroll 
		for(int i = 0; i < Ncolors; i++)	
		#pragma unroll 
		for(int j = 0; j < Ncolors; j++)
			res.e[i][j]  = e[i][j] / A;
	}
	return res;
  }
  M_HOSTDEVICE _matrixsun<Real,Ncolors> operator/=(const complex REF(A))  {
	if(Ncolors == 3){
		e[0][0] /= A; e[0][1] /= A;  e[0][2] /= A;
		e[1][0] /= A; e[1][1] /= A; e[1][2] /= A;
		e[2][0] /= A,  e[2][1] /= A; e[2][2] /= A;
	}
	else{
		#pragma unroll 
		for(int i = 0; i < Ncolors; i++)	
		#pragma unroll 
		for(int j = 0; j < Ncolors; j++)
			e[i][j] /= A;
	}
	return *this;
  }
  M_HOSTDEVICE _matrixsun<Real,Ncolors> operator/(const Real REF(A)) const {
    _matrixsun<Real,Ncolors> res;	
	if(Ncolors == 3){
		res.e[0][0] = e[0][0] / A; res.e[0][1] = e[0][1] / A; res.e[0][2] = e[0][2] / A;
		res.e[1][0] = e[1][0] / A; res.e[1][1] = e[1][1] / A; res.e[1][2] = e[1][2] / A;
		res.e[2][0] = e[2][0] / A, res.e[2][1] = e[2][1] / A; res.e[2][2] = e[2][2] / A;
	}
	else{
		#pragma unroll 
		for(int i = 0; i < Ncolors; i++)	
		#pragma unroll 
		for(int j = 0; j < Ncolors; j++)
			res.e[i][j]  = e[i][j] / A;
	}
	return res;
  }
  M_HOSTDEVICE _matrixsun<Real,Ncolors> operator/=(const Real REF(A))  {
	if(Ncolors == 3){
		e[0][0] /= A; e[0][1] /= A;  e[0][2] /= A;
		e[1][0] /= A; e[1][1] /= A; e[1][2] /= A;
		e[2][0] /= A,  e[2][1] /= A; e[2][2] /= A;
	}
	else{
		#pragma unroll 
		for(int i = 0; i < Ncolors; i++)	
		#pragma unroll 
		for(int j = 0; j < Ncolors; j++)
			e[i][j] /= A;
	}
	return *this;
  }

  //-------------------------------------------------------------------------------
  // compute the determinant
  M_HOSTDEVICE _complex<Real> det() const {
	_complex<Real> res;
	if(Ncolors == 3){ 
		res  = e[0][1] * e[1][2] * e[2][0];
		res -= e[0][2] * e[1][1] * e[2][0];
		res += e[0][2] * e[1][0] * e[2][1];
		res -= e[0][0] * e[1][2] * e[2][1];
		res -= e[0][1] * e[1][0] * e[2][2];
		res += e[0][0] * e[1][1] * e[2][2];
	}
	else{
		_matrixsun<Real,Ncolors> b;
		for(int i = 0; i < Ncolors; i++)
		for(int j = 0; j < Ncolors; j++) 	b.e[i][j] = e[i][j];
		for(int j = 0; j < Ncolors; j++){
			for(int i = 0; i <= j; i++){
				res = b.e[j][i];
				for(int c = 0; c < i; c++) 	res -= b.e[c][i] * b.e[j][c];			    
				b.e[j][i] = res;
			}
			for(int i = (j+1); i < Ncolors; i++){
				res = b.e[j][i];
				for(int c = 0; c < j; c++) 	  res -= b.e[c][i] * b.e[j][c];
				b.e[j][i] = b.e[j][j].conj() * res / b.e[j][j].abs2();
			}
		}
		res = b.e[0][0] * b.e[1][1];
		for(int c = 2; c < Ncolors; c++) res *= b.e[c][c];

		/*_matrixsun<Real,Ncolors> a; 
		for(int i = 0; i < Ncolors; i++)
		for(int j = 0; j < Ncolors; j++) 	
			a.e[i][j] = e[i][j];*/
		//bool DetExists = true;
		//int  l = 1;
		//for (int k = 0; k<NCOLORS; k++) {
			/*if (a.e[k][k].real() == 0.0 && a.e[k][k].imag() == 0.0) {
				DetExists = false;
				for (int i = k+1; i<NCOLORS; i++) {
					if (a.e[i][k].real() != 0.0 || a.e[i][k].imag() != 0.0) {
						for (int j = 1; j<NCOLORS; j++) {
							res = a.e[i][j];
							a.e[i][j] = a.e[k][j];
							a.e[k][j] = res;
						}
						DetExists = true;
						//l=-l;
						break;
					}
				}
				if (DetExists == false) return complex::zero();
			}*/
			/*for (int j = k+1; j<NCOLORS; j++) {
				res = a.e[j][k]/a.e[k][k];
				for (int i = k+1; i<NCOLORS; i++) {
					a.e[j][i] = a.e[j][i]-a.e[k][i]*res;
					a.e[j][i] = a.e[j][i] - res;
				}
			}
		}
		// Calculate determinant by finding product of diagonal elements
		res = a.e[0][0] * a.e[1][1];
		for(int c = 2; c < Ncolors; c++) 	res *= a.e[c][c];*/
	}
	return res;
  }

  //-------------------------------------------------------------------------------
  // negation of this matrix
  M_HOSTDEVICE _matrixsun<Real,Ncolors> operator-() const {
    _matrixsun<Real,Ncolors> res;	
	if(Ncolors == 3){
		res.e[0][0]=-e[0][0]; res.e[0][1]=-e[0][1]; res.e[0][2]=-e[0][2];
		res.e[1][0]=-e[1][0]; res.e[1][1]=-e[1][1]; res.e[1][2]=-e[1][2];
		res.e[2][0]=-e[2][0]; res.e[2][1]=-e[2][1]; res.e[2][2]=-e[2][2];
	}
	else{
		#pragma unroll 
		for(int i = 0; i < Ncolors; i++)	
		#pragma unroll 
		for(int j = 0; j < Ncolors; j++)
			res.e[i][j] = -e[i][j];
	}
	return res;
  }

  //-------------------------------------------------------------------------------
  M_HOSTDEVICE _matrixsun<Real,Ncolors> dagger() const {
    _matrixsun<Real,Ncolors> res;
	if(Ncolors == 3){
		res.e[0][0]=~e[0][0]; res.e[0][1]=~e[1][0]; res.e[0][2]=~e[2][0];
		res.e[1][0]=~e[0][1]; res.e[1][1]=~e[1][1]; res.e[1][2]=~e[2][1];
		res.e[2][0]=~e[0][2]; res.e[2][1]=~e[1][2]; res.e[2][2]=~e[2][2];	
	}
	else{
		#pragma unroll 
		for(int i = 0; i < Ncolors; i++)
		#pragma unroll 
		for(int j = 0; j < Ncolors; j++)
			res.e[i][j] = ~e[j][i];
	}
	return res;
  }
  //-------------------------------------------------------------------------------
  /**
  	@brief This conly conjugates all the SU(Nc) matrix elements.
  */
  M_HOSTDEVICE _matrixsun<Real,Ncolors> conj() const {
    _matrixsun<Real,Ncolors> res;
	if(Ncolors == 3){
	    res.e[0][0]=~e[0][0]; res.e[0][1]=~e[0][1]; res.e[0][2]=~e[0][2];
	    res.e[1][0]=~e[1][0]; res.e[1][1]=~e[1][1]; res.e[1][2]=~e[1][2];
	    res.e[2][0]=~e[2][0]; res.e[2][1]=~e[2][1]; res.e[2][2]=~e[2][2];
	}
	else{
		#pragma unroll 
		    for(int i = 0; i < Ncolors; i++)
		#pragma unroll 
		      for(int j = 0; j < Ncolors; j++)
			res.e[i][j] = ~e[i][j];
	}
	return res;
  }
  //-------------------------------------------------------------------------------
  /**
  	@brief Return the SU(Nc) matrix complex trace
  */
  M_HOSTDEVICE complex trace() const {
	if(Ncolors == 3)  return e[0][0] + e[1][1] + e[2][2];
	else{
		complex res=complex::zero();	
		#pragma unroll 
		for(int i = 0; i < Ncolors; i++) res += e[i][i] ;
		return res;
	}
  }
  //-------------------------------------------------------------------------------
  /**
  	@brief Return the SU(Nc) matrix real trace
  */
  M_HOSTDEVICE Real realtrace()  {
	if(Ncolors == 3)  return e[0][0].real() + e[1][1].real() + e[2][2].real();
	else{
		Real res=(Real)0.0;	
		#pragma unroll 
		for(int i = 0; i < Ncolors; i++) res += e[i][i].real() ;
		return res;
	}
  }
  //-------------------------------------------------------------------------------
  M_HOSTDEVICE _matrixsun<Real,Ncolors> subtraceunit() const  {
	if(Ncolors == 3){
		_complex<Real> tr = (e[0][0] + e[1][1] + e[2][2]) / 3.0;
		_matrixsun<Real,Ncolors> res;
		res.e[0][0] =e[0][0]- tr; res.e[0][1] = e[0][1]; res.e[0][2] = e[0][2];
		res.e[1][0] = e[1][0]; res.e[1][1] = e[1][1]-tr; res.e[1][2] = e[1][2];
		res.e[2][0] = e[2][0]; res.e[2][1] = e[2][1]; res.e[2][2] = e[2][2]-tr;
		return res;
	}
	else{
		complex res=complex::zero();	
		#pragma unroll 
		for(int i = 0; i < Ncolors; i++)	res += e[i][i] ;
		res /= (Real)Ncolors;
		_matrixsun<Real,Ncolors> m;
		#pragma unroll 
		for(int i = 0; i < Ncolors; i++)
		#pragma unroll 
		for(int j = 0; j < Ncolors; j++){
			if(i==j) m.e[i][j] = e[i][j] - res;
			else m.e[i][j] = e[i][j];
		}
		return m;
	}
  }



  //-------------------------------------------------------------------------------
  /**
  	@brief Returns a SU(Nc) matrix with zeros in all elements
  */
  static M_HOSTDEVICE _matrixsun<Real,Ncolors> zero(void) {
    _matrixsun<Real,Ncolors> res;
	if(Ncolors == 3){
	    res.e[0][0] = _complex<Real>::zero(); res.e[0][1] = _complex<Real>::zero(); res.e[0][2] = _complex<Real>::zero();
	    res.e[1][0] = _complex<Real>::zero(); res.e[1][1] = _complex<Real>::zero(); res.e[1][2] = _complex<Real>::zero();
	    res.e[2][0] = _complex<Real>::zero(); res.e[2][1] = _complex<Real>::zero(); res.e[2][2] = _complex<Real>::zero();
	}
	else{
		#pragma unroll 
		for(int i = 0; i < Ncolors; i++)
		#pragma unroll 
		for(int j = 0; j < Ncolors; j++)
			res.e[i][j] = _complex<Real>::zero();
	}
	return res;
  }
  /**
  	@brief Returns the identity SU(Nc) matrix
  */
  static M_HOSTDEVICE _matrixsun<Real,Ncolors> identity(void) {
    _matrixsun<Real,Ncolors> res;
	if(Ncolors == 3){
		res.e[0][0] = _complex<Real>::one(); res.e[0][1] = _complex<Real>::zero(); res.e[0][2] = _complex<Real>::zero();
		res.e[1][0] = _complex<Real>::zero(); res.e[1][1] = _complex<Real>::one(); res.e[1][2] = _complex<Real>::zero();
		res.e[2][0] = _complex<Real>::zero(); res.e[2][1] = _complex<Real>::zero(); res.e[2][2] = _complex<Real>::one();
	}
	else{
		#pragma unroll 
		for(int i = 0; i < Ncolors; i++)
		#pragma unroll 
		for(int j = 0; j < Ncolors; j++){
			if(i==j) res.e[i][j] = _complex<Real>::one();
			else res.e[i][j] = _complex<Real>::zero();
		}
	}
	return res;
  }
  /**
  	@brief Returns the identity SU(Nc) matrix
  */
  static M_HOSTDEVICE _matrixsun<Real,Ncolors> unit(void) {
    _matrixsun<Real,Ncolors> res;
	if(Ncolors == 3){
		res.e[0][0] = _complex<Real>::one(); res.e[0][1] = _complex<Real>::zero(); res.e[0][2] = _complex<Real>::zero();
		res.e[1][0] = _complex<Real>::zero(); res.e[1][1] = _complex<Real>::one(); res.e[1][2] = _complex<Real>::zero();
		res.e[2][0] = _complex<Real>::zero(); res.e[2][1] = _complex<Real>::zero(); res.e[2][2] = _complex<Real>::one();
	}
	else{
		#pragma unroll 
		for(int i = 0; i < Ncolors; i++)
		#pragma unroll 
		for(int j = 0; j < Ncolors; j++){
			if(i==j) res.e[i][j] = _complex<Real>::one();
			else res.e[i][j] = _complex<Real>::zero();
		}
	}
	return res;
  }
  /**
  	@brief Returns a SU(Nc) matrix with all elements equal to one
  */
  static M_HOSTDEVICE _matrixsun<Real,Ncolors> ones(void) {
    _matrixsun<Real,Ncolors> res;
	if(Ncolors == 3){
		res.e[0][0] = _complex<Real>::one(); res.e[0][1] = _complex<Real>::one(); res.e[0][2] = _complex<Real>::one();
		res.e[1][0] = _complex<Real>::one(); res.e[1][1] = _complex<Real>::one(); res.e[1][2] = _complex<Real>::one();
		res.e[2][0] = _complex<Real>::one(); res.e[2][1] = _complex<Real>::one(); res.e[2][2] = _complex<Real>::one();
	}
	else{
		#pragma unroll 
		for(int i = 0; i < Ncolors; i++)
		#pragma unroll 
		for(int j = 0; j < Ncolors; j++)
		res.e[i][j] = _complex<Real>::one();
	}
	return res;
  }

  //-------------------------------------------------------------------------------
  // MULTIPLICATION's
  /**
  	@brief Returns C <- A^\dagger x B
  */
friend   M_HOSTDEVICE _matrixsun<Real,Ncolors> UDaggerU(const _matrixsun<Real,Ncolors> REF(A), const _matrixsun<Real,Ncolors> REF(B)) {
	_matrixsun<Real,Ncolors> C;
	#pragma unroll 
	for(int i = 0; i < Ncolors; i++)	
	#pragma unroll 
	for(int j = 0; j < Ncolors; j++){
		C.e[i][j]  = ~A.e[0][i]  * B.e[0][j];
		#pragma unroll 
		for(int k = 1; k < Ncolors; k++)
			C.e[i][j]  += ~A.e[k][i]  * B.e[k][j];
	}
	return C;
  }
  /**
  	@brief Returns C <- A x B^\dagger
  */
friend   M_HOSTDEVICE _matrixsun<Real,Ncolors> UUDagger(const _matrixsun<Real,Ncolors> REF(A), const _matrixsun<Real,Ncolors> REF(B)) {
	_matrixsun<Real,Ncolors> C;
	#pragma unroll 
	for(int i = 0; i < Ncolors; i++)	
	#pragma unroll 
	for(int j = 0; j < Ncolors; j++){
			C.e[i][j]  = A.e[i][0]  * ~B.e[j][0];
		#pragma unroll 
		for(int k = 1; k < Ncolors; k++)
			C.e[i][j]  += A.e[i][k]  * ~B.e[j][k];
	}
	return C;
  }



  /**
  	@brief Returns the trace of A^\dagger x B
  */
friend   M_HOSTDEVICE complex UDaggerUTrace(const _matrixsun<Real,Ncolors> REF(A), const _matrixsun<Real,Ncolors> REF(B)) {
	complex res = complex::zero();
	#pragma unroll 
	for(int i = 0; i < Ncolors; i++)	
	#pragma unroll 
	for(int k = 0; k < Ncolors; k++)
		res  += ~A.e[k][i]  * B.e[k][i];
	return res;
  }
  /**
  	@brief Returns the trace of A x B^\dagger
  */
friend   M_HOSTDEVICE complex UUDaggerTrace(const _matrixsun<Real,Ncolors> REF(A), const _matrixsun<Real,Ncolors> REF(B)) {
	complex res = complex::zero();	
	#pragma unroll 
	for(int i = 0; i < Ncolors; i++)	
	#pragma unroll 
	for(int k = 0; k < Ncolors; k++)
		res  += A.e[i][k]  * ~B.e[i][k];
	return res;
  }
  /**
  	@brief Returns the real trace of A^\dagger x B
  */
friend   M_HOSTDEVICE Real UDaggerURealTrace(const _matrixsun<Real,Ncolors> REF(A), const _matrixsun<Real,Ncolors> REF(B)) {
	Real res = (Real)0.0;
	#pragma unroll 
	for(int i = 0; i < Ncolors; i++)	
	#pragma unroll 
	for(int k = 0; k < Ncolors; k++)
		res  += A.e[k][i].real()  * B.e[k][i].real() + A.e[k][i].imag()  * B.e[k][i].imag();
	return res;
  }
  /**
  	@brief Returns the real trace of A x B^\dagger
  */
friend   M_HOSTDEVICE Real UUDaggerRealTrace(const _matrixsun<Real,Ncolors> REF(A), const _matrixsun<Real,Ncolors> REF(B)) {
    Real res = (Real)0.0;	
#pragma unroll 
    for(int i = 0; i < Ncolors; i++)	
#pragma unroll 
	for(int k = 0; k < Ncolors; k++){
	  res  += A.e[k][i].real()  * B.e[k][i].real() + A.e[k][i].imag()  * B.e[k][i].imag();
	}
    return res;
  }


  //-------------------------------------------------------------------------------
  M_HOSTDEVICE  void print() {
	for(int i = 0; i < Ncolors; i++){
		for(int j = 0; j < Ncolors; j++){
			if( i == 0 && j == 0 ) 	  printf("[ ");
			else   printf("  ");		
			printf("%.10e + %.10ej", e[i][j].real(), e[i][j].imag());
			if( i == Ncolors - 1 && j == Ncolors - 1 ) 	  printf(" ]\n");	
			else   printf("\t");		
		}
		printf("\n");
	}
  }

  friend M_HOST std::ostream& operator<<( std::ostream& out, _matrixsun<Real,Ncolors> M ) {
    for(int i = 0; i < Ncolors; i++)
      for(int j = 0; j < Ncolors; j++){
      	out << M.e[i][j];
      	if(j + i * Ncolors < Ncolors * Ncolors - 1) out << "\t";
      }
    return out;
  }
};

#define msu3 _matrixsun<Real, NCOLORS>
typedef _matrixsun<float, NCOLORS> msu3s;
typedef _matrixsun<double, NCOLORS> msu3d;
#define msun _matrixsun<Real, NCOLORS>
typedef _matrixsun<float, NCOLORS> msuns;
typedef _matrixsun<double, NCOLORS> msund;



////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
template <class Real> 
 M_HOSTDEVICE complex traceUV(msun a, msun b){
#if (NCOLORS == 3)
complex sum =a.e[0][0] * b.e[0][0]; 
sum+=a.e[0][1] * b.e[1][0];
sum+=a.e[0][2] * b.e[2][0]; 
sum+=a.e[1][0] * b.e[0][1]; 
sum+=a.e[1][1] * b.e[1][1];
sum+=a.e[1][2] * b.e[2][1]; 
sum+=a.e[2][0] * b.e[0][2]; 
sum+=a.e[2][1] * b.e[1][2]; 
sum+=a.e[2][2] * b.e[2][2]; 
  return sum;
#else
  complex sum = complex::zero();
  for(int i=0;i<3;i++)
    for(int j=0;j<3;j++){
	sum += a.e[i][j] * b.e[j][i];
    }
  return sum;
#endif
}
template <class Real> 
 M_HOSTDEVICE complex traceUVdagger(msun a, msun b){
#if (NCOLORS == 3) 
    complex sum = a.e[0][0] * b.e[0][0].conj();
     sum += a.e[0][1] * b.e[0][1].conj();
     sum += a.e[0][2] * b.e[0][2].conj();
     sum += a.e[1][0] * b.e[1][0].conj();
     sum += a.e[1][1] * b.e[1][1].conj();
     sum += a.e[1][2] * b.e[1][2].conj();
     sum += a.e[2][0] * b.e[2][0].conj();
     sum += a.e[2][1] * b.e[2][1].conj();
     sum += a.e[2][2] * b.e[2][2].conj();
    return sum;
#else
    complex sum = complex::zero();
    for(int i=0;i<NCOLORS;i++)
	for(int j=0;j<NCOLORS;j++)
		sum+= a.e[i][j] * b.e[i][j].conj();
    return sum;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
template <class Real> 
 M_HOSTDEVICE Real realtraceUV(msun a, msun b){
#if (NCOLORS == 3) 
Real sum=a.e[0][0].real() * b.e[0][0].real() - a.e[0][0].imag() * b.e[0][0].imag();
sum+=a.e[0][1].real() * b.e[1][0].real() - a.e[0][1].imag() * b.e[1][0].imag();
sum+=a.e[0][2].real() * b.e[2][0].real() - a.e[0][2].imag() * b.e[2][0].imag();
sum+=a.e[1][0].real() * b.e[0][1].real() - a.e[1][0].imag() * b.e[0][1].imag();
sum+=a.e[1][1].real() * b.e[1][1].real() - a.e[1][1].imag() * b.e[1][1].imag();
sum+=a.e[1][2].real() * b.e[2][1].real() - a.e[1][2].imag() * b.e[2][1].imag();
sum+=a.e[2][0].real() * b.e[0][2].real() - a.e[2][0].imag() * b.e[0][2].imag();
sum+=a.e[2][1].real() * b.e[1][2].real() - a.e[2][1].imag() * b.e[1][2].imag();
sum+=a.e[2][2].real() * b.e[2][2].real() - a.e[2][2].imag() * b.e[2][2].imag();
  return sum;
#else
  Real sum = 0.0;
  for(int i=0;i<3;i++)
    for(int j=0;j<3;j++){
	sum += a.e[i][j].real() * b.e[j][i].real() - a.e[i][j].imag() * b.e[j][i].imag();
    }
  return sum;
#endif
}

template <class Real> 
 M_HOSTDEVICE Real realtraceUVdagger(msun a, msun b){
#if (NCOLORS == 3) 
    Real sum = a.e[0][0].real() * b.e[0][0].real()  + a.e[0][0].imag() * b.e[0][0].imag();
     sum += a.e[0][1].real() * b.e[0][1].real()  + a.e[0][1].imag() * b.e[0][1].imag();
     sum += a.e[0][2].real() * b.e[0][2].real()  + a.e[0][2].imag() * b.e[0][2].imag();
     sum += a.e[1][0].real() * b.e[1][0].real()  + a.e[1][0].imag() * b.e[1][0].imag();
     sum += a.e[1][1].real() * b.e[1][1].real()  + a.e[1][1].imag() * b.e[1][1].imag();
     sum += a.e[1][2].real() * b.e[1][2].real()  + a.e[1][2].imag() * b.e[1][2].imag();
     sum += a.e[2][0].real() * b.e[2][0].real()  + a.e[2][0].imag() * b.e[2][0].imag();
     sum += a.e[2][1].real() * b.e[2][1].real()  + a.e[2][1].imag() * b.e[2][1].imag();
     sum += a.e[2][2].real() * b.e[2][2].real()  + a.e[2][2].imag() * b.e[2][2].imag();
    return sum;
#else
    Real sum = 0.0;
    for(int i=0;i<NCOLORS;i++)
	for(int j=0;j<NCOLORS;j++)
		sum+= a.e[i][j].real() * b.e[i][j].real() + a.e[i][j].imag() * b.e[i][j].imag();
    return sum;
#endif
}



template <class Real> 
 M_HOSTDEVICE msun timesMinusI(msun a){
  msun res = msun::zero();
  for(int i=0;i<NCOLORS;i++)
  for(int j=0;j<NCOLORS;j++){
    res.e[i][j].real() = a.e[i][j].imag();
    res.e[i][j].imag() = -a.e[i][j].real();
  }
  return res;
}
template <class Real> 
 M_HOSTDEVICE msun timesI(msun a){
  msun res = msun::zero();
  for(int i=0;i<NCOLORS;i++)
  for(int j=0;j<NCOLORS;j++){
    res.e[i][j].real() = -a.e[i][j].imag();
    res.e[i][j].imag() = a.e[i][j].real();
  }
  return res;
}

}



#endif // #ifndef MATRIXSUN_H


