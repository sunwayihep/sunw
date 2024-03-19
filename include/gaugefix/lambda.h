
#ifndef LAMBDA_H
#define LAMBDA_H


#include <gaugearray.h>

namespace CULQCD{

template <class Real>  
void CreateLambdaB(Real *lambdab, RNG randstates, string filename="", bool binformat=true, bool savebinsingleprec=true);

template <class Real>  
void Check_LambdaB(Real *lambdab);

template <class Real, class SavePrec>  
void saveLambdaB(Real *Lambdab, ReadMode mode, string filename, bool binformat=true);

template <class Real, class SavePrec>  
void readLambdaB(Real *Lambdab, ReadMode mode, string filename, bool binformat=true);


template<class Real>
void RandGaugeTransf(gauge &gaugein, RNG &randstates);




template <class Real>  
void CreateLambdaI( gauge &lambda, Real *lambdab, Real xi);
template <class Real>  
void CreateLambda( gauge &lambda, Real *lambdab, Real xi);







}

#endif
