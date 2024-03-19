#ifndef COMPLEX_H
#define COMPLEX_H


#include <string.h>
#include <iostream>
#include <iomanip>
#include <assert.h>
#include <math.h>


#include <cuda_common.h>
#include <cuda_vector_types.h>


namespace CULQCD{

/**
  @brief Class declaration for complex numbers in single and double precision
*/
template <class Real> 
class _complex {

 public:
    /** 
      @brief complex number representation, float2/double2 for single and double precision respectively.
      val.x hold real part and val.y hold the imaginary part
    */
    typedef typename MakeVector<Real, 2>::type Real2;
    Real2 val;

//Constructors
  M_HOSTDEVICE _complex(){
    val.x = (Real)0.0; val.y = (Real)0.0;
  }
  M_HOSTDEVICE _complex(const Real REF(a), const Real REF(b)){
    val.x = a; val.y = b;
  }
  M_HOSTDEVICE _complex(const Real REF(a)){
    val.x = a; val.y = 0.;
  }
  M_HOSTDEVICE _complex(const _complex<Real> REF(a)){
    val.x = a.real(); val.y = a.imag();
  }

//Assignment operators
  M_HOSTDEVICE _complex<Real>& operator=(const Real REF(a)) {
    val.x = a; val.y = 0;
    return *this;
  };
  M_HOSTDEVICE _complex<Real>& operator=(const  _complex<Real> REF(a)) {
    val.x = a.val.x; val.y = a.val.y;
    return *this;
  };
  // assignment of a pair of Ts to complex
  M_HOSTDEVICE _complex<Real>& operator=(const Real ARRAYREF(a,2)) {
    val.x = a[0]; val.y = a[1];
    return *this;
  };


  M_HOSTDEVICE bool operator==(const _complex<Real> REF(a)) const {
    if(val.x == a.val.x && val.y == a.val.y) return true;
    return false;
  };

  M_HOSTDEVICE bool operator!=(const _complex<Real> REF(a)) const {
    if(val.x == a.val.x && val.y == a.val.y) return false;
    return true;
  };

  // return references to the T and imaginary components
  M_HOSTDEVICE Real& real() {return val.x;};
  M_HOSTDEVICE Real& imag() {return val.y;};

  M_HOSTDEVICE Real real()const {return val.x;};
  M_HOSTDEVICE Real imag()const {return val.y;};


//Operator +
  // add complex numbers
  M_HOSTDEVICE _complex<Real> operator+(const _complex<Real> REF(b)) const {
    return _complex(val.x+b.val.x, val.y+b.val.y);
  }
  // add scalar to complex
  M_HOSTDEVICE _complex<Real> operator+(const Real REF(b)) const {
    return _complex(val.x+b, val.y);
  }
    friend M_HOSTDEVICE _complex<Real> operator+(const Real REF(a), const _complex<Real>& z){
      return _complex(z.val.x + a, z.val.y);
    } 
  // add scalar to complex
  M_HOSTDEVICE _complex<Real> operator+=(const Real REF(b)) {
    val.x += b;
    return *this;
  } 
  // add complex numbers
  M_HOSTDEVICE _complex<Real> operator+=(const _complex<Real> REF(b)) {
    val.x += b.val.x;
    val.y += b.val.y;
    return *this;
  }

  M_HOSTDEVICE    volatile _complex<Real>& operator +=( volatile  _complex<Real> & a ) volatile{
    val.x += a.val.x;
    val.y += a.val.y;
    return *this;
  }


//Operator -
  // subtract complex numbers
  M_HOSTDEVICE _complex<Real> operator-(const _complex<Real> REF(b)) const {
    _complex<Real> result;
    result.val.x = val.x - b.val.x;
    result.val.y = val.y  - b.val.y;
    return result;
  }
    friend M_HOSTDEVICE _complex<Real> operator-(const Real REF(a), const _complex<Real>& z){
      return _complex(a - z.val.x,-z.val.y);
    }
  // negate a complex number
  M_HOSTDEVICE _complex<Real> operator-() const {
    _complex<Real> result;
    result.val.x = -val.x;
    result.val.y = -val.y;
    return result;
  }

  // subtract scalar from complex
  M_HOSTDEVICE _complex<Real> operator-(const Real REF(b)) const {
    return  _complex(val.x-b,val.y);
  }

  // add scalar to complex
  M_HOSTDEVICE _complex<Real> operator-=(const Real REF(b)) {
    val.x -= b;
    return *this;
  } 
  // add complex numbers
  M_HOSTDEVICE _complex<Real> operator-=(const _complex<Real> REF(b)) {
    val.x -= b.val.x;
    val.y -= b.val.y;
    return *this;
  }


//Operator *
  // multiply complex numbers
  M_HOSTDEVICE _complex<Real> operator*(const _complex<Real> REF(b)) const {
    return _complex(val.x * b.val.x - val.y * b.val.y, val.y * b.val.x + val.x * b.val.y);
  }
  friend M_HOSTDEVICE _complex<Real> operator*(const Real REF(a), const _complex<Real>& z){
      return _complex(z.val.x * a, z.val.y * a);
    }
  // multiply complex numbers
  M_HOSTDEVICE _complex<Real> operator*=(const _complex<Real> REF(b)) {
    Real tmp = val.x * b.val.x - val.y * b.val.y;
    val.y = val.y * b.val.x + val.x * b.val.y;
    val.x = tmp;
    return *this;
  }
  // multiply complex with scalar
  M_HOSTDEVICE _complex<Real> operator*(const Real REF(b)) const {
    return _complex(val.x * b, val.y * b);
  } 
  // add scalar to complex
  M_HOSTDEVICE _complex<Real> operator*=(const Real REF(b)) {
    val.x *= b;
    val.y *= b;
    return *this;
  } 

//Operator /
  // divide complex numbers
  M_HOSTDEVICE _complex<Real> operator/(const _complex<Real> REF(b)) const {
    Real tmp = (Real)1.0 / ( b.val.x * b.val.x + b.val.y * b.val.y );
    _complex<Real> result;
    result.val.x = (val.x * b.val.x + val.y * b.val.y ) * tmp;
    result.val.y = (val.y * b.val.x - val.x * b.val.y ) * tmp;
    return result;
  }
  M_HOSTDEVICE _complex<Real> operator/=(const _complex<Real> REF(b)) {
    Real tmp = (Real)1.0 / ( b.val.x * b.val.x + b.val.y * b.val.y );
    _complex<Real> result;
    result.val.x = (val.x * b.val.x + val.y * b.val.y ) * tmp;
    result.val.y = (val.y * b.val.x - val.x * b.val.y ) * tmp;
	*this = result;
    return *this;
  }
  friend M_HOSTDEVICE _complex<Real> operator/(const Real REF(a), const _complex<Real>& b){
    Real tmp = (Real)1.0 / ( b.val.x * b.val.x + b.val.y * b.val.y );
    _complex<Real> result;
    result.val.x = (a * b.val.x ) * tmp;
    result.val.y = (- a * b.val.y ) * tmp;
    return result;
    }
  // divide complex by scalar
  M_HOSTDEVICE _complex<Real> operator/(const Real REF(b)) const {
    return _complex(val.x /b, val.y/b);
  }
  M_HOSTDEVICE _complex<Real> operator/=(const Real REF(b)) {
    val.x /= b;
    val.y /= b;
    return *this;
  }




  // complex conjugate
  M_HOSTDEVICE _complex<Real> operator~() const {
    return _complex(val.x, -val.y);
  }
  // complex conjugate
  M_HOSTDEVICE _complex<Real> conj() const {
    return _complex(val.x, -val.y);
  }

  // complex modulus (complex absolute)
  M_HOSTDEVICE Real abs() const {
    return sqrt( val.x*val.x + val.y*val.y );
  }
  // squaval.x complex modulus
  M_HOSTDEVICE Real abs2() const {
    return  ( val.x*val.x + val.y*val.y );
  }

  // complex phase angle
  M_HOSTDEVICE Real phase() const {
    return atan2( val.y, val.x );
  }

  M_HOSTDEVICE Real angle() const {
    return atan2( val.y, val.x );
  }

  // arg
  M_HOSTDEVICE Real arg() const {
    Real r = sqrt( val.x*val.x + val.y*val.y );
    Real res = acos(val.x /r);
    if(val.y < 0)
      res = -res;
    return res;
  }
  
  
  // a possible alternative to a _complex constructor
  static M_HOSTDEVICE _complex<Real> make_complex(const Real REF(a), const Real REF(b)){
    _complex<Real> res;
    res.val.x = a;
    res.val.y = b;
    return res;
  }
  // a possible alternative to a _complex constructor
  static M_HOSTDEVICE _complex<Real> make_complexVolatile(const volatile Real REF(a), const volatile Real REF(b)){
    _complex<Real> res;
    res.val.x = a;
    res.val.y = b;
    return res;
  }
  // a possible alternative to a _complex constructor
  static M_HOSTDEVICE _complex<Real> make_complexVolatile(const Real REF(a), const volatile Real REF(b)){
    _complex<Real> res;
    res.val.x = a;
    res.val.y = b;
    return res;
  }
  // a possible alternative to a _complex constructor
  static M_HOSTDEVICE _complex<Real> make_complexVolatile(const volatile Real REF(a), const Real REF(b)){
    _complex<Real> res;
    res.val.x = a;
    res.val.y = b;
    return res;
  }
  // a possible alternative to a _complex constructor
  static M_HOSTDEVICE _complex<Real> make_complexVolatile(const Real REF(a), const Real REF(b)){
    _complex<Real> res;
    res.val.x = a;
    res.val.y = b;
    return res;
  }



  // return constant number one
  static M_HOSTDEVICE  _complex<Real> one() {
    return make_complex((Real)1.0, (Real)0.0);
  }
  // return constant number one
  static M_HOSTDEVICE  _complex<Real> unit() {
    return make_complex((Real)1.0, (Real)0.0);
  }

  // return constant number zero
  static M_HOSTDEVICE  _complex<Real> zero() {
    return make_complex((Real)0.0, (Real)0.0);
  }


  // return constant number I
  static M_HOSTDEVICE  _complex<Real> I() {
    return make_complex((Real)0.0, (Real)1.0);
  }

  // print matrix contents 
  M_HOSTDEVICE void print() {
    printf("%.10e + %.10ej\n", val.x, val.y);
  }

    
  friend M_HOST std::ostream& operator<<( std::ostream& out, _complex<Real> M ) {
    //cout << std::scientific;
    //out << std::setprecision(14);
    out << M.real() << '\t' << M.imag();
    return out;
  }



    friend M_HOSTDEVICE _complex<Real> conj(const _complex<Real>& z){
      return make_complex(z.real(), -z.imag());
    }
    friend M_HOSTDEVICE Real abs(const _complex<Real>& z){
      return sqrt(z.real()*z.real()+z.imag()*z.imag());
    }
    friend M_HOSTDEVICE Real abs2(const _complex<Real>& z){
      return z.real()*z.real()+z.imag()*z.imag();
    }
    friend M_HOSTDEVICE Real arg(const _complex<Real>& z){
      return atan2(z.imag(), z.real());
    }


    friend M_HOSTDEVICE _complex<Real> sin(const _complex<Real>& z){
      const Real x = z.val.x;
      const Real y = z.val.y;
      return make_complex(sin(x) * cosh(y), cos(x) * sinh(y));
    }
    friend M_HOSTDEVICE _complex<Real> asin(const _complex<Real>& z){
	  const _complex<Real>  t(-z.imag(), z.real());
	  t = asinh(t);
      return make_complex(t.imag(), -t.real());
    }

    friend M_HOSTDEVICE _complex<Real> sinh(const _complex<Real>& z){
	  return make_complex( sinh(z.real())*cos(z.imag()), cosh(z.real())*sin(z.imag()));
	}
    friend M_HOSTDEVICE _complex<Real> asinh(const _complex<Real>& z){
	  const _complex<Real>  t((z.real() - z.imag()) * (z.real() + z.imag()) + 1., 2. * z.real() * z.imag());
	  t = sqrt(t);
      return log(t+z);
	}

    friend M_HOSTDEVICE _complex<Real> cos(const _complex<Real>& z){
      const Real x = z.val.x;
      const Real y = z.val.y;
      return make_complex(cos(x) * cosh(y), -sin(x) * sinh(y));
    }
    friend M_HOSTDEVICE _complex<Real> acos(const _complex<Real>& z){
	  const _complex<Real>  t = asin(z);
	  const Real pi_2 = 1.5707963267948966192313216916397514L;
      return make_complex(pi_2-t.real(), -t.imag());
    }

    friend M_HOSTDEVICE _complex<Real> cosh(const _complex<Real>& z){
	  return make_complex( cosh(z.real())*cos(z.imag()), sinh(z.real())*sin(z.imag()));
	}
    friend M_HOSTDEVICE _complex<Real> acosh(const _complex<Real>& z){
	  const _complex<Real>  t((z.real() - z.imag()) * (z.real() + z.imag()) - 1., 2. * z.real() * z.imag());
	  t = sqrt(t);
      return log(t+z);
	}

    friend M_HOSTDEVICE _complex<Real> tan(const _complex<Real>& z){
		return sin(z)/cos(z);
    }
    friend M_HOSTDEVICE _complex<Real> atan(const _complex<Real>& z){
		Real r2 = z.real() * z.real();
      	Real x = 1.0 - r2 - z.imag() * z.imag();
      	Real num = z.imag() + 1.0;
      	Real den = z.imag() - 1.0;
        num = r2 + num * num;
      	den = r2 + den * den;
		return _complex<Real>(0.5 * atan2(2.0 * z.real(), x), 0.25 * log(num / den));
    }

    friend M_HOSTDEVICE _complex<Real> tanh(const _complex<Real>& z){
		return sinh(z)/cosh(z);
    }
    friend M_HOSTDEVICE _complex<Real> atanh(const _complex<Real>& z){
		const Real i2 = z.imag() * z.imag();
		const Real x = 1. - i2 - z.real() * z.real();
		Real num = 1. + z.real();
		Real den = 1. - z.real();
		num = i2 + num * num;
		den = i2 + den * den;
		return _complex<Real>(0.25 * (log(num) - log(den)), 0.5 * atan2(2.0 * z.imag(), x));
    }

    friend M_HOSTDEVICE _complex<Real> sqrt(const _complex<Real>& z){
      const Real x = z.val.x;
      const Real y = z.val.y;
      if (x == 0.) {
          Real t = sqrt(std::abs(y) / 2.);
          return _complex<Real>(t, y < 0. ? -t : t);
      }
      else {
          Real t = sqrt(2 * (z.abs() + std::abs(x)));
          Real u = t / 2;
          return x > 0. ? _complex<Real>(u, y / t) : _complex<Real>(std::abs(y) / t, y < 0. ? -u : u);
      }
    }

    friend M_HOSTDEVICE _complex<Real> exp(const _complex<Real>& z){
      const Real rho = exp(z.val.x);
	  if(rho < 0.) printf("rho= %.10e\n",rho);
      assert(rho >= 0.);
      const Real theta = z.val.y;
      return make_complex(rho * cos(theta), rho * sin(theta));
    }

    friend M_HOSTDEVICE _complex<Real> log(const _complex<Real>& z){
      return make_complex(log(z.abs()), z.arg());
    }


    friend M_HOSTDEVICE _complex<Real> log10(const _complex<Real>& z){
      return log(z)/log(10.);
    }


    friend M_HOSTDEVICE _complex<Real> polar(const Real &rho, const Real &theta){
      return make_complex(rho * cos(theta), rho * sin(theta));
    }

 

};





//
// Define the common complex types
//
#define	complex		_complex<Real>
typedef _complex< float> complexs;
typedef _complex< double> complexd;


// a possible alternative to a single complex constructor
static M_HOSTDEVICE complexs make_complexs(float a, float b){
  complexs res;
  res.real() = a;
  res.imag() = b;
  return res;
}

// a possible alternative to a single complex constructor
static M_HOSTDEVICE complexs make_complexs(float2 a){
  complexs res;
  res.real() = a.x;
  res.imag() = a.y;
  return res;
}


// a possible alternative to a double complex constructor
static M_HOSTDEVICE complexd make_complexd(double a, double b){
  complexd res;
  res.real() = a;
  res.imag() = b;
  return res;
}

static M_HOSTDEVICE complexd make_complexd(double2 a){
  complexd res;
  res.real() = a.x;
  res.imag() = a.y;
  return res;
}


}





#endif // #ifndef COMPLEX_H
