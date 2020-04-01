// Note: sin4_taylor and sin4_intrin also supports x outside of [-pi/4,pi/4].
// Please check them out.
//
// Additionally, since sin4_intrin always performs additional computation
// designed to handle the expanded domain, the running time of sin4_intrin is
// expected to be longer than sin4_vector.
//
// Please see the comments in those functions for implementation details.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include "utils.h"
#include "intrin-wrapper.h"

// Headers for intrinsics
#ifdef __SSE__
#include <xmmintrin.h>
#endif
#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __AVX__
#include <immintrin.h>
#endif

// coefficients in the Taylor series expansion of sin(x)
static constexpr double c3  = -1/((double)2*3);
static constexpr double c5  =  1/((double)2*3*4*5);
static constexpr double c7  = -1/((double)2*3*4*5*6*7);
static constexpr double c9  =  1/((double)2*3*4*5*6*7*8*9);
static constexpr double c11 = -1/((double)2*3*4*5*6*7*8*9*10*11);
// sin(x) = x + c3*x^3 + c5*x^5 + c7*x^7 + x9*x^9 + c11*x^11

// coefficients in the Taylor series expansion of cos(x)
static constexpr double c2  = -1/((double)2);
static constexpr double c4  =  1/((double)2*3*4);
static constexpr double c6  = -1/((double)2*3*4*5*6);
static constexpr double c8  =  1/((double)2*3*4*5*6*7*8);
static constexpr double c10 = -1/((double)2*3*4*5*6*7*8*9*10);
static constexpr double c12 =  1/((double)2*3*4*5*6*7*8*9*10*11*12);
// cos(x) = 1 + c2*x^2 + c4*x^4 + c6*x^6 + x8*x^8 + c10*x^10 + c12*x^12

// 4 / pi
static constexpr double pi = 3.1415926535897932384626433832795;
static constexpr double _4_pi = 4. / pi;
static constexpr double pi_2 = pi / 2.;

void sin4_reference(double* sinx, const double* x) {
  for (long i = 0; i < 4; i++) sinx[i] = sin(x[i]);
}

void sin4_taylor(double* sinx, const double* x) {
  for (int i = 0; i < 4; i++) {
    // Domain Expansion:
    //  Let x in [-pi/4, pi/4], we have:
    //   + sin(k*pi+x) = -sin(x) for odd k
    //   + sin(k*pi+x) = sin(x) for even k
    //  The problem is to compute:
    //   + sin(k*pi+x+pi/2) for integer k
    //  Note sin(x+pi/2) = cos(x), hence:
    //   + sin(x+pi/2+k*pi) = -cos(x) for odd k
    //   + sin(x+pi/2+k*pi) = cos(x) for even k
    //  After adding cos(x) calculation for x in [-pi/4, pi/4], we may
    //  calculate any sin(x) by:
    //   + sin(x) = -cos(x-2k*pi+pi/2) for x in [2k*pi-3pi/4, 2k*pi-pi/4]
    //   + sin(x) =  sin(x-2k*pi)      for x in [2k*pi-pi/4,  2k*pi+pi/4]
    //   + sin(x) =  cos(x-2k*pi-pi/2) for x in [2k*pi+pi/4,  2k*pi+3pi/4]
    //   + sin(x) = -sin(x-2k*pi-pi)   for x in [2k*pi+3pi/4, 2k*pi+5pi/4]
    //  To simplify, let t = pi/2, x = k*t+y where k is integer and
    //  -pi/4 < y < pi/4, and we have:
    //   + sin(x) =  sin(y)   for k = 0 (mod 4)
    //   + sin(x) =  cos(y)   for k = 1 = -3 (mod 4)
    //   + sin(x) = -sin(y)   for k = 2 = -2 (mod 4)
    //   + sin(x) = -cos(y)   for k = 3 = -1 (mod 4)
    //  To implement, let t' = pi/4, k' = trunc(x/t') is integer and we have:
    //   | k' = | ... | -4 | -3 | -2 | -1 | 0 | 1 | 2 | 3 | 4 | ... |
    //   | k  = | ... | -2 | -2 | -1 | -1 | 0 | 1 | 1 | 2 | 2 | ... |
    //  Then k can be calculated using k' by:
    //    k = (k' + (k' < 0 ? 0 : 1)) >> 1
    //  And y can be calculating by:
    //    y = x - k*t

    auto x1 = x[i];

    // Calculate k:
    //  ...; -1: (-3pi/2,-pi/2]; 0: (-pi/2,pi/2); 1: [pi/2,3pi/2); ...
    // expand at x = ..., -pi, 0, pi, ..., respectively
    // Note: should have used int64_t, but using int32_t is to stay consistent
    // with sin4_intrin. See sin4_intrin's comments for details.
    auto k = (int32_t) (_4_pi * x1);
    k = (k + (k < 0 ? 0 : 1)) >> 1;

    // Move expansion point to 0 (we always use Maclaurin series)
    // Note: Cody-Waite argument reduction is not implemented.
    x1 -= k * pi_2;
    auto x2 = x1 * x1;

    double s;
    if (k & 1) {
      // Calculate cosine: k mod 4 = 1/3
      s = c12;
      s = s * x2 + c10;
      s = s * x2 + c8;
      s = s * x2 + c6;
      s = s * x2 + c4;
      s = s * x2 + c2;
      s = s * x2 + 1;
    }
    else {
      // Calculate sine: k mod 4 = 0/2
      s = c11;
      s = s * x2 + c9;
      s = s * x2 + c7;
      s = s * x2 + c5;
      s = s * x2 + c3;
      s = s * x2 + 1;
      s *= x1;
    }
    // Fix the sign
    sinx[i] = k & 2 ? -s : s;
  }
}

// Note: both precison and domain update is implemented for AVX2+FMA version.
// The domain update is constrained since I do not have _mm256_cvttpd_epi64()
// or vcvtpd2qq instruction on my machine, which requires AVX512 support so I
// can only use _mm256_cvttpd_epi32() instead.
//
// That means I cannot convert double to int64_t in vector manner. Hence, the
// domain of input is constrained that 4x/pi must be representable within the
// range of int32_t. That is approximately -1686629713 <= x <= 1686629713.
void sin4_intrin(double* sinx, const double* x) {
#if defined(__AVX2__)
  auto x1 = _mm256_load_pd(x);

  // Integer computation (requires AVX2)
  // Calculate k: the domain is constrained here (see above for detail)
  auto k32 = _mm256_cvttpd_epi32(_mm256_mul_pd(x1, _mm256_set1_pd(_4_pi)));
  auto t128i = _mm_cmpgt_epi32(_mm_set1_epi32(0), k32);
  t128i = _mm_andnot_si128(t128i, _mm_set1_epi32(1));
  k32 = _mm_add_epi32(k32, t128i);
  k32 = _mm_srai_epi32(k32, 1);

  auto k64 = _mm256_cvtepi32_epi64(k32);
  auto k = _mm256_cvtepi32_pd(k32);

  // Mask of sin/cos
  auto scmask = _mm256_castsi256_pd(_mm256_slli_epi64(k64, 63));

  // Sign bit
  auto t256i = _mm256_cvtepi32_epi64(_mm_and_si128(k32, _mm_set1_epi32(2)));
  auto sign = _mm256_castsi256_pd(_mm256_slli_epi64(t256i, 62));

  // Adjust x1
  // Note: Cody-Waite argument reduction is not implemented.
#if defined(__FMA__)
  x1 = _mm256_fmadd_pd(k, _mm256_set1_pd(-pi_2), x1);
#else
  x1 = _mm256_sub_pd(x1, _mm256_mul_pd(k, _mm256_set1_pd(pi_2)));
#endif

  auto x2 = _mm256_mul_pd(x1 , x1);

  auto c = _mm256_set1_pd(c12);
  auto s = _mm256_set1_pd(c11);
#if defined(__FMA__)
  c = _mm256_fmadd_pd(c, x2, _mm256_set1_pd(c10));
  c = _mm256_fmadd_pd(c, x2, _mm256_set1_pd(c8 ));
  c = _mm256_fmadd_pd(c, x2, _mm256_set1_pd(c6 ));
  c = _mm256_fmadd_pd(c, x2, _mm256_set1_pd(c4 ));
  c = _mm256_fmadd_pd(c, x2, _mm256_set1_pd(c2 ));
  c = _mm256_fmadd_pd(c, x2, _mm256_set1_pd(1. ));
  s = _mm256_fmadd_pd(s, x2, _mm256_set1_pd(c9 ));
  s = _mm256_fmadd_pd(s, x2, _mm256_set1_pd(c7 ));
  s = _mm256_fmadd_pd(s, x2, _mm256_set1_pd(c5 ));
  s = _mm256_fmadd_pd(s, x2, _mm256_set1_pd(c3 ));
  s = _mm256_fmadd_pd(s, x2, _mm256_set1_pd(1. ));
  s = _mm256_mul_pd(s, x1);
#else
  c = _mm256_add_pd(_mm256_mul_pd(c, x2), _mm256_set1_pd(c10));
  c = _mm256_add_pd(_mm256_mul_pd(c, x2), _mm256_set1_pd(c8 ));
  c = _mm256_add_pd(_mm256_mul_pd(c, x2), _mm256_set1_pd(c6 ));
  c = _mm256_add_pd(_mm256_mul_pd(c, x2), _mm256_set1_pd(c4 ));
  c = _mm256_add_pd(_mm256_mul_pd(c, x2), _mm256_set1_pd(c2 ));
  c = _mm256_add_pd(_mm256_mul_pd(c, x2), _mm256_set1_pd(1. ));
  s = _mm256_add_pd(_mm256_mul_pd(s, x2), _mm256_set1_pd(c9 ));
  s = _mm256_add_pd(_mm256_mul_pd(s, x2), _mm256_set1_pd(c7 ));
  s = _mm256_add_pd(_mm256_mul_pd(s, x2), _mm256_set1_pd(c5 ));
  s = _mm256_add_pd(_mm256_mul_pd(s, x2), _mm256_set1_pd(c3 ));
  s = _mm256_add_pd(_mm256_mul_pd(s, x2), _mm256_set1_pd(1. ));
  s = _mm256_mul_pd(s, x1);
#endif
  auto res = _mm256_blendv_pd(s, c, scmask);
  res = _mm256_xor_pd(res, sign);
  _mm256_store_pd(sinx, res);
#elif defined(__SSE2__)
  constexpr int sse_length = 2;
  for (int i = 0; i < 4; i+=sse_length) {
    __m128d x1, x2, x3;
    x1  = _mm_load_pd(x+i);
    x2  = _mm_mul_pd(x1, x1);
    x3  = _mm_mul_pd(x1, x2);

    __m128d s = x1;
    s = _mm_add_pd(s, _mm_mul_pd(x3 , _mm_set1_pd(c3 )));
    _mm_store_pd(sinx+i, s);
  }
#else
  sin4_reference(sinx, x);
#endif
}

// Note: only precision update is implemented in this function.
// This function does not handle input outside of [-pi/4,pi/4]
void sin4_vector(double* sinx, const double* x) {
  // The Vec class is defined in the file intrin-wrapper.h
  typedef Vec<double,4> Vec4;
  auto x1 = Vec4::LoadAligned(x);
  auto x2 = x1 * x1;

  auto s = Vec4(c11);
  s = s * x2 + c9;
  s = s * x2 + c7;
  s = s * x2 + c5;
  s = s * x2 + c3;
  s = s * x2 + 1;
  s *= x1;
  s.StoreAligned(sinx);
}

double err(double* x, double* y, long N) {
  double error = 0;
  for (long i = 0; i < N; i++) error = std::max(error, fabs(x[i]-y[i]));
  return error;
}

int main() {
  Timer tt;
  long N = 1000000;
  double* x = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_ref = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_taylor = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_intrin = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_vector = (double*) aligned_malloc(N*sizeof(double));
  for (long i = 0; i < N; i++) {
    x[i] = (drand48()-0.5) * M_PI/2; // [-pi/4,pi/4]
    //x[i] = (drand48()-0.5) * M_PI * 100; // also outside of [-pi/4,pi/4]
    sinx_ref[i] = 0;
    sinx_taylor[i] = 0;
    sinx_intrin[i] = 0;
    sinx_vector[i] = 0;
  }

  tt.tic();
  // Note: my compiler tends to optmize this outer loop and eliminate it.
  // volatile is added to prevent this optimization.
  for (volatile long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_reference(sinx_ref+i, x+i);
    }
  }
  printf("Reference time: %6.4f\n", tt.toc());

  tt.tic();
  for (volatile long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_taylor(sinx_taylor+i, x+i);
    }
  }
  printf("Taylor time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref, sinx_taylor, N));

  tt.tic();
  for (volatile long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_intrin(sinx_intrin+i, x+i);
    }
  }
  printf("Intrin time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref, sinx_intrin, N));

  tt.tic();
  for (volatile long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_vector(sinx_vector+i, x+i);
    }
  }
  printf("Vector time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref, sinx_vector, N));

  aligned_free(x);
  aligned_free(sinx_ref);
  aligned_free(sinx_taylor);
  aligned_free(sinx_intrin);
  aligned_free(sinx_vector);
}

