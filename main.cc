#include <cmath>
#include <mathimf.h>
#include <cstdio>
#include <omp.h>
#include <cassert>
#include <unistd.h>
#ifdef __HBM__
#include <hbwmalloc.h>
#endif

#include "advisor-annotate.h"

#define PROBLEM_SIZE 128
//#define PROBLEM_SIZE 256
//#define PROBLEM_SIZE 2048 
#define NUM_MATRICES 100
#define NUM_TRIALS 10
#define TILE_SIZE 8
//#define TILE_SIZE 32

//#define IJK
//#define IJK_PAR
//#define IJK_VEC
//#define IJK_OPT
//#define IJK_SUPER
//#define IKJ
//#define IKJ_VEC
//#define KIJ
//#define KIJ_REG
//#define KIJ_VEC
//#define KIJ_VEC_REG
//#define KIJ_ANNOTI
//#define KIJ_ANNOTK
//#define KIJ_PAR
//#define KIJ_PAR_REG
//#define KIJ_OPT
#define KIJ_OPT_REG

void LU_decomp_ijk(const int n, const int lda, double* const A, double *scratch) {
  // LU decomposition without pivoting (Doolittle algorithm)
  // In-place decomposition of form A=LU
  // L is returned below main diagonal of A
  // U is returned at and above main diagonal

  __assume_aligned(A, 64);
  __assume_aligned(scratch, 64);

  for (int i = 0; i < n; ++i) {
    for (int j = i; j < n; ++j) {
      for (int k = 0; k < i; ++k) {
        A[i*lda + j] -= A[i*lda + k]*A[k*lda + j];
      }
    }
    for (int j = i + 1; j < n; ++j) {
      for (int k = 0; k < i; ++k) {
        A[j*lda + i] -= A[j*lda + k]*A[k*lda + i];
      }
      A[j*lda + i] /= A[i*lda + i];
    }
  }
}

void LU_decomp_ijk_par(const int n, const int lda, double* const A, double *scratch) {
  // LU decomposition without pivoting (Doolittle algorithm)
  // In-place decomposition of form A=LU
  // L is returned below main diagonal of A
  // U is returned at and above main diagonal

  __assume_aligned(A, 64);
  __assume_aligned(scratch, 64);

  for (int i = 0; i < n; ++i) {
#pragma omp parallel for
    for (int j = i; j < n; ++j) {
      for (int k = 0; k < i; ++k) {
        A[i*lda + j] -= A[i*lda + k]*A[k*lda + j];
      }
    }
#pragma omp parallel for
    for (int j = i + 1; j < n; ++j) {
      for (int k = 0; k < i; ++k) {
        A[j*lda + i] -= A[j*lda + k]*A[k*lda + i];
      }
      A[j*lda + i] /= A[i*lda + i];
    }
  }
}

void LU_decomp_ijk_vec(const int n, const int lda, double* const A, double *scratch) {
  // LU decomposition without pivoting (Doolittle algorithm)
  // In-place decomposition of form A=LU
  // L is returned below main diagonal of A
  // U is returned at and above main diagonal

  __assume_aligned(A, 64);
  __assume_aligned(scratch, 64);

  const int cache_line = 8, num_threads = omp_get_max_threads();
  double *holders = &scratch[0];

  __assume_aligned(holders, 64);

  for (int i = 0; i < n; ++i) {
    for (int j = i; j < n; ++j) {
      int tid = omp_get_thread_num();
      holders[cache_line*tid] = A[i*lda + j];
#pragma simd
#pragma ivdep
      for (int k = 0; k < i; ++k) {
        holders[cache_line*tid] -= A[i*lda + k]*A[k*lda + j];
      }
      A[i*lda + j] = holders[cache_line*tid];
    }
    for (int j = i + 1; j < n; ++j) {
      int tid = omp_get_thread_num();
      holders[cache_line*tid] = A[j*lda + i];
#pragma simd
#pragma ivdep
      for (int k = 0; k < i; ++k) {
        holders[cache_line*tid] -= A[j*lda + k]*A[k*lda + i];
      }
      A[j*lda + i] = holders[cache_line*tid]/A[i*lda + i];
    }
  }
  _mm_free(holders);
}

void LU_decomp_ijk_opt(const int n, const int lda, double* const A, double *scratch) {
  // LU decomposition without pivoting (Doolittle algorithm)
  // In-place decomposition of form A=LU
  // L is returned below main diagonal of A
  // U is returned at and above main diagonal

  __assume_aligned(A, 64);
  __assume_aligned(scratch, 64);

  const int cache_line = 8, num_threads = omp_get_max_threads();
  double * holders = &scratch[0];

  __assume_aligned(holders, 64);

#pragma omp parallel
{
  for (int i = 0; i < n; ++i) {
#pragma omp for schedule(static)
    for (int j = i; j < n; ++j) {
      int tid = omp_get_thread_num();
      holders[cache_line*tid] = A[i*lda + j];
#pragma simd
#pragma ivdep
      for (int k = 0; k < i; ++k) {
        holders[cache_line*tid] -= A[i*lda + k]*A[k*lda + j];
      }
      A[i*lda + j] = holders[cache_line*tid];
    }
#pragma omp for schedule(static)
    for (int j = i + 1; j < n; ++j) {
      int tid = omp_get_thread_num();
      holders[cache_line*tid] = A[j*lda + i];
#pragma simd
#pragma ivdep
      for (int k = 0; k < i; ++k) {
        holders[cache_line*tid] -= A[j*lda + k]*A[k*lda + i];
      }
      A[j*lda + i] = holders[cache_line*tid]/A[i*lda + i];
    }
  }
}
  _mm_free(holders);
}

void LU_decomp_ijk_super(const int n, const int lda, double* const A, double *scratch) {
  // LU decomposition without pivoting (Doolittle algorithm)
  // In-place decomposition of form A=LU
  // L is returned below main diagonal of A
  // U is returned at and above main diagonal

  __assume_aligned(A, 64);
  __assume_aligned(scratch, 64);

  const int cache_line = 8, num_threads =  sysconf(_SC_NPROCESSORS_ONLN);
  omp_set_num_threads(num_threads);

  double *holders = &scratch[0];
  double *ATran = &scratch[num_threads*cache_line];

  __assume_aligned(holders, 64);
  __assume_aligned(ATran, 64);

#pragma omp parallel for
  for (int rowCtr = 0; rowCtr < n; ++rowCtr) {
#pragma simd
#pragma ivdep
    for (int colCtr = 0; colCtr < n; ++colCtr) {
      ATran[colCtr*lda + rowCtr] = A[rowCtr*lda + colCtr];
    }
  }

#pragma omp parallel
{
  for (int i = 0; i < n; ++i) {
#pragma omp for schedule(static)
    for (int j = i; j < n; ++j) {
      int tid = omp_get_thread_num();
      holders[cache_line*tid] = A[i*lda + j];
#pragma simd
#pragma ivdep
      for (int k = 0; k < i; ++k) {
        holders[cache_line*tid] -= A[i*lda + k]*ATran[j*lda + k];
      }
      A[i*lda + j] = holders[cache_line*tid];
      ATran[j*lda + i] = A[i*lda + j];
    }
#pragma omp for schedule(static)
    for (int j = i + 1; j < n; ++j) {
      int tid = omp_get_thread_num();
      holders[cache_line*tid] = A[j*lda + i];
#pragma simd
#pragma ivdep
      for (int k = 0; k < i; ++k) {
        holders[cache_line*tid] -= A[j*lda + k]*ATran[i*lda + k];
      }
      A[j*lda + i] = holders[cache_line*tid]/A[i*lda + i];
      ATran[i*lda + j] = A[j*lda + i];
    }
  }
}
    _mm_free(holders);
    _mm_free(ATran);
}

void LU_decomp_ikj(const int n, const int lda, double* const A, double *scratch) {
  // LU decomposition without pivoting (Doolittle algorithm)
  // In-place decomposition of form A=LU
  // L is returned below main diagonal of A
  // U is returned at and above main diagonal

  __assume_aligned(A, 64);
  __assume_aligned(scratch, 64);

  for (int i = 1; i < n; i++) {
    for (int k = 0; k < i; k++) {
      A[i*lda + k] = A[i*lda + k]/A[k*lda + k];
#pragma novector
      for (int j = k+1; j < n; j++) 
        A[i*lda + j] -= A[i*lda+k]*A[k*lda + j];
    }
  }
}

void LU_decomp_ikj_vec(const int n, const int lda, double* const A, double *scratch) {
  // LU decomposition without pivoting (Doolittle algorithm)
  // In-place decomposition of form A=LU
  // L is returned below main diagonal of A
  // U is returned at and above main diagonal

  __assume_aligned(A, 64);
  __assume_aligned(scratch, 64);

  for (int i = 1; i < n; i++) {
    for (int k = 0; k < i; k++) {
      A[i*lda + k] = A[i*lda + k]/A[k*lda + k];
#pragma simd
#pragma ivdep
	  for (int j = k+1; j < n; j++) 
        A[i*lda + j] -= A[i*lda+k]*A[k*lda + j];
    }
  }
}

void LU_decomp_kij(const size_t n, const size_t lda, double* const A, double *scratch) {
  // LU decomposition without pivoting (Doolittle algorithm)
  // In-place decomposition of form A=LU
  // L is returned below main diagonal of A
  // U is returned at and above main diagonal

  __assume_aligned(A, 64);
  __assume_aligned(scratch, 64);

  for (size_t k = 0; k < n; k++) {
    const double recAkk = 1.0/A[k*lda + k];
    for (size_t i = k + 1; i < n; i++) {
      A[i*lda + k] = A[i*lda + k]*recAkk;
#pragma novector
      for (size_t j = k + 1; j < n; j++)
        A[i*lda + j] -= A[i*lda + k]*A[k*lda + j];
    }
  }
}

void LU_decomp_kij_reg(const size_t n, const size_t lda, double* const A, double *scratch) {
  // LU decomposition without pivoting (Doolittle algorithm)
  // In-place decomposition of form A=LU
  // L is returned below main diagonal of A
  // U is returned at and above main diagonal

  __assume_aligned(A, 64);
  __assume_aligned(scratch, 64);

  const int tile = TILE_SIZE;

  for (size_t i = 0; i < n; ++i) {
#pragma novector
    for (size_t j = 0; j < n; ++j) {
      scratch[i*lda + j] = 0.0;
    }
    scratch[i*lda + i] = 1.0;
  }

  for (size_t k = 0; k < n; k++) {
    const size_t jmin = k - k%tile;
    const double recAkk = 1.0/A[k*lda + k];
    for (size_t i = k + 1; i < n; i++) {
      scratch[i*lda + k] = A[i*lda + k]*recAkk;
#pragma novector
      for (size_t j = jmin; j < n; j++) {
	    A[i*lda + j] -= scratch[i*lda + k]*A[k*lda + j];
      }
    }
  }
  for (size_t i = 0; i < n; ++i) {
#pragma novector
    for (size_t j = 0; j < i; ++j) {
      A[i*lda + j] = scratch[i*lda + j];
    }
  }
}

void LU_decomp_kij_vec(const size_t n, const size_t lda, double* const A, double *scratch) {
  // LU decomposition without pivoting (Doolittle algorithm)
  // In-place decomposition of form A=LU
  // L is returned below main diagonal of A
  // U is returned at and above main diagonal

  __assume_aligned(A, 64);
  __assume_aligned(scratch, 64);

  for (size_t k = 0; k < n; k++) {
    const double recAkk = 1.0/A[k*lda + k];
    for (size_t i = k + 1; i < n; i++) {
      A[i*lda + k] = A[i*lda + k]*recAkk;
#pragma simd
#pragma ivdep
      for (size_t j = k + 1; j < n; j++)
	    A[i*lda + j] -= A[i*lda + k]*A[k*lda + j];
    }
  }
}

void LU_decomp_kij_vec_reg(const size_t n, const size_t lda, double* const A, double *scratch) {
  // LU decomposition without pivoting (Doolittle algorithm)
  // In-place decomposition of form A=LU
  // L is returned below main diagonal of A
  // U is returned at and above main diagonal

  __assume_aligned(A, 64);
  __assume_aligned(scratch, 64);

  const int tile = TILE_SIZE;

  for (size_t i = 0; i < n; ++i) {
#pragma simd
#pragma ivdep
    for (size_t j = 0; j < n; ++j) {
      scratch[i*lda + j] = 0.0;
    }
    scratch[i*lda + i] = 1.0;
  }

  for (size_t k = 0; k < n; k++) {
    const size_t jmin = k - k%tile;
    const double recAkk = 1.0/A[k*lda + k];
    for (size_t i = k + 1; i < n; i++) {
      scratch[i*lda + k] = A[i*lda + k]*recAkk;
#pragma vector aligned
#pragma simd aligned
#pragma ivdep
      for (size_t j = jmin; j < n; j++) {
	    A[i*lda + j] -= scratch[i*lda + k]*A[k*lda + j];
      }
    }
  }

  for (size_t i = 0; i < n; ++i) {
#pragma simd
#pragma ivdep
    for (size_t j = 0; j < i; ++j) {
      A[i*lda + j] = scratch[i*lda + j];
    }
  }
}

void LU_decomp_kij_annoti(const size_t n, const size_t lda, double* const A, double *scratch) {
  // LU decomposition without pivoting (Doolittle algorithm)
  // In-place decomposition of form A=LU
  // L is returned below main diagonal of A
  // U is returned at and above main diagonal

  __assume_aligned(A, 64);
  __assume_aligned(scratch, 64);

  const int tile = TILE_SIZE;

ANNOTATE_SITE_BEGIN(Init);
  for (size_t i = 0; i < n; ++i) {
ANNOTATE_ITERATION_TASK(initLoop);
#pragma ivdep
#pragma simd
    for (size_t j = 0; j < n; ++j) {
      scratch[i*lda + j] = 0.0;
    }
    scratch[i*lda + i] = 1.0;
  }
ANNOTATE_SITE_END(Init);

  for (size_t k = 0; k < n; k++) {
    const size_t jmin = k - k%tile;
    const double recAkk = 1.0/A[k*lda + k];;
ANNOTATE_SITE_BEGIN(Decompose);
    for (size_t i = k + 1; i < n; i++) {
ANNOTATE_ITERATION_TASK(iLoop);
      scratch[i*lda + k] = A[i*lda + k]*recAkk;
#pragma vector aligned
#pragma ivdep
#pragma simd
      for (size_t j = jmin; j < n; j++) {
	    A[i*lda + j] -= scratch[i*lda + k]*A[k*lda + j];
      }
    }
ANNOTATE_SITE_END(Decompose);
  }

ANNOTATE_SITE_BEGIN(Copy);
  for (size_t i = 0; i < n; ++i) {
ANNOTATE_ITERATION_TASK(copyLoop);
#pragma ivdep
#pragma simd
    for (size_t j = 0; j < i; ++j) {
      A[i*lda + j] = scratch[i*lda + j];
    }
  }
ANNOTATE_SITE_END(Copy);
}

void LU_decomp_kij_annotk(const size_t n, const size_t lda, double* const A, double *scratch) {
  // LU decomposition without pivoting (Doolittle algorithm)
  // In-place decomposition of form A=LU
  // L is returned below main diagonal of A
  // U is returned at and above main diagonal

  __assume_aligned(A, 64);
  __assume_aligned(scratch, 64);

  const int tile = TILE_SIZE;

ANNOTATE_SITE_BEGIN(Init);
  for (size_t i = 0; i < n; ++i) {
ANNOTATE_ITERATION_TASK(initLoop);
#pragma ivdep
#pragma simd
    for (size_t j = 0; j < n; ++j) {
      scratch[i*lda + j] = 0.0;
    }
    scratch[i*lda + i] = 1.0;
  }
ANNOTATE_SITE_END(Init);

ANNOTATE_SITE_BEGIN(Decompose);
  for (size_t k = 0; k < n; k++) {
ANNOTATE_ITERATION_TASK(kLoop);
    const size_t jmin = k - k%tile;
    const double recAkk = 1.0/A[k*lda + k];
    for (size_t i = k + 1; i < n; i++) {
      scratch[i*lda + k] = A[i*lda + k]*recAkk;
#pragma vector aligned
#pragma ivdep
#pragma simd
      for (size_t j = jmin; j < n; j++) {
	    A[i*lda + j] -= scratch[i*lda + k]*A[k*lda + j];
      }
    }
  }
ANNOTATE_SITE_END(Decompose);

ANNOTATE_SITE_BEGIN(Copy);
  for (size_t i = 0; i < n; ++i) {
ANNOTATE_ITERATION_TASK(copyLoop);
#pragma ivdep
#pragma simd
    for (size_t j = 0; j < i; ++j) {
      A[i*lda + j] = scratch[i*lda + j];
    }
  }
ANNOTATE_SITE_END(Copy);
}

void LU_decomp_kij_par(const size_t n, const size_t lda, double* const A, double *scratch) {
  // LU decomposition without pivoting (Doolittle algorithm)
  // In-place decomposition of form A=LU
  // L is returned below main diagonal of A
  // U is returned at and above main diagonal

  __assume_aligned(A, 64);
  __assume_aligned(scratch, 64);

#pragma omp parallel
{
#pragma omp for
  for (size_t k = 0; k < n; ++k) {
    const double recAkk = 1.0/A[k*lda + k];
//#pragma omp for
    for (size_t i = k + 1; i < n; ++i) {
      A[i*lda + k] = A[i*lda + k]*recAkk;
#pragma novector
      for (size_t j = k + 1; j < n; ++j) {
        A[i*lda + j] -= A[i*lda+k]*A[k*lda + j];
      }
    }
  }
}
}

void LU_decomp_kij_par_reg(const size_t n, const size_t lda, double* const A, double *scratch) {
  // LU decomposition without pivoting (Doolittle algorithm)
  // In-place decomposition of form A=LU
  // L is returned below main diagonal of A
  // U is returned at and above main diagonal

  __assume_aligned(A, 64);
  __assume_aligned(scratch, 64);

  const int tile = TILE_SIZE;

#pragma omp parallel
{
#pragma omp for
  for (size_t i = 0; i < n; ++i) {
#pragma novector
    for (size_t j = 0; j < n; ++j) {
      scratch[i*lda + j] = 0.0;
    }
    scratch[i*lda + i] = 1.0;
  }

  for (size_t k = 0; k < n; k++) {
    const size_t jmin = k - k%tile;
    const double recAkk = 1.0/A[k*lda + k];
#pragma omp for
    for (size_t i = k + 1; i < n; i++) {
      scratch[i*lda + k] = A[i*lda + k]*recAkk;
#pragma novector
      for (size_t j = jmin; j < n; j++) {
	    A[i*lda + j] -= scratch[i*lda + k]*A[k*lda + j];
      }
    }
  }

#pragma omp for
  for (size_t i = 0; i < n; ++i) {
#pragma novector
    for (size_t j = 0; j < i; ++j) {
      A[i*lda + j] = scratch[i*lda + j];
    }
  }
}
}

void LU_decomp_kij_opt(const size_t n, const size_t lda, double* const A, double *scratch) {
  // LU decomposition without pivoting (Doolittle algorithm)
  // In-place decomposition of form A=LU
  // L is returned below main diagonal of A
  // U is returned at and above main diagonal

  __assume_aligned(A, 64);
  __assume_aligned(scratch, 64);

#pragma omp parallel
{
  for (size_t k = 0; k < n; ++k) {
    const double recAkk = 1.0/A[k*lda + k];
#pragma omp for
    for (size_t i = k + 1; i < n; ++i) {
      A[i*lda + k] = A[i*lda + k]*recAkk;
#pragma simd
#pragma ivdep
      for (size_t j = k + 1; j < n; ++j)
	    A[i*lda + j] -= A[i*lda + k]*A[k*lda + j];
    }
  }
}
}

void LU_decomp_kij_opt_reg(const size_t n, const size_t lda, double* const A, double *scratch) {
  // LU decomposition without pivoting (Doolittle algorithm)
  // In-place decomposition of form A=LU
  // L is returned below main diagonal of A
  // U is returned at and above main diagonal

  __assume_aligned(A, 64);
  __assume_aligned(scratch, 64);

  const int tile = TILE_SIZE;

#pragma omp parallel
{
#pragma omp for
  for (size_t i = 0; i < n; ++i) {
#pragma ivdep
#pragma simd
    for (size_t j = 0; j < n; ++j) {
      scratch[i*lda + j] = 0.0;
    }
    scratch[i*lda + i] = 1.0;
  }

  for (size_t k = 0; k < n; k++) {
    const size_t jmin = k - k%tile;
    const double recAkk = 1.0/A[k*lda + k];
#pragma omp for
    for (size_t i = k + 1; i < n; i++) {
      scratch[i*lda + k] = A[i*lda + k]*recAkk;
#pragma vector aligned
#pragma ivdep
#pragma simd
      for (size_t j = jmin; j < n; j++) {
	    A[i*lda + j] -= scratch[i*lda + k]*A[k*lda + j];
      }
    }
  }

#pragma omp for
  for (size_t i = 0; i < n; ++i) {
#pragma ivdep
#pragma simd
    for (size_t j = 0; j < i; ++j) {
      A[i*lda + j] = scratch[i*lda + j];
    }
  }
}
}


/*****************************************************************************/

void VerifyResult(const int n, const int lda, double* LU, double* refA) {

  // Verifying that A=LU
  double *A = static_cast<double*>(_mm_malloc(n*lda*sizeof(double), 64));
  double *L = static_cast<double*>(_mm_malloc(n*lda*sizeof(double), 64));
  double *U = static_cast<double*>(_mm_malloc(n*lda*sizeof(double), 64));
  for (size_t i = 0, arrSize = n*lda; i < arrSize; ++i) {
    A[i] = 0.0f;
      L[i] = 0.0f;
      U[i] = 0.0f;
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < i; j++)
      L[i*lda + j] = LU[i*lda + j];
    L[i*lda+i] = 1.0f;
    for (int j = i; j < n; j++)
      U[i*lda + j] = LU[i*lda + j];
  }
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++)
    A[i*lda + j] += L[i*lda + k]*U[k*lda + j];

  double deviation1 = 0.0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      deviation1 += (refA[i*lda+j] - A[i*lda+j])*(refA[i*lda+j] - A[i*lda+j]);
    }
  }
  deviation1 /= (double)(n*lda);
  if (std::isnan(deviation1) || (deviation1 > 1.0e-2)) {
    printf("ERROR: LU is not equal to A (deviation1 = %e)!\n", deviation1);
    exit(1);
} else {
    printf("OK: LU is equal to A (deviation1 = %e)!\n", deviation1);
}

#ifdef VERBOSE
  printf("\n(L-D)+U:\n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      printf("%10.3e", LU[i*lda+j]);
    printf("\n");
  }

  printf("\nL:\n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      printf("%10.3e", L[i*lda+j]);
    printf("\n");
  }

  printf("\nU:\n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      printf("%10.3e", U[i*lda+j]);
    printf("\n");
  }

  printf("\nLU:\n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      printf("%10.3e", A[i*lda+j]);
    printf("\n");
  }

  printf("\nA:\n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      printf("%10.3e", refA[i*lda+j]);
    printf("\n");
  }

  printf("deviation1=%e\n", deviation1);
#endif

_mm_free(A);
_mm_free(L);
_mm_free(U);

}

int main(const int argc, const char** argv) {

  // Problem size and other parameters
  int n1;
  const char* s1 = getenv("PROBLEM_SIZE");
  if (s1 != NULL) {
    n1 = atoi(s1);
  } else {
    n1 = PROBLEM_SIZE;
  }
  const size_t n = (size_t)n1;
  const size_t lda = n + 16;

  int nMatrices1;
  const char* s2 = getenv("NUM_MATRICES");
  if (s2 != NULL) {
    nMatrices1 = atoi(s2);
  } else {
    nMatrices1 = NUM_MATRICES;
  }
  const size_t nMatrices = (size_t)nMatrices1;

  const double HztoPerf = 1e-9*2.0/3.0*double(n*n*static_cast<double>(lda))*nMatrices;

  const size_t containerSize = sizeof(double)*n*lda+64;
  char* dataA;
  double* scratch;
  const int cache_line = 8, num_threads = omp_get_max_threads();
#ifdef __HBM__
  printf("Using High Bandwidth Memory!\n");
  hbw_posix_memalign((void**) &dataA, 4096, containerSize*nMatrices);
  hbw_posix_memalign((void**) &scratch, 4096, sizeof(double)*(num_threads*cache_line + n*lda) + 64);
#else
  printf("Not using High Bandwidth Memory!\n");
  dataA = (char*)_mm_malloc(containerSize*nMatrices, 64);
  scratch = (double*)_mm_malloc(sizeof(double)*(num_threads*cache_line + n*lda) + 64, 64);
#endif
  double* referenceMatrix = static_cast<double*>(_mm_malloc(n*lda*sizeof(double), 64));

  // Initialize matrices
  for (int m = 0; m < nMatrices; m++) {
    double* matrix = (double*)(&dataA[m*containerSize]);
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        double sum = 0.0f;
        for (int j = 0; j < n; j++) {
            matrix[i*lda+j] = (double)(i*n+j);
            sum += matrix[i*lda+j];
        }
        sum -= matrix[i*lda+i];
        matrix[i*lda+i] = 2.0f*sum;
    }
    matrix[(n-1)*lda+n] = 0.0f; // Touch just in case
  }
  referenceMatrix[0:n*lda] = ((double*)dataA)[0:n*lda];
#pragma omp parallel for
  for (int i = 0; i < num_threads*cache_line + n*lda + 8; ++i) {
    scratch[i] = 0.0;
  }

  // Perform benchmark
  printf("LU decomposition of %d matrices of size %dx%d on %s...\n\n",
     (int)nMatrices, (int)n, (int)n,
#ifndef __MIC__
     "CPU"
#else
     "MIC"
#endif
     );
#if defined IJK
  printf("Dolittle Algorithm (ijk version - baseline)\n");
#elif defined IJK_PAR
  printf("Dolittle Algorithm (ijk version - parallelized)\n");
#elif defined IJK_VEC
  printf("Dolittle Algorithm (ijk version - vectorized)\n");
#elif defined IJK_OPT
  printf("Dolittle Algorithm (ijk version - vectorized + parallelized)\n");
#elif defined IJK_SUPER
  printf("Dolittle Algorithm (ijk version - vectorized + parallelized + transpose)\n");
#elif defined IKJ
  printf("Dolittle Algorithm (ikj version - baseline)\n");
#elif defined IKJ_VEC
  printf("Dolittle Algorithm (ikj version - vectorized)\n");
#elif defined KIJ
  printf("Dolittle Algorithm (kij version - baseline)\n");
#elif defined KIJ_REG
  printf("Dolittle Algorithm (kij_regularized version - baseline)\n");
#elif defined KIJ_VEC
  printf("Dolittle Algorithm (kij version - vectorized)\n");
#elif defined KIJ_VEC_REG
  printf("Dolittle Algorithm (kij_regularized version - vectorized)\n");
#elif defined KIJ_ANNOTI
  printf("Dolittle Algorithm (kij_regularized version - vectorized) + iLoop Annotations\n");
#elif defined KIJ_ANNOTK
  printf("Dolittle Algorithm (kij_regularized version - vectorized) + kLoop Annotations\n");
#elif defined KIJ_PAR
  printf("Dolittle Algorithm (kij version - parallelized)\n");
#elif defined KIJ_PAR_REG
  printf("Dolittle Algorithm (kij_regularized version - parallelized)\n");
#elif defined KIJ_OPT
  printf("Dolittle Algorithm (kij version - vectorized + parallelized)\n");
#elif defined KIJ_OPT_REG
  printf("Dolittle Algorithm (kij_regularized version - vectorized + parallelized)\n");
#endif

  double rate = 0, dRate = 0; // Benchmarking data
  int nTrials1;
  const char* s3 = getenv("NUM_TRIALS");
  if (s3 != NULL) {
    nTrials1  = atoi(s3);
  } else {
    nTrials1 = NUM_TRIALS;
  }
  const size_t nTrials = (size_t)nTrials1;
  const int skipTrials = 3; // First step is warm-up on Xeon Phi coprocessor
  printf("\033[1m%5s %10s %8s\033[0m\n", "Trial", "Time, s", "GFLOP/s");
  for (int trial = 1; trial <= nTrials; trial++) {

    const double tStart = omp_get_wtime(); // Start timing
    for (int m = 0; m < nMatrices; m++) {
      double* matrixA = (double*)(&dataA[m*containerSize]);
#if defined IJK
        LU_decomp_ijk(n, lda, matrixA, scratch);
#elif defined IJK_PAR
        LU_decomp_ijk_par(n, lda, matrixA, scratch);
#elif defined IJK_VEC
        LU_decomp_ijk_vec(n, lda, matrixA, scratch);
#elif defined IJK_OPT
        LU_decomp_ijk_opt(n, lda, matrixA, scratch);
#elif defined IJK_SUPER
        LU_decomp_ijk_super(n, lda, matrixA, scratch);
#elif defined IKJ
        LU_decomp_ikj(n, lda, matrixA, scratch);
#elif defined IKJ_VEC
        LU_decomp_ikj_vec(n, lda, matrixA, scratch);
#elif defined KIJ
        LU_decomp_kij(n, lda, matrixA, scratch);
#elif defined KIJ_REG
        LU_decomp_kij_reg(n, lda, matrixA, scratch);
#elif defined KIJ_VEC
        LU_decomp_kij_vec(n, lda, matrixA, scratch);
#elif defined KIJ_VEC_REG
        LU_decomp_kij_vec_reg(n, lda, matrixA, scratch);
#elif defined KIJ_ANNOTI
        LU_decomp_kij_annoti(n, lda, matrixA, scratch);
#elif defined KIJ_ANNOTK
        LU_decomp_kij_annotk(n, lda, matrixA, scratch);
#elif defined KIJ_PAR
        LU_decomp_kij_par(n, lda, matrixA, scratch);
#elif defined KIJ_PAR_REG
        LU_decomp_kij_par_reg(n, lda, matrixA, scratch);
#elif defined KIJ_OPT
        LU_decomp_kij_opt(n, lda, matrixA, scratch);
#elif defined KIJ_OPT_REG
        LU_decomp_kij_opt_reg(n, lda, matrixA, scratch);
#endif
    }
    const double tEnd = omp_get_wtime(); // End timing

    //if (trial == 1) VerifyResult(n, lda, (double*)(&dataA[0]), referenceMatrix);

    if (trial > skipTrials) { // Collect statistics
      rate  += HztoPerf/(tEnd - tStart);
      dRate += HztoPerf*HztoPerf/((tEnd - tStart)*(tEnd-tStart));
    }

    printf("%5d %10.3e %8.2f %s\n",
       trial, (tEnd-tStart), HztoPerf/(tEnd-tStart), (trial<=skipTrials?"*":""));
    fflush(stdout);
  }
  rate/=(double)(nTrials-skipTrials);
  dRate=sqrt(dRate/(double)(nTrials-skipTrials)-rate*rate);
  printf("-----------------------------------------------------\n");
  printf("\033[1m%s %4s \033[42m%10.2f +- %.2f GFLOP/s\033[0m\n",
     "Average performance:", "", rate, dRate);
  printf("-----------------------------------------------------\n");
  printf("* - warm-up, not included in average\n\n");

#ifdef __HBM__
  hbw_free((void*)dataA);
  hbw_free((void*)scratch);
#else
  _mm_free(dataA);
  _mm_free(scratch);
#endif
  _mm_free(referenceMatrix);
}
