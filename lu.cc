#define PROBLEM_SIZE 64
#define NUM_MATRICES 100

#define IJK
//#define IJK_PAR
//#define IJK_VEC
//#define IJK_OPT
//#define IJK_SUPER
//#define IKJ
//#define IKJ_VEC
//#define KIJ
//#define KIJ_VEC
//#define KIJ_PAR
//#define KIJ_OPT

void LU_decomp_ijk(const int n, const int lda, double* const A) {
  // LU decomposition without pivoting (Doolittle algorithm)
  // In-place decomposition of form A=LU
  // L is returned below main diagonal of A
  // U is returned at and above main diagonal

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

void LU_decomp_ijk_par(const int n, const int lda, double* const A) {
  // LU decomposition without pivoting (Doolittle algorithm)
  // In-place decomposition of form A=LU
  // L is returned below main diagonal of A
  // U is returned at and above main diagonal

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

void LU_decomp_ijk_vec(const int n, const int lda, double* const A) {
  // LU decomposition without pivoting (Doolittle algorithm)
  // In-place decomposition of form A=LU
  // L is returned below main diagonal of A
  // U is returned at and above main diagonal

  const int cache_line = 8, num_threads = omp_get_max_threads();
  double * holders = (double*)_mm_malloc(sizeof(double)*num_threads*cache_line, 64);

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

void LU_decomp_ijk_opt(const int n, const int lda, double* const A) {
  // LU decomposition without pivoting (Doolittle algorithm)
  // In-place decomposition of form A=LU
  // L is returned below main diagonal of A
  // U is returned at and above main diagonal

  const int cache_line = 8, num_threads = omp_get_max_threads();
  double * holders = (double*)_mm_malloc(sizeof(double)*num_threads*cache_line, 64);

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

void LU_decomp_ijk_super(const int n, const int lda, double* const A) {
  // LU decomposition without pivoting (Doolittle algorithm)
  // In-place decomposition of form A=LU
  // L is returned below main diagonal of A
  // U is returned at and above main diagonal
  const int cache_line = 8, num_threads =  sysconf(_SC_NPROCESSORS_ONLN);
  omp_set_num_threads(num_threads);
  double * ATran = (double*)_mm_malloc(sizeof(double)*n*lda + 64, 64);
#pragma omp parallel for
  for (int rowCtr = 0; rowCtr < n; ++rowCtr) {
#pragma simd
#pragma ivdep
    for (int colCtr = 0; colCtr < n; ++colCtr) {
      ATran[colCtr*lda + rowCtr] = A[rowCtr*lda + colCtr];
    }
  }
  double * holders = (double*)_mm_malloc(sizeof(double)*num_threads*cache_line, 64);

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

void LU_decomp_ikj(const int n, const int lda, double* const A) {
  // LU decomposition without pivoting (Doolittle algorithm)
  // In-place decomposition of form A=LU
  // L is returned below main diagonal of A
  // U is returned at and above main diagonal

  for (int i = 1; i < n; i++) {
    for (int k = 0; k < i; k++) {
      A[i*lda + k] = A[i*lda + k]/A[k*lda + k];
#pragma novector
      for (int j = k+1; j < n; j++)
        A[i*lda + j] -= A[i*lda+k]*A[k*lda + j];
    }
  }
}

void LU_decomp_ikj_vec(const int n, const int lda, double* const A) {
  // LU decomposition without pivoting (Doolittle algorithm)
  // In-place decomposition of form A=LU
  // L is returned below main diagonal of A
  // U is returned at and above main diagonal

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

void LU_decomp_kij(const int n, const int lda, double* const A) {
  // LU decomposition without pivoting (Doolittle algorithm)
  // In-place decomposition of form A=LU
  // L is returned below main diagonal of A
  // U is returned at and above main diagonal

  for (int k = 0; k < n-1; k++) {
    for (int i = k+1; i < n; i++) {
      A[i*lda + k] = A[i*lda + k]/A[k*lda + k];
#pragma novector
      for (int j = k+1; j < n; j++)
        A[i*lda + j] -= A[i*lda+k]*A[k*lda + j];
    }
  }
}

void LU_decomp_kij_vec(const int n, const int lda, double* const A) {
  // LU decomposition without pivoting (Doolittle algorithm)
  // In-place decomposition of form A=LU
  // L is returned below main diagonal of A
  // U is returned at and above main diagonal

  for (int k = 0; k < n-1; k++) {
    for (int i = k+1; i < n; i++) {
      A[i*lda + k] = A[i*lda + k]/A[k*lda + k];
      #pragma simd
      #pragma ivdep
      for (int j = k+1; j < n; j++)
	    A[i*lda + j] -= A[i*lda+k]*A[k*lda + j];
    }
  }
}

void LU_decomp_kij_par(const int n, const int lda, double* const A) {
  // LU decomposition without pivoting (Doolittle algorithm)
  // In-place decomposition of form A=LU
  // L is returned below main diagonal of A
  // U is returned at and above main diagonal

#pragma omp parallel
{
  for (int k = 0; k < n-1; k++) {
#pragma omp for
    for (int i = k+1; i < n; i++) {
      A[i*lda + k] = A[i*lda + k]/A[k*lda + k];
#pragma novector
      for (int j = k+1; j < n; j++)
        A[i*lda + j] -= A[i*lda+k]*A[k*lda + j];
    }
  }
}
}

void LU_decomp_kij_opt(const int n, const int lda, double* const A) {
  // LU decomposition without pivoting (Doolittle algorithm)
  // In-place decomposition of form A=LU
  // L is returned below main diagonal of A
  // U is returned at and above main diagonal

#pragma omp parallel
{
  for (int k = 0; k < n-1; k++) {
#pragma omp for
    for (int i = k+1; i < n; i++) {
      A[i*lda + k] = A[i*lda + k]/A[k*lda + k];
      #pragma simd
      #pragma ivdep
      for (int j = k+1; j < n; j++)
	    A[i*lda + j] -= A[i*lda+k]*A[k*lda + j];
    }
  }
}
}
