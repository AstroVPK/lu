#include <mathimf.h>
#include <cstdio>
#include <omp.h>
#include <cassert>
#include <unistd.h>

#include "advisor-annotate.h"

#include "lu.cc"

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
    printf("ERROR: LU is not equal to A (deviation1=%e)!\n", deviation1);
    //    exit(1);
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
  const int n=PROBLEM_SIZE;
  const int lda=n+16;
  const int nMatrices=NUM_MATRICES;
  const double HztoPerf = 1e-9*2.0/3.0*double(n*n*static_cast<double>(lda))*nMatrices;

  const size_t containerSize = sizeof(double)*n*lda+64;
  char* dataA = (char*) _mm_malloc(containerSize*nMatrices, 64);
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

  // Perform benchmark
  printf("LU decomposition of %d matrices of size %dx%d on %s...\n\n",
     nMatrices, n, n,
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
#elif defined KIJ_PAR
  printf("Dolittle Algorithm (kij version - parallelized)\n");
#elif defined KIJ_VEC
  printf("Dolittle Algorithm (kij version - vectorized)\n");
#elif defined KIJ_OPT
  printf("Dolittle Algorithm (kij version - vectorized + parallelized)\n");
#endif

  double rate = 0, dRate = 0; // Benchmarking data
  const int nTrials = 10;
  const int skipTrials = 3; // First step is warm-up on Xeon Phi coprocessor
  printf("\033[1m%5s %10s %8s\033[0m\n", "Trial", "Time, s", "GFLOP/s");
  for (int trial = 1; trial <= nTrials; trial++) {

    const double tStart = omp_get_wtime(); // Start timing
    for (int m = 0; m < nMatrices; m++) {
      double* matrixA = (double*)(&dataA[m*containerSize]);
#if defined IJK
        LU_decomp_ijk(n, lda, matrixA);
#elif defined IJK_PAR
        LU_decomp_ijk_par(n, lda, matrixA);
#elif defined IJK_VEC
        LU_decomp_ijk_vec(n, lda, matrixA);
#elif defined IJK_OPT
        LU_decomp_ijk_opt(n, lda, matrixA);
#elif defined IJK_SUPER
        LU_decomp_ijk_super(n, lda, matrixA);
#elif defined IKJ
        LU_decomp_ikj(n, lda, matrixA);
#elif defined IKJ_VEC
        LU_decomp_ikj_vec(n, lda, matrixA);
#elif defined KIJ
        LU_decomp_kij(n, lda, matrixA);
#elif defined KIJ_VEC
        LU_decomp_kij_vec(n, lda, matrixA);
#elif defined KIJ_PAR
        LU_decomp_kij_par(n, lda, matrixA);
#elif defined KIJ_OPT
        LU_decomp_kij_opt(n, lda, matrixA);
#endif
    }
    const double tEnd = omp_get_wtime(); // End timing

    if (trial == 1) VerifyResult(n, lda, (double*)(&dataA[0]), referenceMatrix);

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

  _mm_free(dataA);
  _mm_free(referenceMatrix);

}
