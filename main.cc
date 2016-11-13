#include <cmath>
#include <cstdio>
#include <omp.h>
#include <cassert>
#include <cilk/cilk.h>
#include <cilk/reducer_opadd.h>

#define VISHAL
//#define VERBOSE
//#define EXAMPLE

void LU_decomp(const int n, const int lda, double* const A) {
    // LU decomposition without pivoting (Doolittle algorithm)
    // In-place decomposition of form A=LU
    // L is returned below main diagonal of A
    // U is returned at and above main diagonal

    for (int i = 1; i < n; i++) {
        for (int k = 0; k < i; k++) {
            A[i*lda + k] = A[i*lda + k]/A[k*lda + k];
            #pragma simd
            #pragma ivdep
            for (int j = k + 1; j < n; j++) {
	            A[i*lda + j] -= A[i*lda + k]*A[k*lda + j];
            }
        }
    }
}

void LU_decomp_vishal(const int n, const int lda, double* const A) {
    // LU decomposition without pivoting (Doolittle algorithm)
    // In-place decomposition of form A=LU
    // L is returned below main diagonal of A
    // U is returned at and above main diagonal
    double invDiag = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < i; ++k) {
            A[i*lda + i] -= A[k*lda + i]*A[i*lda + k];
        }
        invDiag = 1.0/A[i*lda + i];
        #pragma ivdep
        #pragma vector always aligned vecremainder
        #pragma simd
        for (int j = i + 1; j < n; ++j) {
            A[i*lda + j] = A[i*lda + j]*invDiag;
        }
        #pragma omp parallel for default(none) shared(n, lda, A, i)
        for (int j = i + 1; j < n; ++j) {
            for (int k = 0; k < i; ++k) {
                A[j*lda + i] -= A[k*lda + i]*A[j*lda + k];
            }
        }
        #pragma omp parallel for default(none) shared(n, lda, A, i, invDiag)
        for (int j = i + 1; j < n; ++j) {
            for (int k = 0; k < i; ++k) {
                A[i*lda + j] -= A[k*lda + j]*A[i*lda + k]*invDiag;
            }
        }
    }
}

void VerifyResult(const int n, const int lda, double* LU, double* refA) {

    // Verifying that A=LU
    double A[n*lda];
    double L[n*lda];
    double U[n*lda];
    A[:] = 0.0f;
    L[:] = 0.0f;
    U[:] = 0.0f;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            L[i*lda + j] = LU[i*lda + j];
        }
        L[i*lda + i] = 1.0f;
        for (int j = i; j < n; j++) {
            U[i*lda + j] = LU[i*lda + j];
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
  	            A[i*lda + j] += L[i*lda + k]*U[k*lda + j];
            }
        }
    }

    double deviation1 = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            deviation1 += (refA[i*lda + j] - A[i*lda + j])*(refA[i*lda + j] - A[i*lda + j]);
        }
    }
    deviation1 /= (double)(n*lda);
    if (isnan(deviation1) || (deviation1 > 1.0e-2)) {
        printf("ERROR: LU is not equal to A (deviation1=%e)!\n", deviation1);
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
}

void VerifyResult_vishal(const int n, const int lda, double* LU, double* refA) {

    // Verifying that A=LU
    double A[n*lda];
    double L[n*lda];
    double U[n*lda];
    A[:] = 0.0f;
    L[:] = 0.0f;
    U[:] = 0.0f;
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            L[i*lda + j] = LU[i*lda + j];
        }
        L[i*lda + i] = 1.0f;
        for (int j = 0; j <= i; j++) {
            U[i*lda + j] = LU[i*lda + j];
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                A[i*lda + j] += L[k*lda + j]*U[i*lda + k];
            }
        }
    }

    double deviation1 = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            deviation1 += (refA[i*lda + j] - A[i*lda + j])*(refA[i*lda + j] - A[i*lda + j]);
        }
    }
    deviation1 /= (double)(n*lda);
    if (isnan(deviation1) || (deviation1 > 1.0e-2)) {
        printf("ERROR: LU is not equal to A (deviation1=%e)!\n", deviation1);
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
}

template <typename valT>
void viewMatrix(const int n, const int m, const int lda, const valT* const A, const int nStart = 0, const int mStart = 0) {
    //      n: Number of rows to view
    //      m: Number of columns to view
    //    lda: Leading dimension of A
    //      A: matrix to view
    // nStart: Row start location of view
    // mStart: Column start location of view
    for (int colCtr = mStart; colCtr < mStart + m; ++colCtr) {
        for (int rowCtr = nStart; rowCtr < nStart + n; ++rowCtr) {
            printf("%+3.2e ", A[colCtr*lda + rowCtr]);
        }
        printf("\n");
    }
}

void testView() {
    const int n = 8;
    const int lda = n + 1;
    double *A = (double*)_mm_malloc(sizeof(double)*n*lda, 64);
    for (int colCtr = 0; colCtr < n; ++colCtr) {
        double sum = 0.0;
        for (int rowCtr = 0; rowCtr < n; ++rowCtr) {
            A[colCtr*lda + rowCtr] = (double)(colCtr*n + rowCtr);
            sum += (double)(colCtr*n + rowCtr);
        }
        sum -= A[colCtr*lda + colCtr];
        A[colCtr*lda + colCtr] = 2.0*sum;
    }
    A[(n - 1)*lda + n] = 0.0;
    printf("    A    \n");
    printf("---------\n");
    viewMatrix(n, n, lda, A);
    printf("   aux   \n");
    printf("---------\n");
    viewMatrix(1, n, lda, A, n, 0);
}

int main(const int argc, const char** argv) {

    #ifdef EXAMPLE
        testView();
    #endif

    // Problem size and other parameters
    const int n=512;
    const int lda=528;
    const int nMatrices=100;
    const double HztoPerf = 1e-9*2.0/3.0*double(n*n*lda)*nMatrices;

    const size_t containerSize = sizeof(double)*n*lda+64;
    char* dataA = (char*) _mm_malloc(containerSize*nMatrices, 64);
    double referenceMatrix[n*lda];

    // Initialize matrices
    for (int m = 0; m < nMatrices; m++) {
        double* matrix = (double*)(&dataA[m*containerSize]);
        #pragma omp parallel for
        for (int colCtr = 0; colCtr < n; colCtr++) {
            double sum = 0.0f;
            for (int rowCtr = 0; rowCtr < n; rowCtr++) {
                matrix[colCtr*lda + rowCtr] = (double)(colCtr*n + rowCtr);
                sum += matrix[colCtr*lda + rowCtr];
            }
            sum -= matrix[colCtr*lda + colCtr];
            matrix[colCtr*lda + colCtr] = 2.0f*sum;
        }
        matrix[(n - 1)*lda + n] = 0.0f; // Touch just in case
        #ifdef VERBOSE
            printf("    A    \n");
            printf("---------\n");
            viewMatrix(n, n, lda, matrix);
            printf("  Extra  \n");
            printf("---------\n");
            viewMatrix(n, 16, lda, matrix, 0, n);
        #endif
    }
    referenceMatrix[0:n*lda] = ((double*)dataA)[0:n*lda];

    // Perform benchmark
    printf("LU decomposition of %d matrices of size %dx%d on %s...\n\n", nMatrices, n, n,
    #ifndef __MIC__
	   "CPU"
    #else
	   "MIC"
    #endif
	   );

    double rate = 0, dRate = 0; // Benchmarking data
    const int nTrials = 10;
    const int skipTrials = 3; // First step is warm-up on Xeon Phi coprocessor
    printf("\033[1m%5s %10s %8s\033[0m\n", "Trial", "Time, s", "GFLOP/s");
    for (int trial = 1; trial <= nTrials; trial++) {

        const double tStart = omp_get_wtime(); // Start timing
        for (int m = 0; m < nMatrices; m++) {
            double* matrixA = (double*)(&dataA[m*containerSize]);
            #ifdef VISHAL
                LU_decomp_vishal(n, lda, matrixA);
            #else
                LU_decomp(n, lda, matrixA);
            #endif
        }
        const double tEnd = omp_get_wtime(); // End timing

        #ifdef VISHAL
            if (trial == 1) VerifyResult_vishal(n, lda, (double*)(&dataA[0]), referenceMatrix);
        #else
            if (trial == 1) VerifyResult(n, lda, (double*)(&dataA[0]), referenceMatrix);
        #endif

        if (trial > skipTrials) { // Collect statistics
            rate  += HztoPerf/(tEnd - tStart);
            dRate += HztoPerf*HztoPerf/((tEnd - tStart)*(tEnd-tStart));
        }

        printf("%5d %10.3e %8.2f %s\n", trial, (tEnd-tStart), HztoPerf/(tEnd-tStart), (trial<=skipTrials?"*":""));
        fflush(stdout);
    }
    rate/=(double)(nTrials-skipTrials);
    dRate=sqrt(dRate/(double)(nTrials-skipTrials)-rate*rate);
    printf("-----------------------------------------------------\n");
    printf("\033[1m%s %4s \033[42m%10.2f +- %.2f GFLOP/s\033[0m\n", "Average performance:", "", rate, dRate);
    printf("-----------------------------------------------------\n");
    printf("* - warm-up, not included in average\n\n");

    _mm_free(dataA);

}
