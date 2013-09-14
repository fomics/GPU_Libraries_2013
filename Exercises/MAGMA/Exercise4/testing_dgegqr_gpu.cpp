/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @generated d Fri Sep 13 09:22:18 2013
       @author Stan Tomov

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

extern "C" magma_int_t
magma_dgegqr_gpu( magma_int_t m, magma_int_t n,
                  double *dA,   magma_int_t ldda,
                  double *dwork, double *work,
                  magma_int_t *info );


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dgegqr
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double           e1, e2, work[1];
    double  c_neg_one = MAGMA_D_NEG_ONE;
    double *h_A, *h_R, *tau, *dtau, *h_work, tmp[1];
    double *d_A, *dwork;
    magma_int_t M, N, n2, lda, ldda, lwork, info, min_mn;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    magma_opts opts;
    parse_opts( argc, argv, &opts );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    
    printf("  M     N     CPU GFlop/s (ms)    GPU GFlop/s (ms)    ||I - Q'Q||_F    \n");
    printf("=======================================================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[i];
            N = opts.nsize[i];
            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N;
            ldda   = ((M+31)/32)*32;
            gflops = FLOPS_DGEQRF( M, N ) / 1e9 +  FLOPS_DORGQR( M, N, N ) / 1e9;
            
            lwork = -1;
            lapackf77_dgeqrf(&M, &N, h_A, &M, tau, tmp, &lwork, &info);
            lwork = (magma_int_t)MAGMA_D_REAL( tmp[0] );
            lwork = max(lwork, 3*N*N);
            
            TESTING_HOSTALLOC( tau, double, min_mn );
            TESTING_MALLOC(    h_A, double, n2     );
            TESTING_MALLOC(    h_R, double, n2     );
            TESTING_DEVALLOC(  d_A, double, ldda*N );
            TESTING_DEVALLOC( dtau, double, min_mn );
            TESTING_DEVALLOC(dwork, double,    N*N );
            TESTING_HOSTALLOC( h_work, double,  lwork );
            

            double *ddA, *d_T;
            TESTING_DEVALLOC(  ddA, double,    N*N );
            TESTING_DEVALLOC(  d_T, double,    N*N );
            cudaMemset(ddA, 0, N*N*sizeof(double));
            cudaMemset(d_T, 0, N*N*sizeof(double));

            /* Initialize the matrix */
            lapackf77_dlarnv( &ione, ISEED, &n2, h_A );
            lapackf77_dlacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_R, &lda );
            magma_dsetmatrix( M, N, h_R, lda, d_A, ldda );
            
            // warmup
            magma_dgegqr_gpu( M, N, d_A, ldda, dwork, h_work, &info );
            magma_dsetmatrix( M, N, h_R, lda, d_A, ldda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_sync_wtime( 0 );
            if (opts.version == 2) {
                int min_mn = min(M, N);
                int     nb = N;

                double *dtau = dwork;
                
                magma_dgeqr2x3_gpu(&M, &N, d_A, &ldda, dtau, d_T, ddA, 
                                   (double *)(dwork+min_mn), &info);
                magma_dgetmatrix( min_mn, 1, dtau, min_mn, tau, min_mn);  
                magma_dorgqr_gpu( M, N, N, d_A, ldda, tau, d_T, nb, &info );
            }
            else
               magma_dgegqr_gpu( M, N, d_A, ldda, dwork, h_work, &info );
            gpu_time = magma_sync_wtime( 0 ) - gpu_time;

            gpu_perf = gflops / gpu_time;
            if (info != 0)
                printf("magma_dgegqr returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            if ( opts.lapack ) {
                /* =====================================================================
                   Performs operation using LAPACK
                   =================================================================== */
                cpu_time = magma_wtime();

                /* Orthogonalize on the CPU */
                lapackf77_dgeqrf(&M, &N, h_A, &lda, tau, h_work, &lwork, &info);
                lapackf77_dorgqr(&M, &N, &N, h_A, &lda, tau, h_work, &lwork, &info );

                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0)
                    printf("lapackf77_dorgqr returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                
                /* =====================================================================
                   Check the result compared to LAPACK
                   =================================================================== */
                magma_dgetmatrix( M, N, d_A, ldda, h_R, M );

                double one = MAGMA_D_ONE, zero = MAGMA_D_ZERO;
                blasf77_dgemm("t", "n", &N, &N, &M, &one, h_R, &M, h_R, &M, &zero, h_work, &N);
                for(int ii=0; ii<N*N; ii+=(N+1)) h_work[ii] = MAGMA_D_SUB(h_work[ii], one);

                e1    = lapackf77_dlange("f", &N, &N, h_work, &N, work);

                blasf77_dgemm("t", "n", &N, &N, &M, &one, h_A, &M, h_A, &M, &zero, h_work, &N);
                for(int ii=0; ii<N*N; ii+=(N+1)) h_work[ii] = MAGMA_D_SUB(h_work[ii], one);
                e2    = lapackf77_dlange("f", &N, &N, h_work, &N, work);
                
                printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e  %8.2e\n",
                       (int) M, (int) N, cpu_perf, 1000.*cpu_time, gpu_perf, 1000.*gpu_time, e1, e2 );
            }
            else {
                printf("%5d %5d     ---   (  ---  )   %7.2f (%7.2f)     ---  \n",
                       (int) M, (int) N, gpu_perf, 1000.*gpu_time );
            }
            
            TESTING_HOSTFREE(  tau );
            TESTING_FREE(      h_A );
            TESTING_HOSTFREE(h_work);
            TESTING_FREE(    h_R );
            TESTING_DEVFREE( d_A  );
            TESTING_DEVFREE( dtau );
            TESTING_DEVFREE( dwork );

            TESTING_DEVFREE( ddA );
            TESTING_DEVFREE( d_T );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    
    TESTING_FINALIZE();
    return 0;
}
