/*****************************************************************************
 * FILE: omp_bug2.c
 * DESCRIPTION:
 *   Another OpenMP program with a bug.
 * AUTHOR: Blaise Barney
 * LAST REVISED: 04/06/05
 *****************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

/* Note that the whole file is re-formatted */

int main (int argc, char *argv[]) {
  int nthreads, i, tid;
  float total = 0.0; /* Initialization moved here */

  /*** Spawn parallel region ***/
# pragma omp parallel private(i, tid) /* i and tid should be private */
  {
    /* Obtain thread number */
    tid = omp_get_thread_num();
    /* Only master thread does this */
    if (tid == 0) {
      nthreads = omp_get_num_threads();
      printf("Number of threads = %d\n", nthreads);
    }
    printf("Thread %d is starting...\n",tid);

#   pragma omp barrier

    /* do some work */

    /* Initialization of total is moved out */

    /* Consider using reduction
     *  Note that reduction could alternate the floating point path and cause
     *  imprecise result
     */
#   pragma omp for schedule(dynamic,10) reduction(+:total)
    for (i = 0; i < 1000000; i++)
      /* Note that atomic operation is not desired, which implies a sequential
       * execution
       */
      /*#pragma omp atomic update*/
      total = total + i * 1.0;

    printf ("Thread %d is done! Total= %e\n", tid, total);

  } /*** End of parallel region ***/
}
