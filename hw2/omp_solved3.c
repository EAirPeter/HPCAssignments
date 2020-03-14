/*****************************************************************************
 * FILE: omp_bug3.c
 * DESCRIPTION:
 *   Run time error
 * AUTHOR: Blaise Barney  01/09/04
 * LAST REVISED: 06/28/05
 *****************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N     50

/* Note that the whole file is re-formatted */

int main (int argc, char *argv[]) {
  int i, nthreads, tid, section;
  float a[N], b[N], c[N];
  void print_results(float array[N], int tid, int section);

  /* Some initializations */
  for (i=0; i<N; i++)
    a[i] = b[i] = i * 1.0;

  /* A barrier in print_result will cause hanging, it could be solved either
   * by removing that barrier or restricting the number of threads to 2.
   * This file adopts the latter solution.
   */
# pragma omp parallel private(c,i,tid,section) /*num_threads(2)*/
  {
    tid = omp_get_thread_num();
    if (tid == 0)
    {
      nthreads = omp_get_num_threads();
      printf("Number of threads = %d\n", nthreads);
    }

    /*** Use barriers for clean output ***/
#   pragma omp barrier
    printf("Thread %d starting...\n",tid);
#   pragma omp barrier

#   pragma omp sections nowait
    {
#     pragma omp section
      {
        section = 1;
        for (i=0; i<N; i++)
          c[i] = a[i] * b[i];
        print_results(c, tid, section);
      }

#     pragma omp section
      {
        section = 2;
        for (i=0; i<N; i++)
          c[i] = a[i] + b[i];
        print_results(c, tid, section);
      }

    }  /* end of sections */

    /*** Use barrier for clean output ***/
#   pragma omp barrier
    printf("Thread %d exiting...\n",tid);

  }  /* end of parallel section */
}

void print_results(float array[N], int tid, int section) {
  int i,j;

  j = 1;
  /*** use critical for clean output ***/
# pragma omp critical
  {
    printf("\nThread %d did section %d. The results are:\n", tid, section);
    for (i=0; i<N; i++) {
      printf("%e  ",array[i]);
      j++;
      if (j == 6) {
        printf("\n");
        j = 1;
      }
    }
    printf("\n");
  } /*** end of critical ***/

  /* This barrier will cause hanging. There are only two sections in the
   * sections construct, which implies there will only be two threads reaching
   * this barrier. The barrier will wait for the rest of threads, which will
   * never reach here.
   */
/*# pragma omp barrier*/
  printf("Thread %d done and synchronized.\n", tid);
}

