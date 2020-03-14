/*****************************************************************************
 * FILE: omp_bug5.c
 * DESCRIPTION:
 *   Using SECTIONS, two threads initialize their own array and then add
 *   it to the other's array, however a deadlock occurs.
 * AUTHOR: Blaise Barney  01/29/04
 * LAST REVISED: 04/06/05
 *****************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 1000000
#define PI 3.1415926535
#define DELTA .01415926535

/* Note that the whole file is re-formatted */

int main (int argc, char *argv[]) {
  int nthreads, tid, i;
  float a[N], b[N];
  omp_lock_t locka, lockb;

  /* Initialize the locks */
  omp_init_lock(&locka);
  omp_init_lock(&lockb);

  /* Fork a team of threads giving them their own copies of variables */
# pragma omp parallel shared(a, b, nthreads, locka, lockb) private(tid)
  {

    /* Obtain thread number and number of threads */
    tid = omp_get_thread_num();
#   pragma omp master
    {
      nthreads = omp_get_num_threads();
      printf("Number of threads = %d\n", nthreads);
    }
    printf("Thread %d starting...\n", tid);
#   pragma omp barrier

    /* The deadlock occurs because the orders of acquiring locks are different
     * between these two sections. The problem is solved by enforcing a total
     * order that locka must be acquired before lockb when acquiring both locks
     * and lockb must be released before locka when releasing both locks.
     *
     * Additionally, since the two threads will initialize a and b separately,
     * even protected by locks, it is still possible to read uninitialized data
     * from a or b when performing additions. Hence, instead of using locks,
     * it is more reasonable to move the initialization to another sections
     * construct. But I am not doing this because I think this is not what I
     * am supposed to do in this particular problem.
     */
#   pragma omp sections nowait
    {
#     pragma omp section
      {
        printf("Thread %d initializing a[]\n",tid);
        omp_set_lock(&locka);
        for (i=0; i<N; i++)
          a[i] = i * DELTA;
        omp_set_lock(&lockb);
        printf("Thread %d adding a[] to b[]\n",tid);
        for (i=0; i<N; i++)
          b[i] += a[i];
        omp_unset_lock(&lockb);
        omp_unset_lock(&locka);
      }

#     pragma omp section
      {
        /* The above section construct will update b, which is not initialized
         * if the following for loop is uncommented.
         */
        /*for (volatile int i = 0; i < 1000000000; ++i);*/
        printf("Thread %d initializing b[]\n",tid);
        omp_set_lock(&lockb);
        for (i=0; i<N; i++)
          b[i] = i * PI;
        omp_unset_lock(&lockb);
        omp_set_lock(&locka);
        omp_set_lock(&lockb);
        printf("Thread %d adding b[] to a[]\n",tid);
        for (i=0; i<N; i++)
          a[i] += b[i];
        omp_unset_lock(&lockb);
        omp_unset_lock(&locka);
      }
    }  /* end of sections */
  }  /* end of parallel region */

}

