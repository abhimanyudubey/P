#include "mex.h"
#include <stdlib.h>
#include <math.h>


double const tol = 1e-3 ;

double calcErr(double const* X, int N, double const* Y,
               int i, int j)
{
  double e = 0;
  int k ;
  if (i < 0 || j > N - 1) {
    return mxGetInf() ;
  }
  for (k = i ; k <= j ; ++k) {
    double lam = (X[k] - X[i]) / (X[j] - X[i] + 1e-10) ;
    double z = (Y[j] - Y[i]) * lam + Y[i] ;
    double ep = fabs (z - Y[k]) ;
    if (ep > e) e = ep ;
  }
  return e ;
}

typedef struct _HeapEntry {
  double error ;
  int index ;
} HeapEntry ;

typedef struct _Heap {
  HeapEntry* tree ;
  int* reverse_index  ;
  int len ;
} Heap ;


__inline int heap_parent(int node) {
  if (node == 0) return 0 ;
  return (node - 1) / 2 ;
}

__inline int heap_left_child (int node) {
  return 2*node + 1 ;
}

__inline void heap_switch (Heap* heap, int nodea, int nodeb)
{
  HeapEntry tmp = heap->tree [nodea] ;
  heap->tree [nodea] = heap->tree [nodeb] ;
  heap->tree [nodeb] = tmp ;
  heap->reverse_index [heap->tree [nodea].index] = nodea ;
  heap->reverse_index [heap->tree [nodeb].index] = nodeb ;
}

void heap_down (Heap* heap, int node)
{
  int parent ;
  if (node == 0) return ;
  parent = heap_parent(node) ;

  if (heap->tree[node].error < heap->tree[parent].error) {
    heap_switch (heap, node, parent) ;
    heap_down (heap, parent) ;
  }
}

void heap_up (Heap* heap, int node)
{
  int left_child = heap_left_child (node) ;
  int right_child = left_child + 1 ;

  /* no childer: stop */
  if (left_child >= heap->len) return ;

  /* only left childer: easy */
  if (right_child >= heap->len) {
    if (heap->tree[node].error > heap->tree[left_child].error) {
      heap_switch (heap, node, left_child) ;
    }
    return ;
  }

  /* both childern */
  if (heap->tree[left_child].error < heap->tree[right_child].error) {
    /* swap with left */
    if (heap->tree[node].error > heap->tree[left_child].error) {
      heap_switch (heap, node, left_child) ;
      heap_up (heap, left_child) ;
    }
  } else {
    /* swap with right */
    if (heap->tree[node].error > heap->tree[right_child].error) {
      heap_switch (heap, node, right_child) ;
      heap_up (heap, right_child) ;
    }
  }
}

void heap_push (Heap* heap, HeapEntry newValue)
{
  heap->tree [heap->len] = newValue ;
  heap->reverse_index [newValue.index] = heap->len ;
  ++ heap->len ;
  heap_down (heap, heap->len - 1) ;
}

HeapEntry heap_pop (Heap* heap)
{
  HeapEntry e ;
  -- heap->len ;
  e = heap->tree [0] ;
  heap->reverse_index [e.index] = -1 ;
  heap->tree [0] = heap->tree [heap->len] ;
  heap_up (heap, 0) ;
  return e ;
}

void heap_change_pri (Heap* heap, int node, double error)
{
  heap->tree [node].error = error ;
  heap_up (heap, node) ;
  heap_down (heap, node) ;
}


void
mexFunction(int nout, mxArray *out[],
            int nin, const mxArray *in[])
{
  enum {IN_X=0,IN_Y,IN_THR} ;
  enum {OUT_ERR} ;

  int N, i;
  double *X, *Y, *err ;
  double const thr = *mxGetPr(in[IN_THR]) ;
  double const inf = mxGetInf() ;
  int  *prev ;
  int  *next ;
  Heap heap ;

  N = mxGetNumberOfElements(in[IN_X]) ;
  X = mxGetPr(in[IN_X]) ;
  Y = mxGetPr(in[IN_Y]) ;

  out[OUT_ERR] = mxCreateDoubleMatrix(1,N,mxREAL) ;
  err  = mxGetPr(out[OUT_ERR]) ;
  prev = mxMalloc(sizeof(int) * N) ;
  next = mxMalloc(sizeof(int) * N) ;

  for (i = 0 ; i < N ; ++i) {
    prev [i] = i - 1 ;
    next [i] = i + 1 ;
    err  [i] = calcErr(X,N,Y,i-1,i+1) ;
  }

  heap.tree = mxMalloc(sizeof(HeapEntry) * N) ;
  heap.reverse_index = mxMalloc(sizeof(int) * N) ;
  heap.len = 0 ;

  for (i = 0 ; i < N ; ++i) {
    HeapEntry e ;
    e.index = i ;
    e.error = err[i] ;
    heap_push (&heap, e) ;
  }

  while (heap.len > 0) {
    double bestErr = inf ;
    int best = 0, a, b ;
    HeapEntry e = heap_pop(&heap) ;

    /* mexPrintf("error: %g\n", e.error) ; */

    bestErr = e.error ;
    best    = e.index ;

    if (bestErr > thr) break ;

    err[best] = inf ;
    a = prev[best] ;
    b = next[best] ;
    prev[b] = a ;
    next[a] = b ;

    err[a] = calcErr(X,N,Y,prev[a],next[a]) ;
    err[b] = calcErr(X,N,Y,prev[b],next[b]) ;

    heap_change_pri (&heap, heap.reverse_index[a], err[a]) ;
    heap_change_pri (&heap, heap.reverse_index[b], err[b]) ;
  }

  mxFree(prev) ;
  mxFree(next) ;
}
