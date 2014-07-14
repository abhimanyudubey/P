/* file:        tpsumx.c
** author:      Andrea Vedaldi
** description: vl_tpsumx - MEX definition
**/

/* AUTORIGHTS
 Copyright (C) 2007-09 Andrea Vedaldi and Brian Fulkerson

 This file is part of VLFeat, available in the terms of the GNU
 General Public License version 2.
 */

#include <mexutils.h>

#include <stdlib.h>
#include <math.h>

/** Matlab driver.
 **/
#define getM(arg) mxGetM(in[arg])
#define getN(arg) mxGetN(in[arg])
#define getPr(arg) mxGetPr(in[arg])

void
mexFunction(int nout, mxArray *out[], 
            int nin, const mxArray *in[])
{
  enum { X=0,Y } ;
  enum { U } ;
  
  int NP, NCP ;
  int i,j ;
  double *X_pt, *Y_pt, *U_pt ;  
  const double small = 2.220446049250313e-16 ;
	  
  /* -----------------------------------------------------------------
   *                                               Check the arguments
   * -------------------------------------------------------------- */
  if(nin != 2) {
    mexErrMsgTxt("Two input arguments required") ;
  }
  
  if(!uIsRealMatrix(in[X], 2, -1)) {
    mexErrMsgTxt("X must be a 2xNP real matrix") ;
  }
  
  if(!uIsRealMatrix(in[Y], 2, -1)) {
    mexErrMsgTxt("Y must be a 2xNCP real matrix") ;
  }

  NP  = getN(X) ;
  NCP = getN(Y) ;

  X_pt = getPr(X);
  Y_pt = getPr(Y) ;
  
  /* Allocate the result. */
  out[U] = mxCreateDoubleMatrix(NP, NCP, mxREAL) ;
  U_pt = mxGetPr(out[U]) ;
    
  /* -----------------------------------------------------------------
   *                                               Check the arguments
   * -------------------------------------------------------------- */
  for(j = 0 ; j < NCP ; ++j) {
    double xcp = *Y_pt++ ;
    double ycp = *Y_pt++ ;
    for(i = 0 ; i < NP ; ++i) {
      double dx = *X_pt++ - xcp ;
      double dy = *X_pt++ - ycp ;
      double r2 = dx*dx + dy*dy ;
      *U_pt++ = (r2 <= small) ? 0 : r2 * log (r2) ;			
    }
    X_pt -= 2*NP ;		
  }
}
