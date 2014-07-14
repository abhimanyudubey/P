#include "mex.h"
#include "mexeigen.hpp"
   
#include "pmain.cpp"
#include "Utils.hpp"
#include <math.h>
#include <vector>
#include <ctime>


void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
   
   int npoints = *(double *) mxGetPr(prhs[1]);
   int svd_method = *(double *) mxGetPr(prhs[2]);

  
   MatrixXf  A,C;
   mex2eigen<double>(prhs[0] , A);
   C = pComputeCoresetTree(A, npoints, svd_method);	  
   plhs[0] = mxCreateDoubleMatrix(C.rows(),C.cols(),mxREAL);
   eigen2mex<double>(C,plhs[0]);
    
   
}
