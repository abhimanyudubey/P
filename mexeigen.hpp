#include "Eigen/Dense"
using namespace Eigen;

template <class T>
void mex2eigen(const mxArray *m , MatrixXf  &u)
{
if ( !m ) return;

const mwSize *dim = mxGetDimensions(m);
const T *pm = (T *)mxGetData(m);

u.resize(dim[0], dim[1]);

for ( unsigned int i = 0 ; i < dim[0] ; i++ )
for ( unsigned int j = 0 ; j < dim[1] ; j++ ) {
mwSize ind2[2];
ind2[0] = (mwSize)i; ind2[1] = (mwSize)j;
unsigned ij = (unsigned)mxCalcSingleSubscript(m, 2, ind2);

u(i,j) = (T)pm[ij];
}

}


template <class T>
void eigen2mex( MatrixXf  &u ,mxArray *m)
{
T *pm = (T *)mxGetData(m);

for ( unsigned int i = 0 ; i < u.rows() ; i++ )
for ( unsigned int j = 0 ; j < u.cols() ; j++ ) {
mwSize ind2[2];
ind2[0] = (mwSize)i; ind2[1] = (mwSize)j;
unsigned ij = (unsigned)mxCalcSingleSubscript(m, 2, ind2);
pm[ij] = (T)u(i,j);
}

}
