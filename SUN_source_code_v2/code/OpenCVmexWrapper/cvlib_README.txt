cvlib_mex is a collection (over two dozens) of matlab callable routines from the
the OpenCV library (www.intel.com/research/mrl/research/opencv/). OpenCV is a
real time computer vision library with many image processing capabilities and
it is amazingly fast as well as economic.

To get a list of all available functions run cvlib_mex without any arguments, e.g:
cvlib_mex
Dedicated (short) help on each function is obtained by runing
cvlib_mex('funname'), eg:

cvlib_mex('resize')
 
A longer help (but often still too short) may be obtained by consulting the
OpenCV manual pages.
To run this mex you probably need the to have all the dlls in the same directory.

Example 1: let IMG be a MxNx3 uint8 image.
imr = cvlib_mex('resize',IMG,3.4);
will resize the image, using a bilinear interpolation, to 3.4 times the original size.


Example 2: let A and B be MxN single arrays.
C = cvlib_mex('mul',A,B);
will do a per-element multiplication of A and B, or a  C = A .* B
and
cvlib_mex('mul',A,B);
will do the same but in-place, storing the result in A.
