% COMPILE  Compile MATLAB mex files
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

mexOpts = {'-O', '-v', '-g'} ;

switch computer
  case {'PCWIN64', 'GLNXA64'}
    mexOpts64 = {mexOpts{:}, '-largeArrayDims'} ;
  otherwise
    mexOpts64 = mexOpts ;
end

switch computer
  case {'PCWIN', 'PCWIN64'}
    mex('tinyLockClient.c', 'ws2_32.lib',  mexOpts{:}) ;
    mex('features\vggSupport\vgg_kmiter.cxx',    '-outdir', 'features\vggSupport', mexOpts{:}) ;
    mex('features\vggSsim\MEXfindnearestl2.cpp', '-outdir', 'features\vggSsim', mexOpts{:}) ;
    mex('features\vggSsim\mexFindSimMaps.cpp',   '-outdir', 'features\vggSsim', mexOpts{:}) ;
    mex('support\pwlFitMex.c',                   '-outdir', 'support', mexOpts{:}) ;
    mex('mkl\libsvm\svm.cpp',                     '-outdir', 'mkl\libsvm', '-c', mexOpts64{:}) ;
    mex('mkl\libsvm\svm_model_matlab.c',          '-outdir', 'mkl\libsvm', '-c', mexOpts64{:}) ;
    mex('mkl\libsvm\svmtrain.c',                  '-outdir', 'mkl\libsvm', 'mkl\libsvm\svm.obj', 'mkl\libsvm\svm_model_matlab.obj', mexOpts64{:}) ;
    mex('mkl\libsvm\svmpredict.c',                '-outdir', 'mkl\libsvm', 'mkl\libsvm\svm.obj', 'mkl\libsvm\svm_model_matlab.obj', mexOpts64{:}) ;
    mex('mkl\libsvm\read_sparse.c',               '-outdir', 'mkl\libsvm', mexOpts64{:}) ;

  otherwise
    mexOpts = {mexOpts{:}, 'CFLAGS=\$CFLAGS -std=c99'} ;
    mexOpts64 = {mexOpts64{:}, 'CFLAGS=\$CFLAGS -std=c99'} ;
    mex('tinyLockClient.c', mexOpts{:}) ;
    mex('features/vggSupport/vgg_kmiter.cxx',    '-outdir', 'features/vggSupport', mexOpts{:}) ;
    mex('features/vggSsim/MEXfindnearestl2.cpp', '-outdir', 'features/vggSsim', mexOpts{:}) ;
    mex('features/vggSsim/mexFindSimMaps.cpp',   '-outdir', 'features/vggSsim', mexOpts{:}) ;
    mex('support/pwlFitMex.c',                   '-outdir', 'support', mexOpts{:}) ;
    mex('mkl/libsvm/svm.cpp',                    '-outdir', 'mkl/libsvm', '-c', mexOpts64{:}) ;
    mex('mkl/libsvm/svm_model_matlab.c',         '-outdir', 'mkl/libsvm', '-c', mexOpts64{:}) ;
    mex('mkl/libsvm/svmtrain.c',                 '-outdir', 'mkl/libsvm', 'mkl/libsvm/svm.o', 'mkl/libsvm/svm_model_matlab.o', mexOpts64{:}) ;
    mex('mkl/libsvm/svmpredict.c',               '-outdir', 'mkl/libsvm', 'mkl/libsvm/svm.o', 'mkl/libsvm/svm_model_matlab.o', mexOpts64{:}) ;
    mex('mkl/libsvm/read_sparse.c',              '-outdir', 'mkl/libsvm', mexOpts64{:}) ;
end
