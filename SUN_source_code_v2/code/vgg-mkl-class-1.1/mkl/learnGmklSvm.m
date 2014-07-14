function svm = learnGmklSvm(kernels, labels)
% LEARNGMKLSVM  Use GMKL and libsvm to learn an SVM model
%   SVM = LEARNGMKLSVM(KERNELS, LABELS)
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

  numKernels = size(kernels, 3) ;

  % Sum of precomputed kernel matrices
  parms.fncKdashint = @(varargin) [] ;
  parms.fncK        = @(TRN, TST, parms, d) lincomb(kernels(TRN,TST,:), d) ;
  parms.fncKdash    = @(TRN, TST, parms, d, k, Kdashint) kernels(TRN,TST,k) ;

  % L1 Regularization
  parms.REGname     = 'l1';
  parms.sigma       = ones(numKernels, 1) ;
  parms.fncR        = @(parms, d) parms.sigma' * d ;
  parms.fncRdash    = @(parms, d) parms.sigma ;

  % Standard SVM parameters
  parms.TYPEprob=0;           % 0 = C-SVC, 3 = EPS-SVR (try others at your own risk)
  parms.C=10;                 % Misclassification penalty for SVC/SVR ('-c' in libSVM)

  % Gradient descent parameters
  parms.initd=rand(numKernels,1); % Starting point for gradient descent
  parms.TYPEsolver='LIB';     % Use libsvm solver
  parms.TYPEstep=1;           % 0 = Armijo, 1 = Variant Armijo, 2 = Hessian (not yet implemented)
  parms.MAXITER=40;           % Maximum number of gradient descent iterations
  parms.MAXEVAL=200;          % Maximum number of SVM evaluations
  parms.MAXSUBEVAL=20;        % Maximum number of SVM evaluations in any line search
  parms.SIGMATOL=0.3;         % Needed by Armijo, variant Armijo for line search
  parms.BETAUP=2.1;           % Needed by Armijo, variant Armijo for line search
  parms.BETADN=0.3;           % Needed by variant Armijo for line search
  parms.BOOLverbose=true;     % Print debug information at each iteration
  parms.SQRBETAUP = parms.BETAUP * parms.BETAUP ;

  % Call MKL code
  svm = COMPGDoptimize([1:size(kernels, 1)], labels, parms) ;
end

% --------------------------------------------------------------------
function K=lincomb(base,d)
% --------------------------------------------------------------------
  nzind = find(d > 1e-4);
  K = zeros(size(base,1), size(base,2));
  if ~isempty(nzind), for k=1:length(nzind), K=K+d(nzind(k))*base(:,:,nzind(k)); end; end;
end
