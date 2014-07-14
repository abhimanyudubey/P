function ker = calcKernel(ker, hists, testHists, alphay)
% CALCKERNEL  Compute kernel on histograms
%    KER = CALCKERNEL(KER, HISTS) takes the structure KER describing
%    the kernel to compute and the histograms HISTS on which to
%    compute the kernel. HISTS has an histogram per column and N
%    columns and the compute kernel matrix has dimensions N x N.
%
%    KER = CALCKERNEL(KER, HISTS, TESTHISTS) does the same, except
%    that it computes the kernels among histograms from HISTS (along
%    the rows of the kernel matrix) and TESTHISTS (along the
%    columns). So if HISTS has N columns and TESTHITS has M columns,
%    the kernel matrix has dimension N x M.
%
%    SCORES = CALCKERNEL(KER, HISTS, TESTHISTS, ALPHAY) calculates the
%    score vector ALPAHY' * KER.MATRIX. For some kernels, this
%    operation is highly optimized.
%
%    The function returns and extended structure KER, with the
%    kernel matrix stored in KER.MATRIX.
%
%    For the exponential kernels, the constant MU is read/stored in
%    the field KER.MU. Fist, the function tries to use the stored
%    KER.MU, if the field exists and if its value is not empty and not
%    NaN. Otherwise, it sets MU to the inverse average of the
%    histogram distances.
%
%    HISTS and TESTHISTS can be either DOUBLE or SINGLE precision and
%    can be either FULL or SPARSE. TESTHISTS is internally converted
%    to the same format of HISTS.
%
%    Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

auto = nargin < 3 ;
project = nargin == 4 ;

% Make TRAINHISTS format the same as HISTS  ~~~~~~~~~~~~~~~~~~~~~~~~~~

if ~ auto
  if ~isequal(class(hists), class(testHists))
    switch class(hists)
      case 'double'
        testHists = double(testHists) ;
      case 'single'
        testHists = single(testHists) ;
      otherwise
        assert(false) ;
    end
  end
  
  if ~isequal(issparse(hists), issparse(testHists))
    if issparse(hists)
      testHists = sparse(testHists) ;
    else
      testHists = full(testHists) ;
    end
  end
end

% --------------------------------------------------------------------
%                                                    Compute the kernel
% --------------------------------------------------------------------

switch ker.type
  
  case 'echi2'
    if auto
      matrix = alldist2(hists, 'chi2') ;
    else
      if issparse(hists) | issparse(testHists)
        matrix = vl_alldist2(hists, testHists, 'chi2') ;
      else
        matrix = vl_alldist(hists, testHists, 'chi2') ;
      end
    end
    
    % retrieve or compute mu
    if isfield(ker, 'mu') & ~isnan(ker.mu) & ~isempty(ker.mu)
      mu = ker.mu ;
      %      fprintf('calcKernel: retrieved mu value: %g\n', mu) ;
    else
      if ~ auto
        warning('RBF mu parameter computed when testing') ;
      end
      mu     = 1 ./ mean(matrix(:)) ;
    end
    
    matrix = exp(- mu * matrix) ;

    if project
      ker = alphay(:)' * matrix ;
    else
      ker.matrix = matrix ;
      ker.mu     = mu ;
    end
    clear matrix mu ;
    
  case 'kl1'
    if auto
      if project
        ker = fastKL1(alphay, hists, hists) ;
      else
        ker.matrix = vl_alldist2(hists, 'kl1') ;
      end
    else
      if project        
        %        ker = alphay(:)' * vl_alldist2(hists, testHists, 'kl1') ;
        ker = fastKL1(alphay, hists, testHists) ;
      else
        ker.matrix = vl_alldist2(hists, testHists, 'kl1') ;
      end
    end
    
  case 'kl2'
    if auto
      if project
        ker = (alphay(:)' * hists') * hists ;
      else
        ker.matrix = hists' * hists ;
      end
    else
      if project
        ker = (alphay(:)' * hists') * testHists ;
      else        
        ker.matrix = hists' * testHists ;
      end
    end
    
  case 'kchi2'
    if auto
      if project
        ker = alphay(:)' * vl_alldist2(hists, 'kchi2') ;
      else
        ker.matrix = vl_alldist2(hists, 'kchi2') ;
      end
    else
      if project
        ker = alphay(:)' * vl_alldist2(hists, testHists, 'kchi2') ;
      else        
        ker.matrix = vl_alldist2(hists, testHists, 'kchi2') ;
      end
    end
    
  otherwise
    error('Unsupported kernel ''%s''.', ker.type) ;
end
