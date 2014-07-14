function k = pwlKerEval(A,B,th,testHists)
% PWLKEREVAL

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

if issparse(testHists), testHists = full(testHists) ; end

[M,N] = size(testHists) ;
k = zeros(1,N) ;

for m=1:M
  t = testHists(m,:) ;
  i = binsearch(th{m}, t) ;
  k = k + A{m}(i) .* t + B{m}(i) ;
end
