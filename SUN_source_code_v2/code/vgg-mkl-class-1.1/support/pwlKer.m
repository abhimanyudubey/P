function [A,B,th] = pwlKer(type, trainHists, alphay)
% PWLKER  Piecewise-linear compression of quasi-linear kernels
%
%   [A,B,th] = PWLKER(TYPE, TRAINHIST, ALPHAY)
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

M = size(trainHists, 1) ;
A  = cell(1,M) ;
B  = cell(1,M) ;
th = cell(1,M) ;

subs = round(linspace(1, size(trainHists,2)+50, ...
                min(800, size(trainHists,2)+50))) ;
tic
for m=1:M
  h  = full(trainHists(m,:)) ;
  h_ = sort(unique([h, linspace(0,1,50)])) ;
  subs = round(linspace(1, length(h_), min(800, length(h_)))) ;
  h_ = h_(subs) ;
  
  k = alphay' * alldist2(h, h_, type) ;
  
  [drop, A{m}, B{m}, th{m}] = pwlFit(h_, k) ;
  if mod(m-1,30) == 0 
    fprintf('pwkKer: %s: size: %d %.2f %% done %.2f m rem\r', type, M, m/M*100, ...
            toc / m * (M - m) / 60) ;
  end
end
fprintf('\n') ;
