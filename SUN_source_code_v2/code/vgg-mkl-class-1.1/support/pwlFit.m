function [z,A,B,tx] = pwlFit(x,y,tol)
% PWFIT  Greedy piecewise linear fit

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

if nargin < 3, tol = 1e-3 ; end
thr  = (max(y) - min(y)) * tol ;

err = pwlFitMex(full(x),full(y),thr) ;
[z,A,B,tx] = calcApprox(x,y,err) ;

return 

% pure MATLAB version

z_=z;
A_=A;
B_=B;
tx_=tx;


n    = length(x) ;
err  = +inf * ones(1,n) ;
prev = 0:n-1 ;
next = 2:n+1 ;

for i=1:n
  err(i) = calcErr(x,y,i-1,i+1) ;
end

while true
  [e,best] = min(err) ;
  if e > thr, break ; end

  % remove this poin  
  err(best) = +inf ;
  a = prev(best) ;
  b = next(best) ;
  prev(b) = a ;
  next(a) = b ;
  
  % recalculate errors
  err(a) = calcErr(x,y,prev(a), next(a)) ;
  err(b) = calcErr(x,y,prev(b), next(b)) ;
end

err_ = pwlFitMex(x,y,thr) ;
[z,A,B,tx] = calcApprox(x,y,err) ;


% --------------------------------------------------------------------
function e = calcErr(x,y,i,j)
% --------------------------------------------------------------------
n = length(x) ;
if i < 1 | j > n, e=inf; return ;end
sel = i:j ;
lam = (x(sel) - x(i)) / max(x(j) - x(i), eps) ;
z = (y(j) - y(i)) * lam + y(i) ;
e = max(abs(z - y(sel))) ;

% --------------------------------------------------------------------
function [z,A,B,tx] = calcApprox(x,y,err)
% --------------------------------------------------------------------
n = size(x,2) ;
sel = [1 find(~isinf(err)) n] ;
tx  = x(sel) ;

selp = [sel(2:end) sel(end)] ;
A   = (y(selp) - y(sel)) ./ max(x(selp)-x(sel),eps) ;
B   = y(sel) - A .* tx ;
z   = evalApprox(A,B,tx,x) ;

% --------------------------------------------------------------------
function z = evalApprox(A,B,tx,x)
% --------------------------------------------------------------------
i = binsearch(tx,x) ;
z = A(i) .* x + B(i) ;

