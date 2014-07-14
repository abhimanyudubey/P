function [Y, SEL] = randcol(X,n,varargin)
% RANDCOL Randomly select a number of columns
%  Y = RANDCOL(X, N) returns a random subset (without replacement) Y
%  of N columns of X. The selection is order-preserving.  If N is
%  larger than the number of columns of X, all columns are returned
%  (thus you can specfiy N=+inf to select all the columns).
%
%  If 0 < N < 1, then a fraction N of columns is retained (the actual
%  number of columns is rounded to the closest ingeger). Notice that N
%  = 1 returns one column.
%
%  [Y, SEL] = RANDCOL(...) retusn the indexes of the selected
%  columns SEL as well.
%
%  Author:: Andrea Vealdi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

if nargin < 2, n = 1 ; end

mode = 'random' ;
i = 1 ;
while i <= length(varargin)
  switch lower(varargin{i})
    case 'beginning'
      mode = 'beginning' ; i = i + 1 ;
    case 'ending'
      mode = 'ending' ; i = i + 1 ;
    case 'random' ;
      mode = 'random' ; i = i + 1 ;
    case 'uniform'
      mode = 'uniform' ; i = i +1 ;
    otherwise
      error('Unknown option ''%s''.', varargin{i}) ;
  end
end


m = size(X,2) ;

if n < 0, error('N must not be smaller than 0') ; end
if n ~= round(n)
  if n > 1
    error('N must be either integer, +inf, or a fraction in 0 and 1') ;
  end
  n = round(m * n) ;
end

n = min(m,n) ;

switch mode
  case 'random'
    perm = randperm(m) ;
    sel  = sort(perm(1:n)) ;
  case 'beginning'
    perm = 1:m ;
    sel  = sort(perm(1:n)) ;
  case 'ending'
    perm = m:-1:1 ;
    sel  = sort(perm(1:n)) ;
  case 'uniform'
    if n < 1
      sel = [] ;
    else
      sel = round(linspace(1, m, min(m,n))) ;
    end
end

Y = X(:, sel) ;
