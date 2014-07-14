function B =partcol(A, N, i)
% PARTCOL
%   B = PARTCOL(A, N, I)
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

if i < 1 | i > N, error('I out of bounds') ; end

M = size(A,2) ;
delta = M / N ;
start = floor((i-1) * delta + .5 +1) ;
stop  = floor((i  ) * delta + .5) ;

B = A(:, start:stop) ;

%  .5   1       2       3
%   |---*---|---*---|---*