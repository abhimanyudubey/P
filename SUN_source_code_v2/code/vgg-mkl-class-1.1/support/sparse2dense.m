function feat = sparse2dense(sfeat)
% SPARSE2DENSE  Convert sparse feature representation to to dense
%   FEAT = SPARSE2DENSE(SFEAT)
%
%   See also:: DENSE2SPARSE()
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

% Construct a grid xr,yr that can hold the frames.  A complication is
% the fact that features can appear multiple times at the same
% location, for which some coordinates in xr or yr have to appear
% twice or more.

hasWeights = isfield(sfeat, 'weights') ;
hasScales = size(sfeat.frames, 1) > 2 ;

% no features: special case
if size(sfeat.frames, 2) == 0
  feat.wordMap = [0] ;
  feat.xr = [1] ;
  feat.yr = [1] ;
  if hasScales, feat.scaleMap = [1] ; end
  if hasWeights, feat.weightMap = [0] ; end
  return ;
end

% detect unique pairs in P={(x,y)}
P = sfeat.frames(1:2,:) ;
[Q, i, j] = unique(P', 'rows') ;
Q = Q' ;

% sort P' so that duplicate pairs are consecutive
% invariants: isequal(P(:,i), Q), isequal(P, Q(:,j))
[drop, perm] = sort(j) ;
P = P(:, perm) ;
iperm(perm) = 1:numel(perm) ;
i = iperm(i) ;
j = j(perm)' ;

% dis counts duplicates in each consecutive group of duplicate
% it is used to systematically disambiguate them later
dis = (1:numel(j)) - i(j) ;

% disambiguated x coordinate, y coordinate
xd = [P(1,:) ; dis] ;
y  = P(2,:) ;

% grid points
[xdr, drop, xi] = unique(xd', 'rows') ; xr = xdr(:,1)' ; xi = xi' ;
[yr,  drop, yi] = unique(y) ;

% build maps
M = length(yr) ;
N = length(xr) ;
idx = sub2ind([M, N], yi, xi) ;
wordMap  = zeros(M, N, class(sfeat.words)) ;
wordMap(idx) = sfeat.words(perm) ;
feat = struct('xr', xr, 'yr', yr, 'wordMap', wordMap) ;

% additional maps?
if hasScales
  feat.scaleMap = zeros(M, N) ;
  feat.scaleMap(idx) = sfeat.frames(3,perm) ;
end

if hasWeights
  feat.weightMap = zeros(M, N) ;
  feat.weightMap(idx) = sfeat.weights(perm) ;
end
