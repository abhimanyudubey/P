function [map, jump] = greedyCluster(boxes)
% GREEDYCLUSTER  Greedy clustering of boxes
%   [MAP, JUMP] = GREEDYCLUSTER(BOXES) computes an agglomerative
%   clustering of the boxes BOXES based on overlap.
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.


n = size(boxes,2) ;

clus = num2cell(1:n) ;
for i=1:n
  for j=1:n
    D(i,j) = calcMergeCost(boxes,[clus{i} clus{j}]) ;
  end
end

D        = D + diag(inf*ones(1,n)) ;
map      = zeros(1,2*n-1) ;
jump     = map ;
ids      = 1:n ;

for t=1:n-1
  nLeft = n-t+1;
  [minD,idx] = min(D(:)) ;
  [i,j] = ind2sub([nLeft,nLeft],idx) ;
  
  % make sure i <= j
  if i > j
    tmp = j ; j = i ; i = tmp ;
  end

  % create novel node t + n
  map(ids(i)) = t + n ;
  map(ids(j)) = t + n ;
  jump(t+n)   = minD ;
  
  % assign novel node to i-th position
  clus{i} = [clus{i}, clus{j}] ;
  for k=1:nLeft
    D(i,k) = calcMergeCost(boxes, [clus{k}, clus{i}]) ;
    D(k,i) = D(i,k) ;
  end
  D(i,i) = inf ;

  % move last element to the j-th position
  col = D(:,end) ;
  D(:,j) = col ;
  D(j,:) = col' ;  
  D(j,j) = inf ;
  ids(i) = t + n ;
  ids(j) = ids(nLeft) ; 
  clus{j} = clus{nLeft} ;
    
  % remove last element
  D = D(1:end-1, 1:end-1) ;
  %clus(nLeft) = [] ;
  
  if 0
    figure(1) ; clf ;
    subplot(1,3,1) ; imagesc(D) ;
    subplot(1,3,2) ; plot(map) ;
    subplot(1,3,3) ; plot(jump) ;
    drawnow ;
  end
end

% --------------------------------------------------------------------
function d = calcMergeCost(boxes,sel)
% --------------------------------------------------------------------
x1 = min(boxes(3,sel)) ;
x2 = max(boxes(3,sel)) ;
y1 = min(boxes(4,sel)) ;
y2 = max(boxes(4,sel)) ;
b0 = [1;1;sqrt(x1.*x2);sqrt(y1.*y2)] ;
d = max(1 - calcBoxOverlap(b0, boxes(:,sel))) ;
