function [map, gaps] = greedyClusterQuick(boxes)
% GREEDYCLUSTERQUICK  Greedy clustering of boxes with Quick Shift
%   [MAP, GAPS] = GREEDYCLUSTERQUICK(BOXES) uses Quick Shift to
%   cluster the boxes BOXES.
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

sigma = .7 ;
D = 1 - calcBoxOverlap(boxes,boxes) ;

oN = ones(size(D,1),1) ;

E = sum(exp(- .5 * D / sigma^2), 2) ;

dE = E*oN' - oN*E' ;
D(dE<=0) = +inf ;

[gaps,map] = min(D) ;

% Fix root
r = find(gaps == +inf) ;
map(r) = r ;





