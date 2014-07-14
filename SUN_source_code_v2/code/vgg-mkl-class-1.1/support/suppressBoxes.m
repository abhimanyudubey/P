function keep = suppressBoxes(boxes, thresh, maxNum)
% SUPPRESSBOXES  Non-maxima suppression of overlapping boxes
%   KEEP = SUPPRESSBOXES(BOXES, THRESH, MAXNUM) supporesses the
%   boxes BOXES sorted by decreasing score. The function keeps the
%   highly scored box, then eliminates all the ones which overlap
%   by more than THRESH, and keep interating until at most MAXNUM
%   boxes are obtained.
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

if nargin < 3
  maxNum = +inf ;
end

if maxNum == +inf & thresh == 1
  keep = true(1,size(boxes,2)) ;
  return ;
end

suppr  = false(1,size(boxes,2));
keep   = false(1,size(boxes,2));
next   = 1 ;

for i=1:maxNum
  sel = find(~suppr) ;
  if isempty(sel), break ; end
  
  next = min(find(~suppr)) ;
  keep(next) = true ;
  box = boxes(:,next) ;
  
  overl = calcBoxOverlap(box, boxes) ;
  suppr = suppr | (overl >= thresh) ;
end

