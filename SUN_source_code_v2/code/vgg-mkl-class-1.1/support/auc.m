function [z,zpa08]=auc(x,y)
% AUC Area under curve
%
%  Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

if length(y) < 2
  z=0 ;
  zpa08=0 ;
  return ;
end

z = .5 * sum(abs(diff(x)) .* (y(2:end) + y(1:end-1))) ;

% compute auc according to PA08 challenge
ap=0;
for t=0:0.1:1
  p=max(y(x>=t));
  if isempty(p)
    p=0;
  end
  ap=ap+p/11;
end
zpa08=ap ;
