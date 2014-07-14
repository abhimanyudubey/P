function [map, C] = flatmap(map)
% FLATMAP  Flatten Quick Shift map
%
%   Autorights:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

while 1
  map_ = map(map) ;
  if isequal(map_,map) ; break ; end
  map = map_ ;
end

[drop,drop,C] = unique(map)  ;



