function t = aosToSoa(s)
% AOSTOSOA  Array-of-structs to struct of arrays
%   T = AOSTOSOA(S) converts the array of structures S to a structure
%   of arrays T.
%
%   Author:: Andrea vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

t = struct ;
names = fieldnames(s) ;
for name = names'
  name = char(name) ;
  if numel(s) >= 1 && ischar(s(1).(name))
    t.(name) = {s.(name)} ;
  else
    t.(name) = cat(2, s.(name)) ;
  end
end
