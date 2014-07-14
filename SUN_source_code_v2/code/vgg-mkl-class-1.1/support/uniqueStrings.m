function list = uniqueStrings(list)
% UNIQUESTRINGS Remove duplicates from list of strings
%   LIST = UNIQUESTRINGS(LIST) removes from the cell array of strings
%   LIST all the duplicates.
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

list = sort(list) ;
i = 1 ;
while i < length(list)
  if isequal(list{i}, list{i+1})
    list(i+1) = [] ;
    continue ;
  end
  i = i + 1 ;  
end
