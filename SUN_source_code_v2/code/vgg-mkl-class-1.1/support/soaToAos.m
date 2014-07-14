function t = soaToAos(s)
% SOATOAOS  Structure of arrays to array of structures
%  T = SOATOAOS(S) converts the structure of arrays S to an array of
%  structures T.
%
%  Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

names = fieldnames(s)' ;

% normalize s
for name = names
  name = char(name) ;
  if ischar(s.(name)), s.(name) = {s.(name)} ; end
end

pairs = { } ;
for name = names
  name = char(name) ;
  if iscell(s.(name)) && numel(s.(name)) >= 1 && ischar(s.(name){1})
    pairs(end+1:end+2) = {name, s.(name)} ;
  else
    pairs(end+1:end+2) = {name, num2cell(s.(name), 1)} ;
  end
end
t = struct(pairs{:}) ;