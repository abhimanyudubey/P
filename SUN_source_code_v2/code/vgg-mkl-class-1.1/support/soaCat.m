function a = soaCat(varargin)
% SOACAT Concatenation of structure-of-arays
%  X = SOACAT(A, B, C, ...) concatenates the structures of arrays A,
%  B, C, ... into a single structure of arrays X.
%
%  Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

names = fieldnames(varargin{1})' ;

numNames = length(names) ;
numArgs = length(varargin) ;

numRows = NaN + zeros(numNames, numArgs) ;
numCols = NaN + zeros(numNames, numArgs) ;

for ni = 1:numNames
  name = names{ni} ;
  for ai = 1:numArgs
    if isfield(varargin{ai}, name)
      numRows(ni, ai) = size(varargin{ai}.(name), 1) ;
      numCols(ni, ai) = size(varargin{ai}.(name), 2) ;
    end
  end
end

for ai =1:numArgs
  nc = unique(numCols(~isnan(numCols(:, ai)), ai)) ;
  if length(nc) > 1, error('Arguments arnce not consistent') ; end
  numCols(isnan(numCols(:, ai)), ai) = nc ;
end

for ni =1:numNames
  nr = unique(numRows(ni, ~isnan(numRows(ni,:)))) ;
  if length(nr) > 1, error('Arguments are not consistent') ; end
  numRows(ni, isnan(numRows(ni, :))) = nr ;
end

for ni = 1:numNames
  name = names{ni} ;
  tmp = cell(1, numArgs) ;
  tmp{1} = varargin{1}.(name) ;
  for ai = 2:numArgs
    if isfield(varargin{ai}, name)
      tmp{ai} = varargin{ai}.(name) ;
    else
      if iscell(tmp{ai-1})
        tmp{ai} = cell(numRows(ni, ai), numCols(ni, ai)) ;
      elseif isnumeric(tmp{ai-1})
        tmp{ai} = zeros(numRows(ni, ai), numCols(ni, ai), class(tmp{ai-1})) ;
      elseif islogical(tmp{ai-1})
        tmp{ai} = false(numRows(ni, ai), numCols(ni, ai)) ;
      end
    end
  end

  a.(name) = cat(2, tmp{:}) ;
  clear tmp ;
end

end % soaCat
