function qualName = calcQualClassName(className, aspectName)
% CALCQUALCLASSNAME  Compute qualified class name
%   QUALNAME = CALCQUALCLASSNAME(CLASSNAME, ASPECTNAME)
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

if nargin < 2
  aspectName = '.*' ;
end

qualName = className ;
if ~strcmp(aspectName, '.*')
  qualName = [qualName '-' aspectName] ;
  qualName(qualName == '.') = [] ;
  qualName(qualName == '*') = [] ;
  qualName(qualName == '|') = [] ;
end
