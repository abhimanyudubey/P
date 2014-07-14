function name = calcStageName(stagePrefix, retrainNumber, stageSuffix)
% CALCSTAGENAME  Compute stage name
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

if nargin > 2
  if ~isempty(stageSuffix)
    stageSuffix = ['_' stageSuffix] ;
  end
else
  stageSuffix = '' ;
end
name = sprintf('%s_train%d%s', stagePrefix, retrainNumber, stageSuffix) ;
