function histNames = calcHistNames(featNames, featOpts)
% CALCHISTNAMES
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

histNames = {} ;
for fi=1:length(featNames)
  featName = featNames{fi} ;
  pyrLevels = featOpts.(featName).pyrLevels ;

  if isempty(pyrLevels)
    histNames{end+1} = featName ;
  else
    for pi=1:length(pyrLevels)
      histNames{end+1} = sprintf('%s_L%d', featName, pyrLevels(pi)) ;
    end
  end
end
