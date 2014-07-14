function scores = scoresFromLabels(labels, featNames, pyrLevels, appModel, rois, varargin)
% SCORESFROMLABELS
%   SCORES = SCORESFROMLABELS(LABELS, FEATNAMES, PYRLEVELS, APPMODEL)
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

opts.magnif  = 0 ;
opts = vgg_argparse(opts, varargin) ;

rois = uint32(rois) ;

scores = [] ;
fields = {} ;
for fi = 1:length(featNames)
  f = featNames{fi} ;

  % calculate scores from label image
  for li = 1:length(pyrLevels)
    name = sprintf('%s_L%d', f, li - 1) ;

    tmp = pyramidFromLabels(labels.(f), ...
                            pyrLevels(li), ...
                            appModel.(name), ...
                            rois, ...
                            'magnif', opts.magnif) ;

    if isempty(scores)
      scores = tmp ;
    else
      scores = scores + tmp ;
    end

  end % next level
end % next feature
