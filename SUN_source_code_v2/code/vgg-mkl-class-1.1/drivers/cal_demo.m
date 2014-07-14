% CAL_DEMO  Run a Caltech-101 experiment
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

cal_conf ;

cal_preprocDatabases ;
cal_preprocVocabularies ;
cal_preprocKernels ;

roidb = getRoiDb(conf.gtRoiDbPath) ;
allClassNames = fieldnames(roidb.classes) ;

for ci = 1:length(allClassNames)
  magicClassName = allClassNames{ci} ;
  cal_classOneRest ;
end

cal_classMulti ;