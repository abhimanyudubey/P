% CAL_DEMO  Run a Caltech-101 experiment
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

do_preprocDatabases    = 1 ;
do_preprocVocabularies = 1 ;
do_preprocKernels      = 1 ;
do_classOneRest        = 1 ;

expNumRange     = [1 2 3 4 5 6] ;
numTrainRange   = [15 30] ;
randomSeedRange = [1 2 3] ;

magicSmallData = false ;

for magicMasterExpNum = expNumRange
  for magicMasterRandomSeed = randomSeedRange
    for magicMasterNumTrain = numTrainRange
      cal_conf ;

      if do_preprocDatabases
        cal_preprocDatabases ;
      end
      waitJobs = [] ;

      if do_preprocVocabularies
        magicWaitJobs = waitJobs ;
        cal_preprocVocabularies ;
      end

      if do_preprocKernels
        magicWaitJobs = waitJobs ;
        cal_preprocKernels ;
      end

      waitJobs_ker = waitJobs ;

      roidb = getRoiDb(conf.gtRoiDbPath) ;
      allClassNames = fieldnames(roidb.classes) ;

      for ci = 1:length(allClassNames)
        magicClassName = allClassNames{ci} ;
        if do_classOneRest
          magicWaitJobs = waitJobs_ker ;
          cal_classOneRest ;
        end
      end

    end
  end
end
