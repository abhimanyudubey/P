% CAL_CLASSONEREST
%   Classify one class vs the rest.
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

cal_conf ;

if exist('magicWaitJobs', 'var')
  waitJobs = magicWaitJobs ;
  clear magicWaitJobs ;
else
  waitJobs = [] ;
end

if exist('magicClassName', 'var')
  conf.className = magicClassName ;
end

jobDir = fullfile(conf.jobDir, ...
                  ['do' datestr(now,'yymmdd-HHMMSS') ...
                   conf.expPrefix '-' conf.className]) ;

imdb = getImageDb(conf.imageDbPath) ;

% --------------------------------------------------------------------
%                                                             Run jobs
% --------------------------------------------------------------------

kerDir             = fullfile(conf.aggrDir, 'kers') ;
aggrRoiDbDir       = fullfile(conf.aggrDir, 'rois') ;
aggrTestRoiDbDir   = fullfile(conf.aggrDir, 'rois-test') ;
aggrRoiDbPath      = fullfile(aggrRoiDbDir, 'roidb') ;
aggrTestRoiDbPath  = fullfile(aggrTestRoiDbDir, 'roidb') ;

stageDir       = fullfile(conf.trainDir, upper(conf.className), conf.expPrefix) ;
trainRoiDbDir  = fullfile(stageDir, 'rois') ;
trainRoiDbPath = fullfile(trainRoiDbDir, 'roidb') ;
testRoiDbDir   = fullfile(stageDir, 'rois-test') ;
testRoiDbPath  = fullfile(testRoiDbDir, 'roidb') ;
candDir        = fullfile(stageDir, 'cands') ;
modelPath      = fullfile(stageDir, 'model') ;

ensuredir(trainRoiDbDir) ;
ensuredir(testRoiDbDir) ;

conf.do_trainAppModel = 1 ;
conf.do_testAppModel  = 1 ;

% create ROIDBs specific for this class by editing the common ones
trroidb = getRoiDb(aggrRoiDbPath) ;
tsroidb = getRoiDb(aggrTestRoiDbPath) ;

trroidb.labels = 2*(trroidb.rois.class == trroidb.classes.(upper(conf.className))) - 1 ;
tsroidb.labels = 2*(tsroidb.rois.class == tsroidb.classes.(upper(conf.className))) - 1 ;

ssave(trainRoiDbPath, '-STRUCT', 'trroidb') ;
ssave(testRoiDbPath, '-STRUCT', 'tsroidb') ;

kerDb = conf.kerDb ;

% --------------------------------------------------------------------
%                                               Train Appearance Model
% --------------------------------------------------------------------

waitJobs_train = [] ;

if conf.do_trainAppModel
  bkcf                 = bk_trainAppModel ;
  bkcf.trainRoiDbPath  = trainRoiDbPath ;
  bkcf.trainHistDir    = trainRoiDbDir ;
  bkcf.kerDir          = kerDir ;
  bkcf.kerDb           = kerDb ;
  bkcf.learnWeights    = conf.learnWeightMethod ;
  bkcf.modelPath       = modelPath ;
  bkcf.posWeight       = conf.posWeight ;
  bkcf.noClobber       = conf.noClobber ;
  [bkcf, waitJobs_train] = ...
      bk_trainAppModel(bkcf, ...
                       'parallelize', conf.parallelize, ...
                       'jobDir', jobDir, ...
                       'waitJobs', waitJobs, ...
                       'freeMemMb', 1024 * 4) ;
end

% --------------------------------------------------------------------
%                                                                 Test
% --------------------------------------------------------------------

waitJobs_test = [] ;

if conf.do_testAppModel
  bkcf                 = bk_testAppModel ;
  bkcf.trainHistDir    = trainRoiDbDir ;
  bkcf.testRoiDbPath   = testRoiDbPath ;
  bkcf.testHistDir     = testRoiDbDir ;
  bkcf.kerDir          = kerDir ;
  bkcf.kerDb           = kerDb ;
  bkcf.modelPath       = modelPath ;
  bkcf.noClobber       = conf.noClobber ;

  [bkcf, waitJobs_test] = ...
      bk_testAppModel(bkcf, ...
                      'parallelize', conf.parallelize, ...
                      'jobDir', jobDir, ...
                      'waitJobs', [waitJobs_train waitJobs]) ;
end

if ~isempty([waitJobs_train waitJobs_test])
  waitJobs = [waitJobs_train waitJobs_test] ;
end
