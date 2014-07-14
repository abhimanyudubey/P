% CAL_PREPROCKERNELS
%   Preprocess kernel matrices
%
%   Autorighs:: Andrea Vedaldi

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

jobDir = fullfile(conf.jobDir, ['do' datestr(now,'yymmdd-HHMMSS') ...
                    conf.expPrefix '-COMMON']) ;

kerDb = conf.kerDb ;

% --------------------------------------------------------------------
%                                                 Setup Train and Test
% --------------------------------------------------------------------

cal_setupTrainTest ;

% --------------------------------------------------------------------
%                                             Compute training kernels
% --------------------------------------------------------------------

waitJobs_ker = [] ;

for fi=1:length(conf.featNames)
  featName = conf.featNames{fi} ;
  featOpts = conf.featOpts.(featName) ;
  fprintf('******************** %s\n', featName) ;

  % processing differs depending on whether the features is visual word
  % based or not

  waitJobs_metaDist  = [] ;
  waitJobs_hist      = [] ;
  waitJobs_kerTrain  = [] ;
  waitJobs_kerTest   = [] ;

  if featOpts.vocabSize == 0

    bkcf = meta_calcDistMatrix ;
    bkcf.imageDbPath = conf.imageDbPath ;
    bkcf.imageSelTrain = selImageTrain ;
    bkcf.imageSelTest  = selImageTest ;
    bkcf.featDir       = fullfile(conf.ppDir, featName, 'features') ;
    bkcf.distFn        = conf.featOpts.(featName).distFn ;
    bkcf.distDir       = kerDir ;
    bkcf.featPrefix    = featName ;
    bkcf.kerDir        = kerDir ;
    bkcf.noClobber     = conf.noClobber ;
    [bkcf, waitJobs_metaDist] = meta_calcDistMatrix(...
      bkcf, ...
      'parallelize', conf.parallelize, ...
      'jobDir', jobDir, ...
      'waitJobs', waitJobs) ;

  else

    % find kernels that use this feature
    selk = findKernels(kerDb, 'feat', featName) ;

    waitJobs_hist = [] ;

    featHistDir = fullfile(histDir, featName) ;

    opts  = bk_calcRoiHistograms ;
    opts.histDir     = featHistDir ;
    opts.ppDir       = conf.ppDir ;
    opts.imageDbPath = conf.imageDbPath ;
    opts.roiDbPath   = conf.gtRoiDbPath ;
    opts.featOpts    = conf.featOpts ;
    opts.featNames   = {featName} ;
    opts.noClobber   = conf.noClobber ;

    [opts, waitJobs_hist] = ...
        bk_calcRoiHistograms(opts, ...
                             'parallelize', conf.parallelize, ...
                             'jobDir', conf.jobDir, ...
                             'numNodes', 5, ...
                             'waitJobs', waitJobs) ;

    bkcf             = bk_packRoiHistograms ;
    bkcf.roiDbPath   = trainRoiDbPath ;
    bkcf.histDir     = featHistDir ;
    bkcf.histPackDir = trainRoiDbDir ;
    bkcf.featNames   = { featName } ;
    bkcf.featOpts    = conf.featOpts ;
    bkcf.noClobber   = conf.noClobber ;
    [bkcf, waitJobs_kerTrain] = ...
        bk_packRoiHistograms(bkcf, ...
                             'parallelize', conf.parallelize, ...
                             'jobDir', jobDir, ...
                             'waitJobs', [waitJobs waitJobs_hist]) ;

    bkcf             = bk_calcKernels ;
    bkcf.histDir     = trainRoiDbDir ;
    bkcf.kerDir      = kerDir ;
    bkcf.kerDescrDir = kerDir ;
    bkcf.kerDb       = kerDb(selk) ;
    bkcf.noClobber   = conf.noClobber ;
    [bkcf, waitJobs_kerTrain] = ...
        bk_calcKernels(bkcf, ...
                       'parallelize', conf.parallelize, ...
                       'jobDir', jobDir, ...
                       'waitJobs', [waitJobs waitJobs_kerTrain]) ;

    bkcf             = bk_packRoiHistograms ;
    bkcf.roiDbPath   = testRoiDbPath ;
    bkcf.histDir     = featHistDir ;
    bkcf.histPackDir = testRoiDbDir ;
    bkcf.featNames   = { featName } ;
    bkcf.featOpts    = conf.featOpts ;
    bkcf.noClobber   = conf.noClobber ;
    [bkcf, waitJobs_kerTest] = ...
        bk_packRoiHistograms(bkcf, ...
                           'parallelize', conf.parallelize, ...
                           'jobDir', jobDir, ...
                           'waitJobs', [waitJobs waitJobs_hist]) ;

    % need to wait for train kernel to be done to know the gamma
    % scaling factor of the RBF kernel
    bkcf             = bk_calcKernels ;
    bkcf.histDir     = trainRoiDbDir ;
    bkcf.testHistDir = testRoiDbDir ;
    bkcf.kerDir      = kerDir ;
    bkcf.kerDescrDir = kerDir ;
    bkcf.kerDb       = kerDb(selk) ;
    bkcf.noClobber   = conf.noClobber ;
    [bkcf, waitJobs_kerTest] = ...
        bk_calcKernels(bkcf, ...
                       'parallelize', conf.parallelize, ...
                       'jobDir', jobDir, ...
                       'waitJobs', [waitJobs waitJobs_kerTrain waitJobs_kerTest]) ;
  end

  waitJobs_ker = [waitJobs_ker ...
                  waitJobs_hist ...
                  waitJobs_metaDist ...
                  waitJobs_kerTrain ...
                  waitJobs_kerTest] ;
end

if ~isempty(waitJobs_ker), waitJobs = waitJobs_ker ; end
