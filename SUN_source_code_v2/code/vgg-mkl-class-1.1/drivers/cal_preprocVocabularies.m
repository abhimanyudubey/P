% CAL_PREPROCVOCABULARIES  Prepare the visual word vocabularies
%   CAL_PRERPOCVOCABULARIES precompute the features for all images in
%   the database.
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

cal_conf ;

do_extractFeatures                 = 1 ;
do_calcVocabulary                  = 1 ;
do_calcQuantizedFeatures           = 1 ;
do_calcQuantizedFeaturesWithJitter = 1 ;

if exist('magicWaitJobs', 'var')
  waitJobs = magicWaitJobs ;
  clear magicWaitJobs ;
else
  waitJobs = [] ;
end

% --------------------------------------------------------------------
%                                               Find images to process
% --------------------------------------------------------------------

imdb       = getImageDb(conf.imageDbPath) ;
sel        = findImages(imdb, 'set', conf.trainSetName) ;
imageNames = imdb.images.name(sel) ;

% --------------------------------------------------------------------
%                                                 Process all features
% --------------------------------------------------------------------

waitJobs_ = [] ;

for featName = conf.featNames
  featName = char(featName) ;

  waitJobs_extract = [] ;
  waitJobs_vocab   = [] ;
  waitJobs_quant   = [] ;

  if do_extractFeatures && conf.featOpts.(featName).vocabSize > 0
    opts            = bk_calcFeatures ;
    opts.imageDir   = imdb.dir ;
    opts.imageNames = imageNames ;
    opts.jitterName = '' ;
    opts.featOpts   = conf.featOpts.(featName) ;
    opts.featDir    = fullfile(conf.ppDir, featName, 'features') ;
    opts.noClobber  = conf.noClobber ;

    [opts, waitJobs_extract] = ...
        bk_calcFeatures(opts, ...
                        'parallelize', conf.parallelize, ...
                        'jobDir', conf.jobDir, ...
                        'waitJobs', waitJobs) ;
  end

  if do_calcVocabulary && conf.featOpts.(featName).vocabSize > 0
    opts = bk_calcVocabulary ;
    opts.gtRoiDbPath  = conf.gtRoiDbPath ;
    opts.imageDbPath  = conf.imageDbPath ;
    opts.trainSetName = conf.trainSetName ;
    opts.featOpts     = conf.featOpts.(featName) ;
    opts.featDir      = fullfile(conf.ppDir, featName, 'features') ;
    opts.vocabPath    = fullfile(conf.ppDir, featName, 'vocab') ;
    opts.noClobber    = conf.noClobber ;

    [opts, waitJobs_vocab] = ...
        bk_calcVocabulary(opts, ...
                          'parallelize', conf.parallelize, ...
                          'jobDir', conf.jobDir, ...
                          'waitJobs', [waitJobs waitJobs_extract]) ;
  end

  if do_calcQuantizedFeatures
    % process all images without jitter
    sel = findImages(imdb) ;
    imageNames = imdb.images.name(sel) ;

    opts            = bk_calcFeatures ;
    opts.imageDir   = imdb.dir ;
    opts.imageNames = imageNames ;
    opts.featOpts   = conf.featOpts.(featName) ;
    opts.noClobber  = conf.noClobber ;

    if conf.featOpts.(featName).vocabSize > 0
      opts.wordDir    = fullfile(conf.ppDir, featName, 'words') ;
      opts.vocabPath  = fullfile(conf.ppDir, featName, 'vocab') ;
    else
      opts.featDir    = fullfile(conf.ppDir, featName, 'features') ;
    end

    [opts, waitJobs_quant(end+1)] = ...
        bk_calcFeatures(opts, ...
                        'parallelize', conf.parallelize, ...
                        'jobDir', conf.jobDir, ...
                        'waitJobs', [waitJobs waitJobs_extract waitJobs_vocab]) ;
  end

  if do_calcQuantizedFeaturesWithJitter
    % procces jittered training images
    sel = findImages(imdb, 'set', conf.trainSetName) ;
    imageNames = imdb.images.name(sel) ;

    for jitterName = {conf.jitterNames{:}}
      jitterName = char(jitterName) ;
      opts            = bk_calcFeatures ;
      opts.imageDir   = imdb.dir ;
      opts.imageNames = imageNames ;
      opts.jitterName = jitterName ;
      opts.featOpts   = conf.featOpts.(featName) ;
      opts.noClobber  = conf.noClobber ;

      if conf.featOpts.(featName).vocabSize > 0
        opts.wordDir    = fullfile(conf.ppDir, featName, 'words') ;
        opts.vocabPath  = fullfile(conf.ppDir, featName, 'vocab') ;
      else
        opts.featDir    = fullfile(conf.ppDir, featName, 'features') ;
      end

      [opts, waitJobs_quant(end+1)] = ...
          bk_calcFeatures(opts, ...
                          'parallelize', conf.parallelize, ...
                          'jobDir', conf.jobDir, ...
                          'waitJobs', [waitJobs waitJobs_extract waitJobs_vocab]) ;
    end
  end

  waitJobs_ = [waitJobs_ waitJobs_extract waitJobs_vocab waitJobs_quant] ;
end

if ~isempty(waitJobs_), waitJobs = waitJobs_ ; end
