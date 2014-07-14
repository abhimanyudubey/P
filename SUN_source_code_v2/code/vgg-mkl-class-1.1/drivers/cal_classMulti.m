% CAL_CLASSMULTI
%   Gather one-vs-rest results and compute confusion matris.
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

cal_conf ;

waitJobs = [] ;

imdb = getImageDb(conf.imageDbPath) ;
roidb = getRoiDb(conf.gtRoiDbPath) ;
trainImageSel = findImages(imdb, 'set', imdb.sets.TRAIN) ;
testImageSel  = findImages(imdb, 'set', imdb.sets.TEST) ;

conf.do_packScores = 1 ;
conf.do_evalScores = 1 ;

% --------------------------------------------------------------------
%                                                             Run jobs
% --------------------------------------------------------------------

aggrRoiDbDir       = fullfile(conf.aggrDir, 'rois') ;
aggrTestRoiDbDir   = fullfile(conf.aggrDir, 'rois-test') ;
aggrRoiDbPath      = fullfile(aggrRoiDbDir, 'roidb') ;
aggrTestRoiDbPath  = fullfile(aggrTestRoiDbDir, 'roidb') ;

commDir = fullfile(conf.trainDir, 'COMMON', conf.expPrefix) ;

tsdb = load(aggrTestRoiDbPath) ;
classIds = unique(tsdb.rois.class) ;
numClasses = length(classIds) ;
numImages  = length(tsdb.rois.id) ;
classNames = fieldnames(roidb.classes)' ;

% --------------------------------------------------------------------
%                                                          Pack scores
% --------------------------------------------------------------------

if conf.do_packScores

  scores = zeros(numClasses, numImages) ;

  for ci = 1:length(classNames)
    className = classNames{ci} ;

    stageDir = fullfile(conf.trainDir, upper(className), conf.expPrefix) ;
    testRoiDbDir = fullfile(stageDir, 'rois-test') ;
    testRoiScorePath = fullfile(testRoiDbDir, 'test-scores') ;

    data = load(testRoiScorePath) ;

    i = find(classIds == tsdb.classes.(upper(className))) ;
    scores(i, :) = data.scores ;
  end
  ssave(fullfile(commDir, 'scoreMatrix'), 'scores', 'classIds') ;
end


% --------------------------------------------------------------------
%                                                            Confusion
% --------------------------------------------------------------------

if conf.do_evalScores

  % get scores
  load(fullfile(commDir, 'scoreMatrix')) ;

  testConf = zeros(numClasses) ;

  for ri = 1:size(scores,2)
    [drop, predClass] = max(scores(:, ri)) ;
    gtClass = find(classIds == tsdb.rois.class(ri)) ;
    testConf(gtClass, predClass) = testConf(gtClass, predClass) + 1 ;
  end

  confPath = fullfile(commDir, 'conf') ;
  fprintf('Saving confusion matrix to ''%s''.\n', confPath) ;
  ssave(confPath, 'testConf') ;

  testConf = testConf ./ (sum(testConf,2) * ones(1, numClasses)) ;

  figure(1) ;
  imagesc(testConf) ;
  title(sprintf('Confusion matrix (%d training images/cat, %.2f%% accuracy)', ...
                conf.masterNumTrain, mean(diag(testConf)) * 100)) ;
  axis equal ;
  axis tight ;
end