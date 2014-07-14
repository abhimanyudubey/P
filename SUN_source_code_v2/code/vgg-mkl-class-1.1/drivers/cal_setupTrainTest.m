% CAL_SETUPTRAINTEST
%  CAL_SETUPTRAINTEST takes the ROI database containing all the
%  available train and test data and select a subset of those for an
%  experiment.
%
%  Possible variations include: number of training / testing data,
%  using jittered data, excluding some classes.
%
%  Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

rand('state',0) ;
randn('state',0) ;

% --------------------------------------------------------------------
%                                             Load DBs and create dirs
% --------------------------------------------------------------------

imdb = getImageDb(conf.imageDbPath) ;
roidb = getRoiDb(conf.gtRoiDbPath) ;

stageDir       = fullfile(conf.trainDir, 'COMMON', conf.expPrefix) ;
kerDir         = fullfile(conf.aggrDir,  'kers') ;
histDir        = fullfile(conf.aggrDir,  'hists') ;
trainRoiDbDir  = fullfile(conf.aggrDir,  'rois') ;
testRoiDbDir   = fullfile(conf.aggrDir,  'rois-test') ;
trainRoiDbPath = fullfile(trainRoiDbDir, 'roidb') ;
testRoiDbPath  = fullfile(testRoiDbDir,  'roidb') ;

ensuredir(kerDir) ;
ensuredir(trainRoiDbDir) ;
ensuredir(testRoiDbDir) ;

% --------------------------------------------------------------------
%                                               Select training images
% --------------------------------------------------------------------

classNames = fieldnames(roidb.classes)' ;
selTrain = [] ;
selTrainJit = [] ;

% Pick a certain number of images per class, possibly with jitter
for className = classNames
  className = char(className) ;

  % select numPos training images
  sel    = findRois(roidb, ...
                    'set', conf.trainSetName, ...
                    'jitter', 'none', ...
                    'class', className) ;
  sel = randcol(sel, conf.numPos, 'beginning') ;

  % select numJitPost jittered training images, but only from images
  % that have been included already (otherwise this would enlarge the
  % dataset).
  selJit = findRois(roidb, ...
                    'set', conf.trainSetName, ...
                    '~jitter', 'none', ...
                    'class', className) ;
  keep = ismember(roidb.rois.imageId(selJit), roidb.rois.imageId(sel)) ;
  selJit = randcol(selJit(keep), conf.numJitPos, 'uniform') ;

  selTrain    = [selTrain,    sel] ;
  selTrainJit = [selTrainJit, selJit] ;
end
selTrain = sort([selTrain selTrainJit]) ;

trainRoidb = roidb ;
trainRoidb.rois = soaSubsRef(trainRoidb.rois, selTrain) ;

fprintf('Saving ''%s''.\n', trainRoiDbPath) ;
ssave(trainRoiDbPath, '-STRUCT', 'trainRoidb') ;

clear selTrain selTrainJit ;

% --------------------------------------------------------------------
%                                                   Select test images
% --------------------------------------------------------------------

selTest = findRois(roidb, ...
                   'set', conf.testSetName, ...
                   'jitter', 0) ;

testRoidb = roidb ;
testRoidb.rois = soaSubsRef(testRoidb.rois, selTest) ;
keep = false(1, length(testRoidb.rois.id))  ;
for cl = fieldnames(roidb.classes)'
  selCla = findRois(testRoidb, 'class', char(cl)) ;
  keep(selCla(1 : min(15, length(selCla)))) = true ;
end
testRoidb.rois = soaSubsRef(testRoidb.rois, find(keep)) ;

fprintf('Saving ''%s''.\n', testRoiDbPath) ;
ssave(testRoiDbPath, '-STRUCT', 'testRoidb') ;

% convert roi selection in to image selection
imageIndex = [imdb.images.id] ;
selImageTrain = binsearch(imageIndex, [trainRoidb.rois.imageId]) ;
selImageTest  = binsearch(imageIndex, [testRoidb.rois.imageId]) ;
