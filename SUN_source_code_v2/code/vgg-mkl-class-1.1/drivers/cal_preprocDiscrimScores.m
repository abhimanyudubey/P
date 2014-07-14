% CAL_PREPROCDISCRIMSCORES
%
%   CAL_PREPROCDISCRIMSCORES compresses and computes the
%   discriminative scores for the BOW features.
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

cal_conf ;

featName  = 'oldbow3' ;

% Load training data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
imdb  = getImageDb(conf.imageDbPath) ;
roidb = getRoiDb(conf.gtRoiDbPath) ;

numWords      = conf.featOpts.(featName).vocabSize ;
numClasses    = length(fieldnames(roidb.classes)) ;

sel = calcImageSet(conf.imageDbPath, conf.trainSetName) ;
occur = zeros(numWords, numClasses + 1) ;
joint = zeros(numWords, numClasses) ;
docOccur = zeros(numWords, 1) ;

% --------------------------------------------------------------------
%                                                            Scan ROIs
% --------------------------------------------------------------------

numDocs = 0 ;

for ii=1:length(sel)
  s = sel(ii) ;

  % pick next image ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  imagePath = fullfile(imdb.dir, [imdb.images(s).name '.jpg']) ;
  imageName = imdb.images(s).name ;
  imageId   = imdb.images(s).id ;
  imageSize = imdb.images(s).size ;
  fprintf('Processing ''%s''.\n', imageName) ;

  wordPath = fullfile(conf.ppDir, featName, 'words', imageName) ;

  % find ROIs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  selr  = findRois(roidb, 'imageId', imageId, 'jitter', 0) ;
  boxes = uint32(roidb.rois.box(:,selr)) ;

  % count words for each ROI ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  hists = getRoiHists(wordPath, ...
                      conf.featOpts.(featName), ...
                      0, ...
                      [boxes [1;1;imageSize']], ...
                      'normalize', false) ;

  hists = hists{1} ;

  for ri=1:length(selr)
    ci = roidb.rois.class(selr(ri)) ;
    numDocs = numDocs + 1 ;
    occur(:,ci) = occur(:,ci) + (hists(:, ri) > .5) ;
    joint(:,ci) = joint(:,ci) + hists(:,ri) ;
    docOccur = docOccur + (hists(:,ri) > .5) ;
  end
  occur(:,end) = occur(:,end) + (hists(:,end) > .5) ;
end

idf = log(numDocs ./ (docOccur + eps)) ;
ssave(fullfile(conf.ppDir, featName, 'idf'), 'idf') ;
