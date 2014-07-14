function [sel, labels] = calcImageSet(imageDbPath, imageSet, varargin)
% CALCIMAGESET
%   SEL = CALCIMAGESET(IMAGEDBPATH, IMAGESET)
%
%   Images are returned in order (positive first).
%
%   Author:: Andrea VEdaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

conf_.roiDbPath      = '' ;
conf_.className      = '' ;
conf_.aspectName     = '' ;
conf_.numPos         = NaN ;
conf_.numNeg         = NaN ;

conf_.antClassName   = '' ;
conf_.antFraction    = 0 ;

conf_.numParts       = 1 ;

conf = vl_argparse(conf_, varargin{:}) ;

randn('state',0) ;
rand('state',0) ;

% collect image indeces in conf.imageSet ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
imdb = getImageDb(imageDbPath) ;
sel = [] ;

for setName = fieldnames(imdb.sets)'
  setName = char(setName) ;
  str = regexpi(setName, imageSet, 'match', 'once') ;
  if length(str) == length(setName)
    sel = [sel find([imdb.images.set] == imdb.sets.(setName))] ;
  end
end
sel = unique(sel) ;
fprintf('calcImageSet:\n');
fprintf('\tpool size: %d (%s)\n', length(sel), imageSet) ;

% restrict to a subset of images ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if ~isempty(conf.roiDbPath)
  roidb = getRoiDb(conf.roiDbPath) ;
  imageIndex = [imdb.images.id] ;

  fprintf('\trequested %.0f pos and %.0f neg images\n', ...
          conf.numPos, conf.numNeg) ;

  selp = findRois(roidb, ...
                  'class', conf.className, ...
                  'jitter', 0, ...
                  'aspect', conf.aspectName, ...
                  'difficult', 0) ;
  selpo = findRois(roidb, ...
                   'class', conf.className, ...
                   '~aspect', conf.aspectName, ...
                   'difficult', 0) ;
  sela = findRois(roidb, ...
                  '~class', conf.className, ...
                  'class', conf.antClassName) ;
  fprintf('\tclass: %s aspect: %s\n', conf.className, conf.aspectName) ;
  fprintf('\tantagonist classes: %s\n', conf.antClassName) ;

  selp   = roidb.rois.imageId(selp) ;
  selpo  = roidb.rois.imageId(selpo) ;
  sela   = roidb.rois.imageId(sela) ;
  selp   = intersect(sel, unique(binsearch(imageIndex, selp))) ;
  selpo  = intersect(sel, unique(binsearch(imageIndex, selpo))) ;
  sela   = intersect(sel, unique(binsearch(imageIndex, sela))) ;

  selpo = setdiff(selpo, selp) ;
  sela  = setdiff(sela, [selp selpo]) ;
  seln  = setdiff(sel,  [selp selpo sela]) ;

  fprintf('\tfound: %d images matching class and aspect\n', length(selp)) ;
  fprintf('\tfound: %d images matching only the class\n', length(selpo)) ;
  fprintf('\tfound: %d images matching antagonists classes\n', length(sela)) ;
  fprintf('\tfound: %d other images\n', length(seln)) ;

  numPos = min(conf.numPos, length(selp) + length(selpo)) ;
  numAnt = min(round(conf.numNeg * conf.antFraction), length(sela)) ;
  numNeg = min(conf.numNeg - numAnt, length(seln)) ;

  fprintf('\teach partition includes:\n\t\t%d positive (%d from other aspect)\n\t\t%d negative (%d antagonist)\n', ...
          numPos, max(numPos - length(selp),0), numNeg+numAnt, numAnt) ;

  % add incorrect aspect to positive to fill necessity
  selpo = shuffle(selpo) ;
  selp = [selp selpo(1: numPos - length(selp))] ;

  % now create partitions
  selp = shuffle(selp) ;
  seln = shuffle(seln) ;
  sela = shuffle(sela) ;

  cursPos = 1 ;
  cursNeg = 1 ;
  cursAnt = 1 ;

  clear sel ;
  fprintf('\tcreating %d partitions.\n', conf.numParts) ;
  for pi = 1:conf.numParts
    selp_ = mod(cursPos + (1:numPos) - 1, length(selp)) + 1 ;
    seln_ = mod(cursNeg + (1:numNeg) - 1, length(seln)) + 1 ;
    sela_ = mod(cursAnt + (1:numAnt) - 1, length(sela)) + 1 ;
    selp_ = selp(selp_) ;
    seln_ = seln(seln_) ;
    sela_ = sela(sela_) ;
    cursPos = cursPos + numPos ;
    cursNeg = cursNeg + numNeg ;
    cursAnt = cursAnt + numAnt ;
    sel{pi} = [selp_ seln_ sela_] ;
  end
end

% --------------------------------------------------------------------
function A = shuffle(A)
% --------------------------------------------------------------------
A = A(:, randperm(size(A,2))) ;
