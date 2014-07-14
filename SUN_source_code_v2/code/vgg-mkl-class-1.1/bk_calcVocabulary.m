function [conf, jobId] = bk_calcVocabulary(conf, varargin)
% BK_CALCVOCABULARY  Comptue feature vocabulary (visual words)
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

conf_.imageDbPath    = '' ;
conf_.gtRoiDbPath    = '' ;
conf_.featDir        = '' ;
conf_.featOpts       = struct ;
conf_.vocabPath      = '' ;
conf_.trainSetName   = '' ;
conf_.focusOnClasses = [] ;
conf_.addBackgroundImages = false ;
conf_.noClobber      = false ;

if nargin == 0, conf = conf_ ; return ; end
conf = override(conf_, conf, 'warn') ;

[jobId, taskId] = parallelDriver(varargin{:}, 'numNodes', 1) ;
if ~isnan(jobId) & isnan(taskId) ; return ; end

% --------------------------------------------------------------------
%                                                     No-clobber logic
% --------------------------------------------------------------------

if conf.noClobber & checkFile(conf.vocabPath)
  fprintf('\tSkipping to avoid clobbering ''%s''.\n', conf.vocabPath) ;
  return ;
end

% --------------------------------------------------------------------
%                                            Determine training images
% --------------------------------------------------------------------

% We search for featOpts.numImagesPerClass images for each class
% (i.e. images which contain an instance of that class)

opts = conf.featOpts ;

imdb  = getImageDb(conf.imageDbPath) ;
roidb = getRoiDb(conf.gtRoiDbPath) ;

imageIndex = [imdb.images.id] ;
imageSelected = false(1, length(imdb.images.id)) ;
classNames = fieldnames(roidb.classes)' ;

if ~isempty(conf.focusOnClasses)
  classNames = conf.focusOnClasses ;
  classFilter = classNames{1} ;
  for ci=2:length(classNames)
    classFilter = [classFilter '|' classNames{ci}] ;
  end
end

% determine images containing the class of interest
for className = classNames
  className = char(className) ;

  % find TRAIN+VAL ROIs of this class
  sel = findRois(roidb, ...
                 'set', conf.trainSetName, ...
                 'jitter', 0, ...
                 'class', className) ;

  % find corresponding images by ID
  seli = binsearch(imageIndex, roidb.rois.imageId(sel)) ;

  % find the ones that are _not_ already selected
  seli = seli(~imageSelected(seli)) ;

  % select a random subset
  seli = randcol(seli, opts.numImagesPerClass) ;
  if (length(seli) < opts.numImagesPerClass)
    warning(sprintf('Could only get %d of %d for %s.', ...
                    length(seli), opts.numImagesPerClass, className)) ;
  end

  % include images
  imageSelected(seli) = true ;
end

% Add background images
if conf.addBackgroundImages
  classSel =  '' ;
  for ci = 1:length(classNames), classSel = [classSel classNames{ci} '|'] ; end
  classSel(end) = [] ;

  % find TRAIN that do not include any of the classes
  sel = findRois(roidb, ...
                 'set', conf.trainSetName, ...
                 'jitter', 0, ...
                 'class', classSel) ;

  % find corresponding images by ID
  seli = setdiff(1:length(imageIndex), binsearch(imageIndex, roidb.rois.imageId(sel))) ;

  % remove any which is not in the training set
  seli = findImages(imdb, 'subset', seli, 'set', conf.trainSetName) ;

  % find the ones that are _not_ already selected
  seli = seli(~imageSelected(seli)) ;

  % select a random subset
  seli = randcol(seli, opts.numImagesPerClass) ;
  if (length(seli) < opts.numImagesPerClass)
    warning(sprintf('Could only get %d of %d for BACKGROUND IMAGES.', ...
                    length(seli), opts.numImagesPerClass)) ;
  end

  % include images
  imageSelected(seli) = true ;
end

% --------------------------------------------------------------------
%                                                     Read descriptors
% --------------------------------------------------------------------

descrs = {} ;
for ii = find(imageSelected)
  imageName = imdb.images.name{ii} ;
  fprintf('\tLoading descriptors from ''%s.mat''.\n', imageName) ;
  feat = load(fullfile(conf.featDir, [imageName '.mat'])) ;
  self = 1:size(feat.frames,2) ;

  % optionally, filter features on classes
  if ~isempty(conf.focusOnClasses)
    selr = findRois(roidb, ...
                    'imageId', imdb.images.id(ii), ...
                    'class', classFilter) ;

    if isempty(selr)
      keep = true(1, size(feat.frames,2)) ;
      warning('%s does not contain class instances, sampling from whole image') ;
    else
      keep = false(1, size(feat.frames,2));
      box  = roidb.rois.box(:, selr) ;

      for bi = 1:size(box,2)
        keep = keep | ...
               box(1,bi) <= feat.frames(1,:) & ...
               box(2,bi) <= feat.frames(2,:) & ...
               box(3,bi) >= feat.frames(1,:) & ...
               box(4,bi) >= feat.frames(2,:) ;
      end
    end

    self = self(keep) ;
  end
  descrs{end+1} = randcol(feat.descrs(:, self), opts.numFeatsPerImage) ;
end
descrs = cat(2, descrs{:}) ;

% --------------------------------------------------------------------
%                                                     Train Vocabulary
% --------------------------------------------------------------------

fprintf('\tRunning ''%s'' on %d descriptors.\n', ...
        func2str(opts.clusterFn), size(descrs,2)) ;

V = feval(opts.clusterFn, ...
          opts, ...
          descrs) ;

% --------------------------------------------------------------------
%                                                               Finish
% --------------------------------------------------------------------

fprintf('\tSaving vocabulary to ''%s''.\n', conf.vocabPath) ;
ssave(conf.vocabPath, 'V') ;
