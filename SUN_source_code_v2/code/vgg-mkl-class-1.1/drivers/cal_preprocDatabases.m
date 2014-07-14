% CAL_PREPROCDATABASES Prepare the image and ROI databases
%  CAL_PREPORCDATABASES prepares the image and ROI databases for the
%  Caltech-101 dataset.
%
%  The function expects conf.calDir to point to a directory with
%  images preprocessed by the shell script preprocCal101.sh. It then
%  extracts all the images and class names, building an image and ROI
%  (region of interest) database.
%
%  Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

% source configuration
cal_conf ;

randn('state', conf.masterRandomSeed) ;
rand('state',  conf.masterRandomSeed) ;

% noClobber logic
if checkFile(conf.imageDbPath) & checkFile(conf.gtRoiDbPath)
  fprintf('Skipping to avoid clobbering ''%s'' and ''%s''.\n', ...
          conf.imageDbPath, conf.gtRoiDbPath) ;
end

% ------------------------------------------------------------------
%                                           Scan directory of images
% ------------------------------------------------------------------

contents = dir(fullfile(conf.calDir, '*.jpg')) ;

clear imdb ;

imdb.dir = conf.calDir ;
imdb.sets.TRAIN = uint8(1) ;
imdb.sets.TEST  = uint8(2) ;

roidb.sets = imdb.sets ;
roidb.aspects.MISC = uint8(1) ;
roidb.jitters = struct ;
roidb.classes = struct ;
roidb.rois = struct([]) ;

rois = {} ;

classes = struct ;
imageName = {} ;
imageClass = [] ;
numClasses = 0 ;

% scan images for classes
ii = 0 ;
for ci=1:length(contents)
  name = contents(ci).name ;
  mo = regexp(name, '(?<class>[\w_]+)_(?<image>image_\w+)\.jpg', 'names') ;
  linkName = [mo.class '_' mo.image]   ;
  className = upper(mo.class) ;

  % see if this adds a new class in
  if ~ isfield(roidb.classes, className)
    numClasses = numClasses + 1 ;
    if conf.smallData & numClasses > 3
      continue ;
    end
    roidb.classes.(className) = uint8(length(fieldnames(roidb.classes)) + 1) ;
  end

  ii = ii + 1 ;
  imageName{ii}  = linkName ;
  imageClass(ii) = roidb.classes.(className) ;
end

% ------------------------------------------------------------------
%                                                   Split train test
% ------------------------------------------------------------------

dbi = 0 ;
for ci=1:numClasses

  sel = find(imageClass == ci) ;
  sel = sel(randperm(length(sel))) ;

  for k = 1:min(conf.masterNumTrain + 15, length(sel))
    ii = sel(k) ;
    info = imfinfo(fullfile(imdb.dir, [imageName{ii} '.jpg'])) ;

    dbi = dbi + 1 ;
    imdb.images.id(dbi) = dbi ;
    imdb.images.size(:,dbi) = [info.Width; info.Height] ;
    imdb.images.name{dbi} = imageName{ii} ;

    if k <= conf.masterNumTrain
      imdb.images.set(dbi) = imdb.sets.TRAIN ;
    else
      imdb.images.set(dbi) = imdb.sets.TEST ;
    end

    roi = newRoi ;
    roi.id = length(rois) + 1 ;
    roi.set = imdb.images.set(dbi) ;
    roi.imageId = imdb.images.id(dbi) ;
    roi.class = imageClass(ii) ;
    roi.aspect = roidb.aspects.MISC ;
    roi.box = [1;1;imdb.images.size(:,dbi)] ;
    rois{end+1} = roi  ;
  end
end

clear imageClass imageName ;

% ------------------------------------------------------------------
%                                                        Add jitters
% ------------------------------------------------------------------

ji_rois = {} ;
for ji = 1:length(conf.jitterNames)
  jitterName = upper(conf.jitterNames{ji}) ;
  roidb.jitters.(jitterName) = uint8(ji) ;

  imageIndex = [imdb.images.id] ;
  for ri = 1: length(rois)
    roi = rois{ri} ;

    ii = binsearch(imageIndex, roi.id) ;
    imageSize = imdb.images.size(:,ii) ;

    if roi.set ~= roidb.sets.TRAIN, continue ; end

    roi.id  = roi.id + 1e6 * double(roidb.jitters.(jitterName)) ;
    roi.box = tfBox(imageSize(1), imageSize(2), jitterName, roi.box) ;
    roi.jitter = roidb.jitters.(jitterName) ;

    ji_rois{end+1} = roi ;
  end
end

rois = {rois{:}, ji_rois{:}} ;

% ------------------------------------------------------------------
%                                                    Finish and save
% ------------------------------------------------------------------

rois = cat(2, rois{:}) ;

fprintf('\tSorting ROIs by id.\n') ;
[drop,perm] = sort([rois.id]) ;
rois = rois(perm) ;

fprintf('\tSwitching to structure-of-array representation.\n') ;
roidb.rois = aosToSoa(rois) ;
clear rois ;

fprintf('\tSaving Image DB to ''%s''.\n', conf.imageDbPath) ;
ssave(conf.imageDbPath, '-STRUCT', 'imdb') ;

fprintf('\tSaving ROI DB to ''%s''.\n', conf.gtRoiDbPath) ;
ssave(conf.gtRoiDbPath, '-STRUCT', 'roidb') ;
