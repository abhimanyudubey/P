function [conf, jobId] = bk_calcRoiHistograms(conf, varargin)
% BK_CALCROIHISTOGRAMS Compute feature histogram for multiple ROIs
%
%   histDir:       output dir
%   ppDir:         input (feature) dir
%   imageDbPath:   path to the image database
%   roidDbPath:    path to the ROI database
%   roiSel:        selection of ROIs to process ([] for all)
%   roiMagnif:     a magnification factor to enarge / shrink the
%                  effective ROI (0 for none)
%   className:     name of the class being processed (for feature
%                  that include class-specific settings)
%   normalize:     wether to l1 normalize the histograms%
%   noClobber:     avoid overwriting files
%
%   featNames:     which features to process
%   featOpts:      configuration of each feature
%
%   This block calculates the histogram for the specified ROI and
%   visual word feature types. For each ROI, a file with the ROI ID is
%   saved to HISTDIR. This file contains one or multiple histograms,
%   depending on the combination of feature types and pyramid levels.
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

conf_.histDir      = '' ;
conf_.ppDir        = '' ;
conf_.imageDbPath  = '' ;
conf_.roiDbPath    = '' ;
conf_.roiSel       = [] ;
conf_.className    = '' ;
conf_.featNames    = {} ;
conf_.featOpts     = struct ;
conf_.roiMagnif    = 0 ;
conf_.maxDenseDim  = +inf ;
conf_.normalize    = true ;
conf_.noClobber    = false ;

if nargin < 1, conf = conf_ ; return ; end
conf = override(conf_, conf, 'skip', 'warn') ;

[jobId, taskId] = parallelDriver('numNodes', 40, varargin{:}) ;
if ~isnan(jobId) & isnan(taskId) ; return ; end

ensuredir(conf.histDir) ;

roidb = getRoiDb(conf.roiDbPath) ;
imdb  = getImageDb(conf.imageDbPath) ;

imageIndex = [imdb.images.id] ;

if isempty(conf.roiSel)
  roiSel = 1:length([roidb.rois.id]) ;
else
  roiSel = conf.roiSel ;
end

% --------------------------------------------------------------------
%                                                   Compute histograms
% --------------------------------------------------------------------

numSkipped = 0 ;

for rsi = 1:length(roiSel)

  % try to lock the ROI
  locked = parallelLock(rsi, varargin{:}) ;
  if ~locked, continue ; end

  idx        = roiSel(rsi) ;
  roiId      = roidb.rois.id(idx) ;
  roiImageId = roidb.rois.imageId(idx) ;
  roiBox     = roidb.rois.box(:,idx) ;
  roiJitter  = roidb.rois.jitter(:,idx) ;

  % no clobber logic
  roiHistPath = fullfile(conf.histDir, sprintf('%012.0f', roiId)) ;
  if conf.noClobber & checkFile(roiHistPath)
    numSkipped = numSkipped + 1 ;
    continue ;
  end

  fprintf('Processing ROI ''%012.0f'' (%d of %d).\n', ...
          roiId, rsi, length(roiSel)) ;

  % find image name
  ii = binsearch(imageIndex, roiImageId) ;
  imageName = imdb.images.name{ii} ;

  % apply jitter
  if roiJitter > 0
    jitterName = decodeEnum(roidb.jitters, roiJitter) ;
    imageName = [imageName '_' lower(jitterName)] ;
  end

  % compute hist
  fprintf('\tComputing histograms.\n') ; tic ;
  roiHists = getAllRoiHists(conf, ...
                            imageName, ...
                            roiBox, ...
                            'className', conf.className, ...
                            'magnif', conf.roiMagnif, ...
                            'maxDenseDim', conf.maxDenseDim, ...
                            'normalize', conf.normalize) ;
  fprintf('\tComputed in %.2f s.\n', toc) ;

  % save
  roiHistPath = fullfile(conf.histDir, sprintf('%012.0f', roiId)) ;
  fprintf('\tSaving to ''%s''.\n', roiHistPath) ;
  ssave(roiHistPath, '-STRUCT', 'roiHists')  ;
end

fprintf('\t%d where skipped to avoid clobbering.\n', numSkipped) ;
