function [conf, jobId] = bk_calcFeatures(conf, varargin)
% BK_CALCFEATURES  Compute (quantized) features for multiple images
%
%  CONF parameters:
%
%    imageNames:   name of the images to process (without .jpg suffix).
%    imageDir:     directory containing the jpg images.
%    featDir:      output feature directory.
%    wordDir:      output quantized feature directory (optional).
%    featOpts:     feature options (defininig the features to process).
%    jitterName:   which jitter to apply to images.
%    noClobber:    TRUE to avoid clobbering existing files.
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

conf_.imageNames  = {} ;
conf_.imageDir    = '' ;
conf_.featDir     = '' ;
conf_.wordDir     = '' ;

conf_.featOpts    = struct ;
conf_.jitterName  = '' ;
conf_.vocabPath   = '' ;
conf_.noClobber   = false ;

if nargin == 0, conf = conf_ ; return ; end
conf = override(conf_, conf) ;

[jobId, taskId] = parallelDriver(varargin{:}, 'numNodes', 5) ;
if ~isnan(jobId) & isnan(taskId) ; return ; end

if ~ isempty(conf.featDir), ensuredir(conf.featDir) ; end
if ~ isempty(conf.wordDir), ensuredir(conf.wordDir) ; end

% --------------------------------------------------------------------
%                                                   Process all images
% --------------------------------------------------------------------

rand('state', vl_getpid) ;
randn('state', vl_getpid) ;
perm = randperm(length(conf.imageNames)) ;

for ii = perm

  % try to lock the image
  locked = parallelLock(ii, varargin{:}) ;
  if ~locked, continue ; end

  imageName = conf.imageNames{ii} ;
  imagePath = fullfile(conf.imageDir, imageName) ;

  if ~isempty(conf.jitterName)
    imageName = [imageName '_' conf.jitterName] ;
  end

  fprintf('Processing ''%s''.\n', imageName) ;

  % ------------------------------------------------------------------
  %                                                   No clobber logic
  % ------------------------------------------------------------------

  skip = logical(conf.noClobber) ;
  if ~isempty(conf.featDir)
    skip = skip & checkFile(fullfile(conf.featDir, [imageName '.mat'])) ;
  end
  if ~isempty(conf.wordDir)
    skip = skip & checkFile(fullfile(conf.wordDir, [imageName '.mat'])) ;
  end
  if skip
    fprintf('\tSkipping to avoid clobbering.\n') ;
    continue ;
  end

  % ------------------------------------------------------------------
  %                                              Read and jitter image
  % ------------------------------------------------------------------

  fprintf('\tReading image ''%s''.\n', imagePath) ;
  im = getImage(imagePath) ;
  im = im2double(im) ;

  if ~isempty(conf.jitterName)
    fprintf('\tApplying transformation ''%s''.\n', conf.jitterName) ;
    im = tfImage(conf.jitterName, im) ;
  end

  % ------------------------------------------------------------------
  %                                                         Extraction
  % ------------------------------------------------------------------

  fprintf('\tRunning ''%s''.\n', func2str(conf.featOpts.extractFn)) ;
  feat = feval(conf.featOpts.extractFn, conf.featOpts, im) ;

  if ~isempty(conf.featDir)
    featPath = fullfile(conf.featDir, imageName);
    fprintf('\tSaving feature file ''%s''.\n', featPath) ;
    ssave(featPath, '-STRUCT', 'feat') ;
  end

  % ------------------------------------------------------------------
  %                                                       Quantization
  % ------------------------------------------------------------------

  if ~ isempty(conf.wordDir)
    fprintf('\tRunning ''%s''.\n', func2str(conf.featOpts.quantizeFn)) ;
    vocab = getVocab(conf.vocabPath) ;
    words = feval(conf.featOpts.quantizeFn, ...
                  vocab, ...
                  feat.descrs) ;
    feat = rmfield(feat, 'descrs') ;
    feat.words = words ;

    switch conf.featOpts.format
      case 'dense'
        feat = sparse2dense(feat) ;
    end

    wordPath = fullfile(conf.wordDir, imageName) ;
    fprintf('\tSaving word file ''%s''.\n', wordPath) ;
    ssave(wordPath, '-STRUCT', 'feat') ;
  end

end % next bit
