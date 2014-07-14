function [conf, jobId] = bk_packDistMatrix(conf, varargin)
% BK_PACKGBDISTANCES
%
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

conf_.blockPaths     = '' ;
conf_.distMatrixPath = '' ;
conf_.kerPath        = '' ;
conf_.kerDescrPath   = '' ;
conf_.testMode       = false ;
conf_.noClobber      = false ;

if nargin < 1, conf = conf_ ; return ; end
conf = override(conf_, conf) ;

[jobId, taskId] = parallelDriver(varargin{:}, 'numNodes', 1) ;
if ~isnan(jobId) & isnan(taskId) ; return ; end

canSkip = conf.noClobber ;
canSkip = canSkip & (exist(conf.distMatrixPath, 'file') || ...
                     exist([conf.distMatrixPath '.mat'], 'file')) ;
if ~isempty(conf.kerPath)
  canSkip = canSkip & (exist(conf.kerPath, 'file') || ...
                       exist([conf.kerPath '.mat'], 'file')) ;
end
if canSkip
  fprintf('Skipping to avoid clobbeing.\n') ;
  return ;
end

% load blocks
numRows = 0 ;
numCols = 0 ;
blocks = {} ;
for bi = 1:length(conf.blockPaths)
  blocks{bi} = load(conf.blockPaths{bi}) ;
  numRows = max(numRows, blocks{bi}.rowEnd) ;
  numCols = max(numCols, blocks{bi}.colEnd) ;  
end

% fill matrix
matrix = zeros(numRows, numCols) ;
for bi = 1:length(blocks)
  block = blocks{bi} ;
  rs = block.rowStart ;
  re = block.rowEnd ;
  cs = block.colStart; 
  ce = block.colEnd ;
  matrix(rs:re, cs:ce) = block.matrix ;
end
 
% save back
fprintf('Saving ''%s''.\n', ...
        conf.distMatrixPath) ;
ssave(conf.distMatrixPath, 'matrix') ;

% optionally compute kernel
if ~isempty(conf.kerPath)
  
  if ~isempty(conf.kerDescrPath) & conf.testMode
    fprintf('Loading ''%s''.\n', conf.kerDescrPath) ;
    descr = load(conf.kerDescrPath) ;
    mu = descr.mu ;
  else
    mu = 1 / mean(matrix(:)) ;
  end
  
  matrix = exp(- mu * matrix) ;
  
  ker.matrix = matrix ;
  ker.type = 'el2'; 
  ker.feat = 'gb' ;
  ker.pyrLevel = [] ;
  ker.histName = '' ;
  ker.name     = 'el2_gb' ;
  ker.mu       = mu ;
  
  % save back
  fprintf('Saving ''%s''.\n', ...
          conf.kerPath) ;
  ssave(conf.kerPath, '-STRUCT', 'ker') ;

  if ~isempty(conf.kerDescrPath) & ~conf.testMode
    fprintf('Saving ''%s''.\n', ...
            conf.kerDescrPath) ;
    ker = rmfield(ker, 'matrix') ;
    ssave(conf.kerDescrPath, '-STRUCT', 'ker') ;
  end
  
end

