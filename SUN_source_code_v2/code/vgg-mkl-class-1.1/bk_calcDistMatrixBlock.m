function [conf, jobId] = bk_calcDistMatrixBlock(conf, varargin)
% BK_CALCDISTMATRIXBLOCK
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

conf_.gbPaths      = '' ;
conf_.gbTestPaths  = '' ;
conf_.blockDefs    = struct ;
conf_.blockPaths   = '' ;
conf_.noClobber    = false ;
conf_.distFn       = [] ;

if nargin < 1, conf = conf_ ; return ; end
conf = override(conf_, conf) ;

[jobId, taskId] = parallelDriver(varargin{:}, 'numNodes', length(conf.blockDefs)) ;
if ~isnan(jobId) & isnan(taskId) ; return ; end

for bi = 1:length(conf.blockDefs)

  % try to lock the Block
  locked = parallelLock(bi, varargin{:}) ;
  if ~locked, continue ; end

  block = conf.blockDefs(bi) ;
  blockPath = conf.blockPaths{bi} ;

  fprintf('Computing block %d (%d-%d, %d-%d).\n', ...
          bi, block.rowStart, block.rowEnd, ...
          block.colStart, block.colEnd) ;

  % No clobber logic
  if conf.noClobber & checkFile(blockPath)
    sprintf('\tSkipping to avoid clobbering ''%''.\n', blockPath) ;
    continue ;
  end

  % Load descriptors for range of images in block
  row = block.rowStart : block.rowEnd ;
  col = block.colStart : block.colEnd ;

  clear rowDescrs ;
  clear colDescrs ;

  for i = 1:length(row)
    data = load(conf.gbPaths{row(i)}) ;
    rowDescrs{i} = data.descrs ;
  end

  for j = 1:length(col)
    if ~isempty(conf.gbTestPaths)
      data = load(conf.gbTestPaths{col(j)}) ;
    else
      data = load(conf.gbPaths{col(j)}) ;
    end
    colDescrs{j} = data.descrs ;
  end

  % Compute distance
  tic
  if 1
    block.matrix = feval(conf.distFn, rowDescrs, colDescrs) ;
  else
    block.matrix = zeros(length(row), length(col)) ;
    for i = 1:length(row)
      for j = 1:length(col)
        da = rowDescrs{i} ;
        db = colDescrs{j} ;
        block.matrix(i,j) = feval(conf.distFn,da,db) ;
      end
      fprintf('%5.2f %%: %5.1fm remaining.\n', ...
              i/length(row)*100, ...
              toc / i * (length(row)-i) / 60) ;
    end
  end

  % Save block
  ensuredir(fileparts(blockPath)) ;
  fprintf('Saving block to ''%s''.\n', blockPath) ;
  ssave(blockPath, '-STRUCT', 'block') ;
end
