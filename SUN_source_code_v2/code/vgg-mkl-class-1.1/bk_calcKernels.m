function [conf, jobId] = bk_calcKernels(conf, varargin)
% BK_CALCKERNELS  Compute kerenel matrices
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.


conf_.histDir      = '' ;
conf_.testHistDir  = '' ;
conf_.kerDir       = '' ;
conf_.kerDb        = struct ;
conf_.kerDescrDir  = '' ;
conf_.noClobber    = false ;

if nargin < 1, conf = conf_ ; return ; end
conf = override(conf_, conf) ;

[jobId, taskId] = parallelDriver(varargin{:}, 'numNodes', length(conf.kerDb)) ;
if ~isnan(jobId) & isnan(taskId) ; return ; end

ensuredir(conf.kerDir) ;

for ki = 1:length(conf.kerDb)

  % try to lock the kernel
  locked = parallelLock(ki, varargin{:}) ;
  if ~locked, continue ; end

  ker = conf.kerDb(ki) ;

  fprintf('Calculating kernel ''%s''.\n', ker.name) ;

  if isempty(conf.testHistDir)
    % ------------------------------------------------------------------
    %                                              Kernel on train-train
    % ------------------------------------------------------------------

    % no clobber logic
    kerPath = fullfile(conf.kerDir, ker.name) ;
    if conf.noClobber & checkFile(kerPath)
      fprintf('\tSkipping to avoid clobbering ''%s''.\n', kerPath) ;
      continue ;
    end

    % load histograms
    histPath = fullfile(conf.histDir, ['hist-' ker.histName]) ;
    fprintf('\tLoading ''%s''.\n', histPath) ;
    hists = load(histPath) ;
    hists = hists.hists ;

    % compute kernel
    ker = calcKernel(ker, hists) ;

    % save
    kerPath = fullfile(conf.kerDir, ker.name) ;
    fprintf('\tSaving kernel ''%s''.\n', kerPath) ;
    ssave(kerPath, '-STRUCT', 'ker', '-v7.3') ;

    ker = rmfield(ker, 'matrix') ;

    % optionally save the kernel descriptor (includes gamma for the RBF)
    if ~isempty(conf.kerDescrDir)
      kerDescrPath = fullfile(conf.kerDir, ['descr-' ker.name]) ;
      fprintf('\tSaving kernel descriptor ''%s''.\n', kerDescrPath) ;
      ssave(kerDescrPath, '-STRUCT', 'ker', '-v7.3') ;
    end

  else
    % ------------------------------------------------------------------
    %                                               Kernel on train-test
    % -----------------------------------------------------------------

    kerPath = fullfile(conf.kerDir, ['test-' ker.name]) ;
    if conf.noClobber & checkFile(kerPath)
      fprintf('\tSkipping to avoid clobbering ''%s''.\n', kerPath) ;
      continue ;
    end

    % load hisstograms
    histPath = fullfile(conf.histDir, ['hist-' ker.histName]) ;
    fprintf('\tLoading ''%s''.\n', histPath) ;
    hists = load(histPath) ;
    hists = hists.hists ;

    testHistPath = fullfile(conf.testHistDir, ['hist-' ker.histName]) ;
    fprintf('\tLoading ''%s''.\n', testHistPath) ;
    testHists = load(testHistPath) ;
    testHists = testHists.hists ;

    % try to load the descriptor (to get variable kernel parameters such
    % as gamma for the RBF).
    if ~isempty(conf.kerDescrDir)
      kerDescrPath = fullfile(conf.kerDescrDir, ['descr-' ker.name]) ;
      fprintf('\tLoading kernel descriptor ''%s''.\n', kerDescrPath) ;
      ker = load(kerDescrPath) ;
    end

    % compute kernel
    testKer = calcKernel(ker, hists, testHists) ;

    % save
    kerPath = fullfile(conf.kerDir, ['test-' ker.name]) ;
    fprintf('\tSaving kernel ''%s''.\n', kerPath) ;
    ssave(kerPath, '-STRUCT', 'testKer', '-v7.3') ;
  end

end
