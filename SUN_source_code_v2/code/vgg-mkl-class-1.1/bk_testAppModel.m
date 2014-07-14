function [conf, jobId] = bk_testAppModel(conf, varargin)
% BK_TESTAPPMODEL
%
%  modelDirs:    one for each class
%  roiDir:       where to find GT rois
%
%  Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

conf_.kerDir         = '' ;
conf_.kerDb          = struct ;
conf_.trainHistDir   = '' ;
conf_.testRoiDbPath  = '' ;
conf_.testHistDir    = '' ;
conf_.modelPath      = '' ;
conf_.noClobber      = false ;

if nargin == 0, conf = conf_ ; return ; end
conf = override(conf_, conf, 1) ;

[jobId, taskId] = parallelDriver('numNodes', 1, ...
                                 'freeMemMb', 2 * 1024, ...
                                 varargin{:} );

if ~isnan(jobId) & isnan(taskId) ; return ; end

% --------------------------------------------------------------------
%                                      Process all classes and aspects
% --------------------------------------------------------------------

rand('state', 0) ;
randn('state', 0) ;

scorePath = fullfile(conf.testHistDir, 'test-scores.mat') ;
if conf.noClobber & checkFile(scorePath)
  fprintf('\tSkipping to avoid clobbering ''%s''.\n', ...
          scorePath) ;
  return ;
end
  
% --------------------------------------------------------------------
%                                                          Load pieces
% --------------------------------------------------------------------

% Test ROI DB ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fprintf('\tLoading testing ROI DB ''%s''.\n', ...
        conf.testRoiDbPath) ;
tsdb = load(conf.testRoiDbPath) ;

% Model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fprintf('\tLoading appearance model ''%ss''.\n', ...
        conf.modelPath) ;
model  = load(conf.modelPath) ;

% Kernel ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
base = loadTestMixedKernel(conf.kerDir, conf.kerDb, model.d, model.svind) ;

% ------------------------------------------------------------------
%                                                            Test it
% ------------------------------------------------------------------

scores = model.alphay' * base + model.b ;

scorePath = fullfile(conf.testHistDir, 'test-scores.mat') ;
fprintf('\tSaving scores to ''%s''.\n', scorePath) ;
ssave(scorePath, 'scores') ;

figure(100) ; clf ; hold on ;
[recall,precision,info] = pr(tsdb.labels, scores) ;
plot(recall,precision,'b-','linewidth',2) ;
grid on ;
xlim([0 1]) ;
ylim([0 1]) ;
xlabel('recall') ;
ylabel('precision') ;
title(sprintf('PR on test ROI DB - AUC: %.2f %% (%.2f %%)', ...
              info.auc, info.auc_pa08)) ;
axis square ;
drawnow ;

figPath = fullfile(conf.testHistDir, 'test-pr.eps') ;
printsize(.97 + 0.001*randn) ; % ugly hack for matlab repeat 
print('-depsc', figPath) ;

% ------------------------------------------------------------------
function base =  loadTestMixedKernel(kerDir, kernels, weights, svs)
% ------------------------------------------------------------------

base = [] ;
K = length(kernels) ;
for ki = 1:K
  kerPath = fullfile(kerDir, ['test-' kernels(ki).name '.mat']) ;
  fprintf('\tLoading kernel matrix ''%s''.\n', kerPath) ;
  kernels_ = load(kerPath) ;

  % only SVs
  kernels_.matrix = kernels_.matrix(svs, :) ;    

  % accumulate to base
  if isempty(base)
    base = zeros(size(kernels_.matrix)) ;
  end
  base = base + weights(ki) * kernels_.matrix ;  
end

info = whos('base') ;
fprintf('\tKernel matrix size %.2f GB\n', info.bytes / 1024^3) ;
clear info ;
