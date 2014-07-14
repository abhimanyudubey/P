function [conf, jobId] = bk_trainAppModel(conf, varargin)
% BK_TRAINAPPMODEL
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
conf_.trainRoiDbPath = '' ;
conf_.trainHistDir   = '' ;
conf_.testRoiDbPath  = '' ;
conf_.testHistDir    = '' ;
conf_.modelPath      = '' ;
conf_.featWeights    = [] ;
conf_.learnWeights   = 'none' ;
conf_.posWeight      = 1 ;
conf_.noClobber      = false ;

if nargin == 0, conf = conf_ ; return ; end
conf = override(conf_, conf, 1) ;

[jobId, taskId] = parallelDriver('numNodes', 1, ...
                                 'freeMemMb', 4 * 1024, ...
                                 varargin{:}) ;
if ~isnan(jobId) & isnan(taskId) ; return ; end

% --------------------------------------------------------------------
%                                      Process all classes and aspects
% --------------------------------------------------------------------

if conf.noClobber & checkFile(conf.modelPath)
  fprintf('\tSkipping to avoid clobbering ''%s''.\n', conf.modelPath) ;
  return ;
end

rand('state', 0) ;
randn('state', 0) ;

ensuredir(fileparts(conf.modelPath)) ;

% ------------------------------------------------------------------
%                                               Load training ROI DB
% ------------------------------------------------------------------

fprintf('\tLoading training ROI DB ''%s''.\n', ...
        conf.trainRoiDbPath) ;
trdb = load(conf.trainRoiDbPath) ;
labels = trdb.labels(:)' ;

fprintf('\tWeight learning stragegy: %s\n', conf.learnWeights) ;
fprintf('\tInitial weights: %s\n', sprintf('%g ', conf.featWeights)) ;

% ------------------------------------------------------------------
%                                                      Train weights
% ------------------------------------------------------------------

if ~isempty(conf.featWeights)
  initWeights = conf.featWeights ;
else
  initWeights = ones(1, length(conf.kerDb)) ;
end

% if there is only one kernel, MKL is not needed
if length(conf.kerDb) == 1,
  conf.learnWeights = 'none' ;
end

switch conf.learnWeights
  case 'none'
    weights = initWeights ;

  case 'equalMean' % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    [base, kernels, labels] = ...
        loadBase(conf.kerDir, conf.kerDb, trdb.labels(:)', ...
                 'weights', initWeights, ...
                 'maxSize', 5000, ...
                 'squeeze', false) ;
    for i=1:size(base,3)
      tmp = base(:,:,i) ;
      weights(i) = initWeights(i) ./ mean(tmp(:)) ;
      clear tmp ;
    end
    svm.d = weights ;
    fprintf('\tComputed weights: %s\n', sprintf('%g ', weights)) ;

    % KERNELS contains the full list of kernels, with gamma for the exp


  case 'manik' % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    [base, kernels, labels] = ...
        loadBase(conf.kerDir, conf.kerDb, trdb.labels(:)', ...
                 'weights', initWeights, ...
                 'maxSize', 5000, ...
                 'squeeze', false) ;
    svm = learnGmklSvm(base, labels(:)) ;
    svm = svmflip(svm, labels) ;
    weights = svm.d' .* initWeights ;
    svm.d = weights ;
    fprintf('\tComputed weights: %s\n', sprintf('%g ', weights)) ;

    % KERNELS contains the full list of kernels, with gamma for the exp

  case 'manikPartition' % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    numPartitions = 0 ;
    kerTypes = uniqueStrings({conf.kerDb.type}) ;
    featNames = uniqueStrings({conf.kerDb.feat}) ;
    for kerType = kerTypes
      kerType = char(kerType) ;

      for featName = featNames
        featName = char(featName) ;
        fprintf('Partition learning %d: %s %s\n', ...
                numPartitions, kerType, featName) ;

        selKer = findKernels(conf.kerDb, ...
                             'type', kerType,  ...
                             'feat', featName) ;

        if isempty(selKer), continue ; end
        numPartitions = numPartitions + 1 ;

        [base, kernels{numPartitions}, labels] = ...
            loadBase(conf.kerDir, conf.kerDb(selKer), trdb.labels(:)', ...
                     'weights', initWeights(selKer), ...
                     'maxSize', 10000, ...
                     'squeeze', false) ;
        svm = learnGmklSvm(base, labels(:)) ;
        svm = svmflip(svm, labels) ;
        weights(selKer) = svm.d' .* initWeights(selKer) ;
        fprintf('\tRelative weights: %s\n', sprintf('%g ', svm.d)) ;
        clear base ;
      end
    end

    fprintf('Partition learning: intra weights\n') ;
    base = [] ;
    thisPartition = 0 ;
    for kerType = kerTypes
      kerType = char(kerType) ;

      for featName = featNames
        featName = char(featName) ;

        selKer = findKernels(conf.kerDb, ...
                             'type', kerType,  ...
                             'feat', featName) ;

        if isempty(selKer), continue ; end
        thisPartition = thisPartition + 1 ;

        [partition, drop, labels] = ...
            loadBase(conf.kerDir, conf.kerDb(selKer), ...
                     trdb.labels(:)', ...
                     'weights', weights(selKer), ...
                     'maxSize', 7000, ...
                     'squeeze', true) ;

        if isempty(base)
          base = zeros([size(partition) numPartitions]) ;
        end
        base(:,:,thisPartition) = partition ;
        clear partition ;
      end
    end

    svm = learnGmklSvm(base, labels(:)) ;
    svm = svmflip(svm, labels) ;
    fprintf('\tRelative weights: %s\n', sprintf('%g ', svm.d)) ;
    clear base ;

    thisPartition = 0 ;
    for kerType = kerTypes
      kerType = char(kerType) ;

      for featName = featNames
        featName = char(featName) ;

        selKer = findKernels(conf.kerDb, ...
                             'type', kerType,  ...
                             'feat', featName) ;

        if isempty(selKer), continue ; end
        thisPartition = thisPartition + 1 ;

        weights(selKer) = svm.d(thisPartition) .* weights(selKer) ;
      end
    end

    % kernels contains the full list of kernels, with gamma for the exp
    kernels = cat(2, kernels{:}) ;
    svm.d = weights ;

  otherwise
    error('Unknown weight learning method') ;
end

% ------------------------------------------------------------------
%                                                          Train SVM
% ------------------------------------------------------------------

if ~exist('svm', 'var') | length(labels) < length(trdb.labels) | ...
    conf.posWeight
  fprintf('\tFinal SVM learning (pos weight: %f)\n', conf.posWeight) ;
  [base,kernels,labels] = ...
      loadBase(conf.kerDir, conf.kerDb, trdb.labels(:)', ...
               'weights', weights, ...
               'squeeze', true) ;

  svm = svmkernellearn(base, labels(:)',   ...
                       'type', 'C',        ...
                       'C', 10,            ...
                       'verbosity', 1,     ...
                       'weights', [+1 conf.posWeight ; -1 1]') ;
  svm = svmflip(svm, trdb.labels) ;

  % test it on train
  scores = svm.alphay' * base(svm.svind, :) + svm.b ;
  errs = scores .* trdb.labels(:)' < 0 ;
  err  = mean(errs) ;
  selPos = find(trdb.labels(:)' > 0) ;
  selNeg = find(trdb.labels(:)' < 0) ;
  werr = sum(errs(selPos)) * conf.posWeight + sum(errs(selNeg)) ;
  werr = werr / (length(selPos) * conf.posWeight + length(selNeg)) ;
  fprintf('\tSVM training error: %.2f%% (weighed: %.2f%%).\n', ...
          err*100, werr*100) ;

  svm.d = weights ;
end

svm.kernels = kernels ;
clear base ;



% ------------------------------------------------------------------
%                                               Compress linear SVMs
% ------------------------------------------------------------------


isLinear = true ;

for ki=1:length(svm.kernels)
  kernel = svm.kernels(ki) ;
  if ~strcmp(kernel.type, 'kl2'), isLinear = false ; continue ; end
  histPath = fullfile(conf.trainHistDir, ['hist-' kernel.histName '.mat']) ;
  fprintf('\tLoading histograms''%s''.\n', histPath) ;
  tmp = load(histPath) ;
  svm.(kernel.histName) = ...
      svm.d(ki) * tmp.hists(:, svm.svind) * svm.alphay ;
  clear tmp ;
end

if isLinear
  svm.type = 'linear' ;
else
  svm.type = 'nonlinear' ;
end

fprintf('\tNumber of support vectors: %d\n', length(svm.svind)) ;
fprintf('\tSVM type: %s\n', svm.type) ;

fprintf('\tSaving model ''%s''.\n', conf.modelPath) ;
ssave(conf.modelPath, '-STRUCT', 'svm') ;

return

% ------------------------------------------------------------------
%                                                            Test it
% ------------------------------------------------------------------

if isempty(conf.testRoiDbPath), return ; end

fprintf('\tLoading test ROI DB ''%s''.\n', conf.testRoiDbPath) ;
tsdb = load(conf.testRoiDbPath) ;

scores = zeros(size(tsdb.labels)) + svm.b ;
for ki=1:K
  matPath = fullfile(conf.kerDir, ['test-' kernels(ki).name '.mat']) ;
  fprintf('\tLoading kernel matrix ''%s''.\n', matPath) ;
  kernels_ = load(matPath) ;

  scores = scores + svm.d(ki) * ...
           (svm.alphay' * kernels_.matrix(svm.svind, :)) ;
  clear kernels_ ;
end

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

[a,b,c] = fileparts(conf.modelPath) ;

figPath = fullfile(a, [b '-pr-test.eps']) ;
printsize(.97 + 0.001*randn) ; % ugly hack for matlab repeat
print('-depsc', figPath) ;

prPath = fullfile(a, [b '-pr-test.mat']) ;
ssave(prPath, 'recall', 'precision', 'info') ;


% ------------------------------------------------------------------
function [base, kernels, labels] =  ...
    loadBase(kerDir, kernels, labels, varargin)
% ------------------------------------------------------------------

opts.squeeze = false ;
opts.weights = NaN ;
opts.maxSize = +inf ;
opts = vl_argparse(opts,varargin{:}) ;

base = [] ;
sel  = NaN ;
K = length(kernels) ;

% calculate subset of kernel to load ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if length(labels) > opts.maxSize
  selp = find(labels > 0) ;
  seln = find(labels < 0) ;
  np = length(selp) ;
  nm = length(seln) ;
  alpha = opts.maxSize / length(labels) ;
  mp = floor(alpha * np) ;
  mm = floor(alpha * nm) ;
  if (mp + mm < opts.maxSize), mp = mp + 1 ; end
  if (mp + mm < opts.maxSize), mm = mm + 1 ; end

  randn('state', 0) ;
  rand('state', 0) ;
  selp = randcol(selp, mp) ;
  seln = randcol(seln, mm) ;
  sel  = [selp seln] ;

  labels = labels(sel) ;

  fprintf('\tLoading kernels for only %.2f %% of the data (pos: %d neg: %d).\n', ...
          alpha * 100, length(selp), length(seln)) ;
end

% load matrices ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
clear kernels_ kernels__ ;
for ki = 1:K
  kerPath = fullfile(kerDir, [kernels(ki).name '.mat']) ;
  fprintf('\tLoading kernel matrix ''%s''.\n', kerPath) ;
  kernels_ = load(kerPath) ;
  if ~isfield(kernels_,'mu')
    [kernels_.mu] = deal([]) ;
  end
  if ~isnan(sel)
    fprintf('\t\tCutting down data.\n') ;
    kernels_.matrix = kernels_.matrix(sel, sel) ;
  end
  if ~isnan(opts.weights)
    fprintf('\t\tWeighing by %g.\n', opts.weights(ki)) ;
    kernels_.matrix = kernels_.matrix * opts.weights(ki) ;
  end
  if opts.squeeze
    if isempty(base)
      base = zeros(size(kernels_.matrix)) ;
    end
    base = base + kernels_.matrix ;
  else
    if isempty(base)
      base = zeros([size(kernels_.matrix) K]) ;
    end
    base(:,:,ki) = kernels_.matrix ;
  end
  kernels__(ki) = normalizeKernelStructure(kernels_);
  clear kernels_ ;
end

kernels = kernels__ ;
clear kernels__ ;

info = whos('base') ;
fprintf('\tKernel matrices size %.2f GB\n', info.bytes / 1024^3) ;
clear info ;

% ------------------------------------------------------------------
function kernel_ = normalizeKernelStructure(kernel)
% ------------------------------------------------------------------
kernel_.type     = kernel.type ;
kernel_.feat     = kernel.feat ;
kernel_.pyrLevel = kernel.pyrLevel ;
kernel_.histName = kernel.histName ;
kernel_.name     = kernel.name ;
