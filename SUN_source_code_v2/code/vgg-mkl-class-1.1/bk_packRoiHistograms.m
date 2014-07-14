function [conf, jobId] = bk_packRoiHistograms(conf, varargin)
% BK_PACKROIHISTOGRAMS
%
%  Note:: all images have pre-processed features.
%
%  noClobber:     true/false ?

conf_.roiDbPath    = '' ;
conf_.roiSel       = [] ;
conf_.histDir      = '' ;
conf_.histPackDir  = '' ;
conf_.histNames    = '' ;

conf_.featNames    = {} ;
conf_.featOpts     = struct ;

conf_.noClobber    = false ;

if nargin < 1, conf = conf_ ; return ; end
conf = override(conf_, conf) ;

[jobId, taskId] = parallelDriver(varargin{:}, 'numNodes', 1) ;
if ~isnan(jobId) & isnan(taskId) ; return ; end

if isempty(conf.histNames)
  histNames = calcHistNames(conf.featNames, conf.featOpts) ;
else
  histNames = conf.histNames ;
end

% no-clobber logic
noClobber = conf.noClobber ;
for name = histNames
  name = char(name) ;
  histPath = fullfile(conf.histPackDir, sprintf('hist-%s', name)) ;
  noClobber = noClobber & checkFile(histPath) ;
end
if noClobber
  fprintf('Skipping to avoid clobbering.\n') ;
  return ;
end

ensuredir(conf.histPackDir) ;
fprintf('\tLoading ROI database ''%s''.\n', conf.roiDbPath) ;
roidb = load(conf.roiDbPath) ;

if isempty(conf.roiSel)
  fprintf('\tProcessing all %d ROIs in database\n', ...
          length([roidb.rois.id])) ;
  roiSel = 1:length([roidb.rois.id]) ;
else
  fprintf('\tProcessing %d out of %d ROIs in database\n', ...
          length(conf.roiSel), length([roidb.rois.id])) ;
  roiSel = conf.roiSel ;
end

% --------------------------------------------------------------------
%                                                      Pack histograms
% --------------------------------------------------------------------

histCollection = {} ;
for rsi = 1:length(roiSel)
  ri = roiSel(rsi) ;
  roiId = roidb.rois.id(ri) ;
  histPath = fullfile(conf.histDir, sprintf('%012.0f', roiId)) ;
  histCollection{end+1} = load(histPath) ;
end
histCollection = cat(2, histCollection{:}) ;

for name = histNames
  name = char(name) ;
  histPath = fullfile(conf.histPackDir, sprintf('hist-%s', name)) ;
  if conf.noClobber & checkFile(histPath),
    fprintf('Skipping to avoid clobbering ''%s''.\n', ...
            histPath) ;
    continue ;
  end
  hists = [histCollection.(name)] ;
  fprintf('\tSaving ''%s''.\n', histPath) ;
  ssave(histPath, 'hists') ;
end
