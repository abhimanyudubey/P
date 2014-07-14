function hists = getAllRoiHists(conf, imageName, boxes, varargin)
% GETALLROIHISTS  Get all histograms for a ROI
%
%    Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

opts.className = '' ;
opts.magnif = 0.0 ;
opts.maxDenseDim = +inf ;
opts.histNames = {} ;
opts.project = [] ;
[opts, varargin] = vl_argparse(opts, varargin{:}) ;

featNames = conf.featNames ;
for fi = 1:length(featNames)
  featName = char(featNames{fi}) ;
  featOpts = conf.featOpts.(featName) ;

  if featOpts.vocabSize == 0, continue ; end

  wordPath = fullfile(conf.ppDir, ...
                      featName,  ...
                      'words', ...
                      imageName   ) ;

  comprMap = [] ;
  vocabWeights = [] ;
  project = {} ;
  
  % class-specific weighing and compression ~~~~~~~~~~~~~~~~~~~~~~~~~~
  if ~ isempty(opts.className) & featOpts.compress
    comprPath = fullfile(conf.ppDir, featName, 'compr', upper(opts.className));
    compr = load(comprPath) ;
    comprMap = compr.map ;
    vocabWeights = compr.idf ;
  end

  % non class-specific weighing ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if isfield(featOpts, 'hasIDFWeights') & featOpts.hasIDFWeights
    idfPath = fullfile(conf.ppDir, featName, 'idf') ;
    tmp = load(idfPath) ;
    vocabWeights = tmp.idf ;
  end

  % determine which pyramid levels are requested ~~~~~~~~~~~~~~~~~~~~~
  pyrLevels = [] ;
  for pi=1:length(featOpts.pyrLevels)
    L = featOpts.pyrLevels(pi) ;
    histName = sprintf('%s_L%d', featName, L) ;
    hists.(histName) = [] ;
    if isempty(opts.histNames) || any(strcmpi(histName, opts.histNames))
      pyrLevels(end+1) = L ;
      if ~ isempty(opts.project)
        project{end+1} = opts.project.(histName) ;
      end
    end
  end

  if length(pyrLevels) == 0, continue ; end

  % extract features
  temp = getRoiHists(wordPath, ...
                     conf.featOpts.(featName), pyrLevels, boxes, ...
                     'magnif', opts.magnif, ...
                     'comprMap', comprMap, ...
                     'vocabWeights', vocabWeights', ...
                     'project', project, ...
                     varargin{:}) ;

  % fix output format
  for li=1:length(pyrLevels)
    L = pyrLevels(li) ;
    histName = sprintf('%s_L%d', featName, L) ;
    if size(temp{li},1) <= opts.maxDenseDim
      hists.(histName) = full(temp{li}) ;
    else
      hists.(histName) = sparse(temp{li}) ;
    end
  end

end
