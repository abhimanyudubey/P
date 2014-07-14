function [conf, waitJobs] = meta_calcDistMatrix(conf, varargin)
% META_PROCESSDISTMATRIX
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

conf_.expPrefix           = '' ;

conf_.imageDbPath         = '' ;
conf_.imageSelTrain       = NaN ;
conf_.imageSelTest        = NaN ;

conf_.featPrefix          = '' ;
conf_.featDir             = '' ;
conf_.distFn              = [] ;

conf_.distDir             = '' ;
conf_.kerDir              = '' ;
conf_.noClobber           = false ;

if nargin == 0, conf = conf_ ; return ; end
conf = override(conf_, conf, 'warn') ;

imdb = getImageDb(conf.imageDbPath) ;

if isempty(conf.distFn)
  error('conf.distFn not specified') ;
end

% --------------------------------------------------------------------
%                                                            Configure
% --------------------------------------------------------------------

jobopts.waitJobs    = [] ;
jobopts.parallelize = 0 ;
jobopts.jobDir      = '' ;
jobopts = override(jobopts, struct(varargin{:}), 'warn') ;
waitJobs      = jobopts.waitJobs ;

ensuredir(conf.distDir) ;

conf.do_calcDist = 1 ;
conf.do_packDist = 1 ;
conf.do_calcTestDist = 1 ;
conf.do_packTestDist = 1 ;

% --------------------------------------------------------------------
%                                              Compute train distances
% --------------------------------------------------------------------

trainGbPaths      = {} ;
trainBlockDefs    = {} ;
trainBlockPaths   = {} ;
trainMatrixPath   = fullfile(conf.distDir, [conf.featPrefix '-dist-matrix']) ;
trainKerPath      = fullfile(conf.kerDir,  ['el2_' conf.featPrefix]) ;
trainKerDescrPath = fullfile(conf.kerDir,  ['descr-el2_', conf.featPrefix]) ;

% Find the name of the training images to compute their distance
for i=1:length(conf.imageSelTrain)
  imageName = imdb.images.name{conf.imageSelTrain(i)} ;
  trainGbPaths{i} =  fullfile(conf.featDir, imageName) ;
end

% Partition the distance matrix in a number of blocks to be computed
% in parallel

numBlocks = 10 ;
rows = fix(linspace(1, length(conf.imageSelTrain), numBlocks + 1)) ;
cols = fix(linspace(1, length(conf.imageSelTrain), numBlocks + 1)) ;
for i=1:numBlocks
  for j=1:numBlocks
    block.rowStart = rows(i) ;
    block.rowEnd   = rows(i + 1) - 1 ;
    block.colStart = cols(j) ;
    block.colEnd   = cols(j + 1) - 1 ;
    if i==numBlocks, block.rowEnd = rows(end) ; end
    if j==numBlocks, block.colEnd = cols(end) ; end

    trainBlockDefs{end+1} = block ;
    trainBlockPaths{end+1} = ...
        fullfile(conf.distDir, ...
                 sprintf('%s-dist-block-%d', ...
                         conf.featPrefix, ...
                         length(trainBlockDefs))) ;
  end
end
trainBlockDefs = cat(2, trainBlockDefs{:}) ;

if conf.do_calcDist
  bkcf = bk_calcDistMatrixBlock ;
  bkcf.gbPaths         = trainGbPaths ;
  bkcf.blockDefs       = trainBlockDefs ;
  bkcf.blockPaths      = trainBlockPaths ;
  bkcf.distFn          = conf.distFn ;
  bkcf.noClobber       = conf.noClobber ;

  [opts, waitJobs] = ...
      bk_calcDistMatrixBlock(bkcf, ...
                             'parallelize', jobopts.parallelize, ...
                             'jobDir',      jobopts.jobDir, ...
                             'waitJobs',    waitJobs ) ;

end

if conf.do_packDist
  bkcf = bk_packDistMatrix ;
  bkcf.blockPaths      = trainBlockPaths ;
  bkcf.distMatrixPath  = trainMatrixPath ;
  bkcf.kerPath         = trainKerPath ;
  bkcf.kerDescrPath    = trainKerDescrPath ;
  bkcf.noClobber       = conf.noClobber ;

  [opts, waitJobs] = ...
      bk_packDistMatrix(bkcf, ...
                        'parallelize', jobopts.parallelize, ...
                        'jobDir',      jobopts.jobDir, ...
                        'waitJobs',    waitJobs) ;
end


% --------------------------------------------------------------------
%                                               Compute test distances
% --------------------------------------------------------------------


testGbPaths    = {} ;
testBlockDefs  = {} ;
testBlockPaths = {} ;
testMatrixPath = fullfile(conf.distDir, [conf.featPrefix '-dist-test-matrix']) ;
testKerPath    = fullfile(conf.kerDir,  ['test-el2_' conf.featPrefix]) ;

for i=1:length(conf.imageSelTest)
  imageName = imdb.images.name{conf.imageSelTest(i)} ;
  testGbPaths{i} = fullfile(conf.featDir, imageName) ;
end

numBlocks = 10 ;
rows = fix(linspace(1, length(conf.imageSelTrain), numBlocks + 1)) ;
cols = fix(linspace(1, length(conf.imageSelTest),  numBlocks + 1)) ;
for i=1:numBlocks
  for j=1:numBlocks
    block.rowStart = rows(i) ;
    block.rowEnd   = rows(i + 1) - 1 ;
    block.colStart = cols(j) ;
    block.colEnd   = cols(j + 1) - 1 ;
    if i==numBlocks, block.rowEnd = rows(end) ; end
    if j==numBlocks, block.colEnd = cols(end) ; end

    testBlockDefs{end+1} = block ;
    testBlockPaths{end+1} = ...
        fullfile(conf.distDir, ...
                 sprintf('%s-dist-test-block-%d', ...
                         conf.featPrefix, ...
                         length(testBlockDefs))) ;
  end
end
testBlockDefs = cat(2, testBlockDefs{:}) ;

if conf.do_calcTestDist
  bkcf = bk_calcDistMatrixBlock ;
  bkcf.gbPaths         = trainGbPaths ;
  bkcf.gbTestPaths     = testGbPaths ;
  bkcf.blockDefs       = testBlockDefs ;
  bkcf.blockPaths      = testBlockPaths ;
  bkcf.distFn          = conf.distFn ;
  bkcf.noClobber       = conf.noClobber ;

  [opts, waitJobs] = ...
      bk_calcDistMatrixBlock(bkcf, ...
                         'parallelize', jobopts.parallelize, ...
                         'jobDir',      jobopts.jobDir, ...
                         'waitJobs',    waitJobs) ;
end

if conf.do_packTestDist
  bkcf = bk_packDistMatrix ;
  bkcf.blockPaths      = testBlockPaths ;
  bkcf.distMatrixPath  = testMatrixPath ;
  bkcf.kerPath         = testKerPath ;
  bkcf.kerDescrPath    = trainKerDescrPath ;
  bkcf.testMode        = true ;
  bkcf.noClobber       = conf.noClobber ;

  [opts, waitJobs] = ...
      bk_packDistMatrix(bkcf, ...
                         'parallelize', jobopts.parallelize, ...
                         'jobDir',      jobopts.jobDir, ...
                         'waitJobs',    waitJobs) ;
end
