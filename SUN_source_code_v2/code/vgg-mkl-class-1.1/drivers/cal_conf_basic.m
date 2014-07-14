% CAL_CONF Caltech-101 configuration
%
%  Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

conf.noClobber        = 1 ;
conf.parallelize      = 0 ;
conf.masterRandomSeed = 1 ;
conf.masterNumTrain   = 15 ;
conf.masterExpNum     = 1 ;
conf.smallData        = true ;

if exist('magicSmallData', 'var')
  conf.smallData = magicSmallData ;
end
if exist('magicMasterRandomSeed', 'var')
  conf.masterRandomSeed = magicMasterRandomSeed ;
end
if exist('magicMasterExpNum', 'var')
  conf.masterExpNum = magicMasterExpNum ;
end
if exist('magicMasterNumTrain', 'var')
  conf.masterNumTrain = magicMasterNumTrain ;
end

% Directories ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

dataDir = fullfile(pwd,'data', ...
                   sprintf('cal-%d-%d', conf.masterNumTrain, conf.masterRandomSeed)) ;
if conf.smallData
  dataDir = [dataDir '-small'] ;
end
calDir  = fullfile(pwd,'data','caltech-101-prep') ;
tmpDir  = tempdir ;

conf.calDir      = calDir ;
conf.ppDir       = fullfile(dataDir, 'pp')  ;
conf.jobDir      = fullfile(dataDir, 'jobs') ;
conf.aggrDir     = fullfile(dataDir, 'aggr') ;
conf.trainDir    = fullfile(dataDir, 'train') ;
conf.imageDbPath = fullfile(dataDir, 'imdb.mat') ;
conf.gtRoiDbPath = fullfile(dataDir, 'roidb.mat') ;
conf.gtRoiTrainSetName = 'train' ;
conf.tmpDir      = tmpDir ;

ensuredir(dataDir) ;

clear dataDir vocDir tmpDir ;

% --------------------------------------------------------------------
%                                               Training configuration
% --------------------------------------------------------------------

% Train and test set labels
conf.trainSetName  = 'train' ;
conf.testSetName   = 'test' ;

% Which jitters to use when training
conf.jitterNames = {'zm1', 'zm2'} ;

% Enlarge ROIs by this much when computing histograms
conf.roiMagnif = 0 ;

conf.maxDenseDim = 300 ;

% Training appearance model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% Method used to learn the wieghts: none, manik, manikPartition
conf.learnWeightMethod = 'manik' ;
%conf.learnWeightMethod = 'equalMean' ;

% Weights of positive examples
conf.posWeight = 1.0 ;
%conf.posWeight = 10.0 ;

conf.kerType   = 'echi2' ;

% How many and which GT ROIs to use as training data.
conf.numPos    = 15 ;
conf.numJitPos = 0 ;

% --------------------------------------------------------------------
%                                                Feature configuration
% --------------------------------------------------------------------

conf.featNames = {'gb', 'phowColor', 'phowGray', 'ssim'} ;

% PHOW features ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

conf.feat.phowGray.format            = 'dense' ;
conf.feat.phowGray.extractFn         = @phow ;
conf.feat.phowGray.clusterFn         = @ikmeansCluster ;
conf.feat.phowGray.quantizeFn        = @ikmeansQuantize ;
conf.feat.phowGray.vocabSize         = 600 ;
conf.feat.phowGray.numImagesPerClass = 30 ;
conf.feat.phowGray.numFeatsPerImage  = 100 ;
conf.feat.phowGray.compress          = false ;
conf.feat.phowGray.step              = 2 ;
conf.feat.phowGray.sizes             = [5 7 9 12] ;
conf.feat.phowGray.color             = false ;
conf.feat.phowGray.pyrLevels         = 0:2 ;
conf.feat.phowGray.fast              = true ;

conf.feat.phowColor                  = conf.feat.phowGray ;
conf.feat.phowColor.color            = true ;

% GB features ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

conf.feat.gb.format             = 'sparse' ;
conf.feat.gb.extractFn          = @gb ;
conf.feat.gb.clusterFn          = @vggCluster ;
conf.feat.gb.quantizeFn         = @vggQuantize ;
conf.feat.gb.distFn             = @gbDistance ;
conf.feat.gb.vocabSize          = 0 ;
conf.feat.gb.compress           = false ;
conf.feat.gb.pyrLevels          = [] ;

conf.feat.gb.sampleRadii        = [0 4 8 16 32 50] ;
conf.feat.gb.numSamplePerRadius = [1 8 8 10 12 12] ;
conf.feat.gb.blurRate           = 0.5 ;
conf.feat.gb.blurBase           = 1 ;
conf.feat.gb.numFeats           = 300 ;
conf.feat.gb.repulsionRadius    = 5 ;
conf.feat.gb.usePBEdges         = false ;

conf.feat.gbpb                  = conf.feat.gb ;
conf.feat.gbpb.usePBEdges       = true ;

% SSIM features ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

conf.feat.ssim.format            = 'dense' ;
conf.feat.ssim.extractFn         = @vggSsim ;
conf.feat.ssim.clusterFn         = @vggCluster ;
conf.feat.ssim.quantizeFn        = @vggQuantize ;
conf.feat.ssim.vocabSize         = 300 ;
conf.feat.ssim.numImagesPerClass = 10 ;
conf.feat.ssim.numFeatsPerImage  = 300 ;
conf.feat.ssim.compress          = false ;
conf.feat.ssim.pyrLevels         = 0:2 ;

conf.feat.ssim.coRelWindowRadius = 40 ;
conf.feat.ssim.subsample_x       = 4 ;
conf.feat.ssim.subsample_y       = 4 ;
conf.feat.ssim.numRadiiIntervals = 3 ;
conf.feat.ssim.numThetaIntervals = 10 ;
conf.feat.ssim.saliencyThresh    = 1 ;
conf.feat.ssim.size              = 5 ;
conf.feat.ssim.varNoise          = 150 ;
conf.feat.ssim.nChannels         = 3 ;
conf.feat.ssim.useMask           = 0 ;
conf.feat.ssim.autoVarRadius     = 1 ;

% rename feat -> featOpts
conf.featOpts = conf.feat ;
conf = rmfield(conf, 'feat') ;

conf.numPos              = conf.masterNumTrain ;
conf.numJitPos           = 15 ;
conf.learnWeightMethod   = 'manik' ;

switch conf.masterExpNum
  case 1
    conf.expPrefix  = 'baseline' ;

  case 2
    conf.expPrefix  = 'baseline-gb' ;
    conf.featNames  = {'gb'} ;

  case 3
    conf.expPrefix  = 'baseline-phowGray' ;
    conf.featNames  = {'phowGray'} ;

  case 4
    conf.expPrefix  = 'baseline-phowColor' ;
    conf.featNames  = {'phowColor'} ;

  case 5
    conf.expPrefix  = 'baseline-ssim' ;
    conf.featNames  = {'ssim'} ;

  case 6
    conf.expPrefix  = 'baseline-avg' ;
    conf.learnWeightMethod = 'equalMean' ;
end

if conf.smallData
  conf.numJitPos = 0 ;
end

% Setup the kernels
conf.kerDb = [] ;
for fi = 1:length(conf.featNames)
  featName = conf.featNames{fi} ;

  if conf.featOpts.(featName).vocabSize == 0
    conf.kerDb = [conf.kerDb ...
                  calcKernelDb({'el2'}, { featName }, conf.featOpts) ;] ;
  else
    conf.kerDb = [conf.kerDb ...
                  calcKernelDb({'echi2'}, { featName }, conf.featOpts) ;] ;
  end
end
