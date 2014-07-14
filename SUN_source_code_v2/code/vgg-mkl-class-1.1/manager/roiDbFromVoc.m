function roiDb = roiDbFromVoc(root, imDb, imSel, varargin)
% ROIDBFROMVOC  Construct ROI DB from VOC data
%   ROIDB = ROIDBFROMVOC initializes an empty DB with the VOC classes.
%
%   ROIDB = ROIDBFROMVOC(ROOT, IMDB, IMSEL) constructs a ROI database
%   from the VOC database. ROOT is the path to the VOC toolkit, IMBD
%   an image DB constructed by IMAGEDBFROMVOC(), IMSEL a selection of
%   such images. The database includes all the ROIs found for those
%   images.
%
%   Options:
%
%   jitter::
%     Apply the specified jitter all the ROIs according to the
%     transformation JITTERNAME (see TFIMAGE(), TFBOX(),
%     TFASPECT()).
%
%   idOffset
%     Add this offset to the ROI IDs.
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

opts.jitter = '' ;
opts.idOffset = 0 ;
opts = vl_argparse(opts, varargin{:}) ;

roiDb.TRAIN     = 1 ;
roiDb.VAL       = 2 ;
roiDb.TEST      = 3 ;
roiDb.JIT_TRAIN = 4 ;
roiDb.JIT_VAL   = 5 ;

roiDb.aspects.FRONT = uint8(1) ;
roiDb.aspects.REAR  = uint8(2) ;
roiDb.aspects.LEFT  = uint8(3) ;
roiDb.aspects.RIGHT = uint8(4) ;
roiDb.aspects.MISC  = uint8(5) ;

roiDb.jitters.FLIPLR     = uint8(1) ;
roiDb.jitters.RP5        = uint8(2) ;
roiDb.jitters.RM5        = uint8(3) ;
roiDb.jitters.FLIPLR_RP5 = uint8(4) ;
roiDb.jitters.FLIPLR_RM5 = uint8(5) ;

roiDb.classes.AEROPLANE   = uint8(1) ;
roiDb.classes.BICYCLE     = uint8(2) ;
roiDb.classes.BIRD        = uint8(3) ;
roiDb.classes.BOAT        = uint8(4) ;
roiDb.classes.BOTTLE      = uint8(5) ;
roiDb.classes.BUS         = uint8(6) ;
roiDb.classes.CAR         = uint8(7) ;
roiDb.classes.CAT         = uint8(8) ;
roiDb.classes.CHAIR       = uint8(9) ;
roiDb.classes.COW         = uint8(10) ;
roiDb.classes.DININGTABLE = uint8(11) ;
roiDb.classes.DOG         = uint8(12) ;
roiDb.classes.HORSE       = uint8(13) ;
roiDb.classes.MOTORBIKE   = uint8(14) ;
roiDb.classes.PERSON      = uint8(15) ;
roiDb.classes.POTTEDPLANT = uint8(16) ;
roiDb.classes.SHEEP       = uint8(17) ;
roiDb.classes.SOFA        = uint8(18) ;
roiDb.classes.TRAIN       = uint8(19) ;
roiDb.classes.TVMONITOR   = uint8(20) ;

roiDb.sets.TRAIN07   = uint8(1) ;
roiDb.sets.VAL07     = uint8(2) ;
roiDb.sets.TEST07    = uint8(3) ;
roiDb.sets.TRAIN08   = uint8(4) ;
roiDb.sets.VAL08     = uint8(5) ;
roiDb.sets.TEST08    = uint8(6) ;
roiDb.sets.TRAIN09   = uint8(7) ;
roiDb.sets.VAL09     = uint8(8) ;
roiDb.sets.TEST09    = uint8(9) ;

if nargin == 0, return ; end

VOCopts09 = getVocOpts(root, '2009') ;
VOCopts08 = getVocOpts(root, '2008') ;
VOCopts07 = getVocOpts(root, '2007') ;

rois = {} ;

if ~exist('PASreadrecotd')
  addpath(fullfile(VOCopts09.datadir, 'VOCcode')) ;
end

n = 0 ;
for ii = imSel
  imageName = imDb.images.name{ii} ;
  imageSet  = imDb.images.set(ii) ;
  imageId   = imDb.images.id(ii) ;
  imageSize = imDb.images.size(:,ii) ;
  
  [drop, imageNameNoSfx] = fileparts(imageName) ;

  fprintf('Processing image ''%s''.\r', imageName) ;

  % try to retrieve the annotations
  annoPath09 = sprintf(VOCopts09.annopath, imageNameNoSfx) ;
  annoPath08 = sprintf(VOCopts08.annopath, imageNameNoSfx) ;
  annoPath07 = sprintf(VOCopts07.annopath, imageNameNoSfx) ;

  annoPath = [] ;
  if exist(annoPath09, 'file')
    annoPath = annoPath09 ;
  elseif exist(annoPath08, 'file')
    annoPath = annoPath08 ;
  elseif exist(annoPath07)
    annoPath = annoPath07 ;
  else
    warning('Could not find annotations for iamge ''%s''', imageNameNoSfx) ;
  end

  if ~ isempty(annoPath)
    anno = PASreadrecord(annoPath) ;

    % now create ROI
    for o = anno.objects
      n = n + 1 ;

      rois{n}.id        = n + opts.idOffset ;
      rois{n}.imageId   = imageId ;
      rois{n}.set       = imageSet ;
      rois{n}.class     = roiDb.classes.(upper(o.class)) ;

      switch o.view
        case 'Frontal'
          rois{n}.aspect = roiDb.aspects.FRONT ;
        case 'Rear'
          rois{n}.aspect = roiDb.aspects.REAR ;
        case {'SideFaceLeft', 'Left'}
          rois{n}.aspect = roiDb.aspects.LEFT ;
        case {'SideFaceRight', 'Right'}
          rois{n}.aspect = roiDb.aspects.RIGHT ;
        case ''
          rois{n}.aspect = roiDb.aspects.MISC ;
        otherwise
          error(sprintf('Unknown view ''%s''', o.view)) ;
      end

      box = [max(round(o.bbox(1:2)), 1)'; min(round(o.bbox(3:4)), imageSize')'] ;
      jitter = opts.jitter ;
      jitterCode = uint8(0) ;

      if ~strcmp(jitter, '')
        box = tfBox(imageSize(1), imageSize(2), box) ;
        jitterCode = roiDb.jitters(upper(jitter)) ;
      end

      rois{n}.difficult = logical(o.difficult) ;
      rois{n}.truncated = logical(o.truncated) ;
      rois{n}.occluded  = logical(o.occluded) ;
      rois{n}.box       = box ;
      rois{n}.jitter    = jitterCode ;
    end
  end
end

roiDb.rois = cat(2, rois{:}) ;

fprintf('\tDone processing.\n') ;