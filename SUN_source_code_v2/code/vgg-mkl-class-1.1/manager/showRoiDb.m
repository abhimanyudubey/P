function showRoiDb(conf, roidb)
% SHOWROIDB
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

roiDbPath = '?' ;

imdb  = getImageDb(conf.imageDbPath) ;
if isstr(roidb)
  roiDbPath = roidb ;
  roidb = getRoiDb(roiDbPath) ;
end
imageIndex = [imdb.images.id] ;
hasLabels  = isfield(roidb, 'labels') ;

if isfield(roidb.rois, 'score')
  roidb.roiScores = roidb.rois.score ;
end
hasScores  = isfield(roidb, 'roiScores') ;

if hasScores & 1
  fprintf('%s: sorting ROIS by score\n', mfilename) ;
  [roidb.roiScores, perm] = sort(roidb.roiScores, 'descend') ;
  roidb.rois = soaSubsRef(roidb.rois, perm) ;
  if hasLabels
    roidb.labels = roidb.labels(perm) ;
  end
end

if hasLabels
  numPosLabels = sum(roidb.labels > 0) ;
  numNegLabels = sum(roidb.labels < 0) ;
end

figure(101) ; clf ;
plot(roidb.rois.id) ;
xlabel('ROI index') ;
ylabel('ROI id') ;

fh = figure(100) ; clf ;
set(gcf,'units', 'normalized') ;
roiSel = uicontrol(fh, 'style', 'listbox', ...
                   'String', arrayfun(@(x) sprintf('%012.0f', x), roidb.rois.id, ...
                                     'uniformoutput', false), ...
                   'Value', 1, ...
                   'Units', 'normalized', 'Position', [0 0 .25 1], ...
                   'Callback', @display) ;
infoH = axes('position', [.25 0 .25 1]) ;
imgH  = axes('position', [.5 0 .5 1]) ;
display() ;

  function display(varargin)
      ri = get(roiSel, 'value') ;
      roi = soaSubsRef(roidb.rois, ri) ;

      ii = binsearch(imageIndex, roi.imageId) ;
      imagePath = fullfile(imdb.dir, imdb.images.name{ii}) ;
      imageSize = imdb.images.size(:,ii) ;
      imageName = imdb.images.name{ii} ;
      imageSet  = decodeEnum(imdb.sets, imdb.images.set(ii)) ;
      im = getImage(imagePath) ;

      className = 'BACKGROUND' ;
      if roi.class > 0
        className = decodeEnum(roidb.classes, roi.class) ;
      end
      aspectName = decodeEnum(roidb.aspects, roi.aspect) ;

      jitterName = '' ;
      if roi.jitter
        jitterName = decodeEnum(roidb.jitters, roi.jitter) ;
        im = tfImage(lower(jitterName), im2double(im)) ;
      end
      if isempty(jitterName), jitterName = 'none' ; end

      str = '' ;
      str = [str sprintf('number:\t%d', ri)] ;
      str = [str sprintf('\nid:\t%012.0f', roi.id)] ;
      str = [str sprintf('\nset:\t%s', decodeEnum(roidb.sets, roi.set))] ;
      str = [str sprintf('\nclass:\t%s', className)] ;
      str = [str sprintf('\naspect:\t%s', aspectName)] ;

      str = [str sprintf('\n\ngeometry:\t%.0f %.0f %.0f %.0f', roi.box)] ;
      str = [str sprintf('\njitter:\t%s', jitterName)] ;
      str = [str sprintf('\ntruncated:\t%d', roi.truncated)] ;
      str = [str sprintf('\noccluded:\t%d', roi.occluded)] ;
      str = [str sprintf('\ndifficult:\t%d', roi.difficult)] ;

      if hasLabels
        str = [str sprintf('\n\nclass label:\t%d', roidb.labels(ri))] ;
        str = [str sprintf('\npos-neg labels:\t%d-%d', numPosLabels, numNegLabels)] ;
      end

      if hasScores
        str = [str sprintf('\n\nscore\t%f', roidb.roiScores(ri))] ;
      end

      str = [str sprintf('\n\nimg geom:\t%dx%d\nimg name:\t%s\nimg set:\t%s',  ...
                         imageSize(1), imageSize(2), imageName, imageSet)] ;

      retrNumber   = floor(roi.id / 1e11) ;
      sourceNumber = mod(floor(roi.id / 1e8), 1e3) ;
      jitterNumber = mod(floor(roi.id / 1e6), 1e2) ;
      seqNumber    = mod(floor(roi.id / 1e0), 1e6) ;

      str = [str sprintf('\n\nretr num:\t%d\nsource num:\t%d\njitter num:\t%d\nseq num:\t%d', ...
                         retrNumber, sourceNumber, jitterNumber, seqNumber)] ;

      str = [str sprintf('\n\nroidb: %s', roiDbPath)];
      str = [str sprintf('\nroidb size:\t%d', length(roidb.rois.id))] ;

      cla(infoH) ; axes(infoH) ;
      text(0,0, str, ...
           'backgroundcolor', 'w', 'verticalalign', 'top', 'interpreter','none') ;
      axis off ;
      set(gca,'ydir', 'reverse') ;

      cla(imgH) ; axes(imgH) ;
      %image(im,'parent',imgH, 'CLim', [0 1], 'CDataMapping', 'scaled', 'CLimInclude', 'off') ; hold(imgH,'on') ;
      imagesc(im,'parent',imgH) ; hold(imgH,'on') ;
      axis equal ;
      plotroi(roi.box, 'k', 'linewidth', 3) ;
      plotroi(roi.box, 'g', 'linewidth', 2) ;
      axis off ;

      drawnow ;
  end

end
