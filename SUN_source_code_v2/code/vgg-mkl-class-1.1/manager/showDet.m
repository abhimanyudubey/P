function showDet(detDir, className, aspectName, suppThresh)
% SHOWDET
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

if nargin < 3, aspectName = '.*' ; end
if nargin < 4, suppThresh = NaN  ; end
if ischar(suppThresh), suppThresh = sscanf(suppThresh, '%f'); end

conf = do_conf_fun() ;

fprintf('Class name: %s\n', className) ;
fprintf('Aspect name: %s\n', aspectName) ;
if ~isnan(suppThresh), 
  fprintf('Suppress threshold: %.2f\n', suppThresh) ; 
end ;

% --------------------------------------------------------------------
%                                                       Load databases
% --------------------------------------------------------------------

fprintf('Loading databases.\n') ;
imdb       = getImageDb(conf.imageDbPath) ; 
roidb      = getRoiDb(conf.gtRoiDbPath) ;
imageIndex = [imdb.images.id] ;
imageNames = char({imdb.images.name}) ;

% --------------------------------------------------------------------
%                                                      Load candidates
% --------------------------------------------------------------------

fprintf('Reading candidates for ''%s''.\n', detDir) ;
candList = cacheGet(mfilename, ...
                    {detDir, className, aspectName, suppThresh}, {}) ;
if isempty(candList)
  
  detDirList = dir(detDir) ;
  if isempty(detDirList)
    error(sprintf('The directory ''%s'' is empty', detDir)) ;
  end

  candList = {} ;  
  n = 1 ;
  for detDirEntry = detDirList'
    extract = regexp(detDirEntry.name, '^(?<imageName>[0-9]\w+).mat$', 'names') ;
    if isempty([extract.imageName]), continue ; end
    
    candList{n} = load(fullfile(detDir, detDirEntry.name)) ;
    candList{n}.imageName = extract.imageName ;
    
    if ~isnan(suppThresh)      
      sel = suppressBoxes(candList{n}.box, suppThresh, 5) ;
      %      sel = sel(1:5) ;
      candList{n}.box = candList{n}.box(:,sel) ;
      candList{n}.score = candList{n}.score(:,sel) ;
    end
    n = n + 1 ;
  end
  candList = cat(2, candList{:}) ;
  
  sourcedImageNames = char({candList.imageName}) ;
  [ok, ii] = ismember(sourcedImageNames, imageNames, 'rows') ;
  tmp = {imdb.images(ii).id} ;
  [candList.imageId] = deal(tmp{:}) ;
  
  % remove stuff which is not in test...
  sel = find([imdb.images(binsearch(imageIndex, [candList.imageId])).set] == imdb.sets.TEST07) ;
  candList = candList(sel) ;
  
  
  if ~all(ok), error('Some of the image could not be found in DB') ; end
  cacheStore(mfilename, candList, ...
             {detDir, className, aspectName, suppThresh}, {}, 1) ;
end

% --------------------------------------------------------------------
%                                               Match candidates to GT
% --------------------------------------------------------------------

fprintf('Comparing to GT.\n') ;
roiList = cacheGet([mfilename '_roi'], ...
                   {detDir, className, aspectName, suppThresh}, {}) ;
if isempty(roiList)
  roiList = {} ;
  for ci = 1:length(candList)
    match = matchToGt(ci) ;

    selMiss = find(isnan(match.gtBoxToDet)) ;
    missBoxes  = roidb.rois.box(:, sel(selMiss)) ;
    missScores = -inf * ones(1, size(missBoxes,2)) ; 
    missFlags  = ones(1, size(missBoxes,2)) ;
    
    roiList{ci}.candSel = ci * ones(1, length(candList(ci).score) + size(missBoxes,2)) ;
    roiList{ci}.scores  = [candList(ci).score missScores] ;
    roiList{ci}.boxes   = [candList(ci).box   missBoxes] ;
    roiList{ci}.flags   = [match.detBoxFlags missFlags] ;
  end
  roiList = aosToSoa(cat(2, roiList{:})) ;
  cacheStore([mfilename '_roi'], roiList, ...
             {detDir, className, aspectName, suppThresh}, {}, 1) ;
end
% --------------------------------------------------------------------
%                                                              Display
% --------------------------------------------------------------------

fprintf('Instantiating GUI.\n') ;
% sort candidates by best match score
bestScores  = arrayfun(@(x) max([x.score -inf]), candList) ;
worstScores = arrayfun(@(x) min([x.score +inf]), candList) ;
bestScore   = max(bestScores) ;
worstScore  = min(worstScores) ;

[dorp, candOrder] = sort(bestScores,'descend') ;
[drop, roiOrder]  = sort(roiList.scores, 'descend') ;


recall    = cumsum(roiList.flags(roiOrder) == +1) /     sum(roiList.flags(roiOrder) == +1) ;
precision = cumsum(roiList.flags(roiOrder) == +1) ./ ((cumsum(roiList.flags(roiOrder) ~= 0) + eps)) ;

stop = max(find(roiList.scores(roiOrder) > -inf)) ;

%fh = figure(101) ; clf ;
fh = figure ; clf ;
%subplot(1,2,1) ; plot(roiList.scores(roiOrder)) ; title('ROI scores') ;
%subplot(1,2,2) ; 
plot(recall, precision) ; 
grid on ; axis equal ;
xlim([0,1]) ;
ylim([0,1]) ;
[a,a08] = auc([0 recall(1:stop)], [1 precision(1:stop)]) ;
title(sprintf('precision-recall %.2f %% (%.2f %%)', a*100, a08*100)) ;
drawnow ;

candLabels = arrayfun(@(x) sprintf('%012.0f', candList(x).imageId), ...
                      candOrder, ...
                      'UniformOutput', false) ;

roiLabels  = arrayfun(@(x) sprintf('%2.3f %3d [%6.2f %6.2f]', ...
                                   roiList.scores(roiOrder(x)), roiList.flags(roiOrder(x)), ...
                                   100*recall(x), 100*precision(x)), ...
                      1:length(roiOrder), ...
                      'UniformOutput', false) ;


fh = figure ; clf ;
set(gcf,'units', 'normalized') ;

candSel = uicontrol(fh, 'style', 'listbox', ...
                    'String', candLabels, ...
                    'Value', 1, ...
                    'Units', 'normalized', 'Position', [0 0 .10 1], ...
                    'Callback', @displayCand) ;

roiSel = uicontrol(fh, 'style', 'listbox', ...
                   'String', roiLabels, ...
                   'Value', 1, ...
                   'Units', 'normalized', 'Position', [.10 0 .15 1], ...
                   'Callback', @displayRoi) ;

infoH = axes('position', [.25 0  .25  1]) ;
imgH  = axes('position', [.5  0  .5  .5]) ;
gtH   = axes('position', [.5 .5  .5  .5]) ;
display(1) ;

  function [match,gtRois] = matchToGt(ci) % ~~~~~~~~~~~~~
  cands = candList(ci) ;  
  ii = binsearch(imageIndex, cands.imageId) ;
  imageId = imdb.images(ii).id ;
  
  % find instances of this class 
  sel = findRois(roidb, ...
                 'imageId', imageId, ...
                 'jitter', 0, ...
                 'class', roidb.classes.(upper(className))) ;
  
  % and specifically of this aspect
  selAspect = findRois(roidb, ...
                       'subset', sel, ...
                       'aspect', aspectName) ;
  
  % difficult are also other aspects
  gtRois = soaSubsRef(roidb.rois, sel) ;
  gtRois.difficult(~ismember(sel, selAspect)) = 1 ;
  
  match = evalDetections(gtRois.box, ...
                         gtRois.difficult, ...
                         candList(ci).box, ...                
                         candList(ci).score) ;
  end
  
  function displayCand(varargin) % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ci = candOrder(get(candSel, 'value')) ;
  display(ci)
  end

  function displayRoi(varargin) % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ri = roiOrder(get(roiSel, 'value')) ;
  display(roiList.candSel(ri)) ;
  axes(imgH) ; plotroi(roiList.boxes(:,ri), 'b', 'linewidth', 1) ;
  axes(gtH) ; plotroi(roiList.boxes(:,ri), 'b', 'linewidth', 1) ;
  end

  function display(ci) % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  cands = candList(ci) ;
  
  ii = binsearch(imageIndex, cands.imageId) ;
  imagePath = fullfile(imdb.dir, [imdb.images(ii).name '.jpg']) ;
  imageId = imdb.images(ii).id ;
  imageSize = imdb.images(ii).size ;
  imageName = imdb.images(ii).name ;
  imageSet  = decodeEnum(imdb.sets, imdb.images(ii).set) ;
  im = imread(imagePath) ;  
  [match, gtRois] = matchToGt(ci) ;

  str = '' ;
  str = [str sprintf('num cands:\t%d',     length(cands.box))] ;
  str = [str sprintf('\nmax score:\t%g',   max(cands.score))] ;
  str = [str sprintf('\nmin score:\t%g',   min(cands.score))] ;
  str = [str sprintf('\n')] ;
  str = [str sprintf('\nimg id:\t%d',      imageId)] ;
  str = [str sprintf('\nimg name:\t%s',    imageName)] ;
  str = [str sprintf('\nimg set:\t%s',     imageSet)] ;
  str = [str sprintf('\nimg geom:\t%dx%d', imageSize(1), imageSize(2))] ;
  str = [str sprintf('\n')] ;
  
  cla(gtH) ; axes(gtH) ;
  image(im,'parent',gtH) ; hold(gtH,'on') ;
  axis equal off ;
  for bi = size(cands.box,2):-1:1 ;
    switch match.detBoxFlags(bi)
      case +1, cl = 'g' ;
      case -1, cl = 'r' ;
      case 0, cl = 'y' ;
    end    
    plotroi(cands.box(:,bi), 'linewidth', 2, 'color', cl) ;
  end
  for gti = 1:length(gtRois.id)
    if gtRois.difficult(gti)
      plotroi(gtRois.box(:, gti),'w--', 'linewidth', 3, ...
                          'label', sprintf('%d', gti)) ;
    else
      plotroi(gtRois.box(:, gti), 'w-', 'linewidth', 3, ...
                          'label', sprintf('%d', gti)) ;
    end
    str = [str sprintf('\n%d [%012.0f]',gti, gtRois.id(gti))] ;
    str = [str sprintf('\n    aspect:\t%s', decodeEnum(roidb.aspects, gtRois.aspect(gti)))] ;
    str = [str sprintf('\n    difficult:\t%d',gtRois.difficult(gti))] ;
  end
  
  cla(imgH) ; axes(imgH) ;      
  image(im,'parent',imgH) ; hold(imgH,'on') ;
  axis equal off ;  
  cl = colorizeScore(cands.score, bestScore, worstScore) ;
  for bi = size(cands.box,2):-1:1 ;
    plotroi(cands.box(:,bi), 'linewidth', 2, 'color', cl(bi,:)) ;
  end
  
  cla(infoH) ; axes(infoH) ;
  text(0,0, str, ...
       'Backgroundcolor', 'w', ...
       'VerticalAlign', 'top', ...
       'Interpreter','none') ;
  set(gca,'ydir', 'reverse') ;
  axis off ;
  
  drawnow ;
  end % display
end % main

% --------------------------------------------------------------------
function cl = colorizeScore(score, bestScore, worstScore)
% --------------------------------------------------------------------

selp = find(score >= 0) ;
seln = find(score <  0) ;

lamp = score(selp) / bestScore ;
lamn = score(seln) / worstScore ;

cl = zeros(length(score), 3) ;
cl(selp, :) =  lamp(:) * [0 1 0] + (1 - lamp(:)) * [.5 .5 .5] ;
cl(seln, :) =  lamn(:) * [1 0 0] + (1 - lamn(:)) * [.5 .5 .5] ;
end % colorizeScore