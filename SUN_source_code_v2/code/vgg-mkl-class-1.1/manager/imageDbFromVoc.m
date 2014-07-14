function imdb = imageDbFromVoc(root, varargin)
% IMAGEDBFROMVOC   Construct an image database from VOC data
%   IMDB = IMAGEDBFROMVOC(ROOT, SUB1, SUB2, ...) returns the specified
%   subsets of the VOC 2009 data. Here SUBN is one of 'train09',
%   'val09', 'test09', 'train08', 'val08', 'test08', 'train07',
%   'val07', 'test07'. ROOT is the path of the VOCDevkit package.
%
%   IMDB.IMAGES.ID        = image ID
%   IMDB.IMAGES.NAME      = image name
%   IMDB.IMAGES.SET       = image set code
%   IMDB.DIR              = directory holding the .JPG files
%   IMDB.SETS.TRAIN09     = code for 2009 TRAIN set
%   IMDB.SETS.VAL09       = code for 2009 VAL set
%   IMDB.SETS.TEST09      = code for 2009 TEST set
%   IMDB.SETS.TRAIN08     = code for 2008 TRAIN set
%   IMDB.SETS.VAL08       = code for 2008 VAL set
%   IMDB.SETS.TEST08      = code for 2008 TEST set
%   IMDB.SETS.TRAIN07     = code for 2007 TRAIN set
%   IMDB.SETS.VAL07       = code for 2007 VAL set
%   IMDB.SETS.TEST07      = code for 2007 TEST set
%
%   Note that the 2009 data is a superset of the 2008 data. Specifying
%   both may result in duplicated images being selected.
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

sets = varargin ;

VOCopts09 = getVocOpts(root, '2009') ;
VOCopts08 = getVocOpts(root, '2008') ;
VOCopts07 = getVocOpts(root, '2007') ;

templ09 = VOCopts09.imgsetpath ;
templ08 = VOCopts08.imgsetpath ;
templ07 = VOCopts07.imgsetpath ;

imdb.sets.TRAIN07   = uint8(1) ;
imdb.sets.VAL07     = uint8(2) ;
imdb.sets.TEST07    = uint8(3) ;
imdb.sets.TRAIN08   = uint8(4) ;
imdb.sets.VAL08     = uint8(5) ;
imdb.sets.TEST08    = uint8(6) ;
imdb.sets.TRAIN09   = uint8(7) ;
imdb.sets.VAL09     = uint8(8) ;
imdb.sets.TEST09    = uint8(9) ;

% determine the most recend edition installed
imdb.dir = root ;

names    = {} ;
setCodes = {} ;
for si = 1:length(sets)
  switch sets{si}
    case 'train09'
      [n, s] = loadImageSetHelper(VOCopts09, 'train', imdb.sets.TRAIN09) ;
    case 'val09'
      [n, s] = loadImageSetHelper(VOCopts09, 'val',   imdb.sets.VAL09) ;
    case 'test09'
      [n, s] = loadImageSetHelper(VOCopts09, 'test',  imdb.sets.TEST09) ;

    case 'train08'
      [n, s] = loadImageSetHelper(VOCopts08, 'train', imdb.sets.TRAIN08) ;
    case 'val08'
      [n, s] = loadImageSetHelper(VOCopts08, 'val',   imdb.sets.VAL08) ;
    case 'test08'
      [n, s] = loadImageSetHelper(VOCopts08, 'test',  imdb.sets.TEST08) ;

    case 'train07'
      [n, s] = loadImageSetHelper(VOCopts07, 'train', imdb.sets.TRAIN07) ;
    case 'val07'
      [n, s] = loadImageSetHelper(VOCopts07, 'val',   imdb.sets.VAL07) ;
    case 'test07'
      [n, s] = loadImageSetHelper(VOCopts07, 'test',  imdb.sets.TEST07) ;

    otherwise
      error('Unknown subset ''%s''', sets{si}) ;
  end
  names{end+1} = n ;
  setCodes{end+1} = s ;
end

names = cat(1, names{:})';
setCodes = num2cell(cat(2, setCodes{:})) ;

% check for duplicates
tmp = strvcat(names) ;
if size(tmp, 2) > size(unique(tmp, 'rows'), 2)
  error('Detected duplicated images. See HELP %s', mfilename) ;
end

% check all images
fprintf('%s: checking images and getting sizes ...\n', mfilename) ;
for ii=1:length(names)
  fprintf('\r%5.1f %%: %s', ii/length(names)*100, names{ii}) ;
  info = imfinfo(fullfile(imdb.dir, names{ii})) ;
  imdb.images(ii).name  = names{ii} ;
  imdb.images(ii).set   = setCodes{ii} ;
  imdb.images(ii).size  = [info.Width; info.Height] ;
end
fprintf('\n') ;

% assign IDs
groupNumber = 0 ;
setNumbers  = { 00, 01, 100, ...
                10, 11, 110, ...
                20, 21, 120  } ;
setNames    = {'train07', 'val07', 'test07', ...
               'train08', 'val08', 'test08', ...
               'train09', 'val09', 'test09'  } ;

for si=1:length(setNames)
  setName = upper(char(setNames{si})) ;
  sel = find([imdb.images.set] == imdb.sets.(setName)) ;
  for ii=1:length(sel)
    imdb.images(sel(ii)).id = ii + 1e7 * setNumbers{si} ;
  end
end

% sort by ID
[drop, perm] = sort([imdb.images.id]) ;
imdb.images = imdb.images(perm) ;
imdb.images = aosToSoa(imdb.images) ;

% --------------------------------------------------------------------
function [names, setCodes] = loadImageSetHelper(VOCopts, setName, setCode)
% --------------------------------------------------------------------

root     = VOCopts.datadir ;
imgpatht = VOCopts.imgpath ;
setpatht = VOCopts.imgsetpath ;

setPath = sprintf(setpatht, setName) ;
fprintf('%s: adding ''%s''\n', mfilename, setPath) ;
names = textread(setPath, '%s') ;
setCodes = setCode(ones(1, length(names))) ;

for ni=1:length(names)
  fullName = sprintf(imgpatht, names{ni}) ;
  fullName = fullName(length(root)+1:end) ;
  if fullName(1) == '/', fullName(1) = [] ; end ;
  names{ni} = fullName ;
end
