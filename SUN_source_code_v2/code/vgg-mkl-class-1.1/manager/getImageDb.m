function imdb = getImageDb(imageDbPath)
% GETIMAGEDB Cached loading of the image DB
%   IMDB = GETIMAGEDB(IMAGEDBPATH)
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

imdb = cacheGet(mfilename, {}, {imageDbPath}) ;
if isempty(imdb)
  imdb = load(imageDbPath) ;
  cacheStore(mfilename, imdb, {}, {imageDbPath}) ;
end
