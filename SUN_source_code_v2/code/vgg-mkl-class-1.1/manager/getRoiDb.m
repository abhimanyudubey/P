function roidb = getRoiDb(roiDbPath)
% GETROIDB Cached loading of a ROI DB
%   ROIDB = GETROIDB(ROIDBPATH)
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

roidb = cacheGet(mfilename, {}, {roiDbPath}) ;
if isempty(roidb)
  roidb = load(roiDbPath) ;
  cacheStore(mfilename, roidb, {}, {roiDbPath}) ;
end
