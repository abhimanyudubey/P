function X = cacheGet(cacheName, depArgs, depFiles)
% CACHEGET
%   X = CACHEGET(CACHENAME, DEPARGS, DEPFILES)
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

ci = cacheMatch(cacheName, depArgs, depFiles) ;
if isempty(ci), X = [] ; return ; end

global caches ;
X = caches.(cacheName){ci}.X ;
