function cacheStore(cacheName, X, depArgs, depFiles, maxEntries)
% CACHESTORE  Store value in cache
%
%   CACHESTORE(CACHENAME, X, DEPARGS, DEPFILES, MAXENTRIES)
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

global caches ;

if nargin < 5
  maxEntries = 2 ;
end

if isfield(caches, cacheName)
  ci = cacheMatch(cacheName, depArgs, depFiles) ;
  if isempty(ci)
    if length(caches.(cacheName)) < maxEntries
      ci = length(caches.(cacheName)) + 1 ;
    else
      [caches.(cacheName){2:end}] = caches.(cacheName){1:end-1} ;
      ci = 1 ;
    end
  end
else
  ci = 1 ;
end

caches.(cacheName){ci}.X = X ;
caches.(cacheName){ci}.depArgs = depArgs ;
for fi=1:length(depFiles)
  d = dir(depFiles{fi}) ;
  caches.(cacheName){ci}.depFiles(fi).path = depFiles{fi} ;
  caches.(cacheName){ci}.depFiles(fi).date = d.date ;
end
