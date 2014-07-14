function ci = cacheMatch(cacheName, depArgs, depFiles)
% CACHEMATCH  Match cache entry
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

if nargin < 3
  depFiles = {}  ;
end

% check if cache exists
global caches ;
if ~isfield(caches, cacheName)
  ci = [] ;
  return ;
end

% number of entries in cache
cn = length(caches.(cacheName)) ;

% search for valid entry
good = 0 ;
for ci=1:cn
  good = 1 ;  
  good = good & isequalwithequalnans(depArgs, ...
                                     caches.(cacheName){ci}.depArgs) ;  
  for fi = 1:length(depFiles)
    if exist(depFiles{fi}, 'file')
      d = dir(depFiles{fi}) ;
      fileDate = d.date ;
    elseif exist([depFiles{fi} '.mat'], 'file')
      d = dir([depFiles{fi} '.mat']) ;
      fileDate = d.date ;
    else
      fileDate = 'future' ;
    end
    if ~isfield(caches.(cacheName){ci}, 'depFiles'), continue ; end
    if length(caches.(cacheName){ci}.depFiles) < fi, continue ; end
    good = good & ...
           strcmp(caches.(cacheName){ci}.depFiles(fi).path, depFiles{fi}) ;
    good = good & ...
           strcmp(caches.(cacheName){ci}.depFiles(fi).date, fileDate) ;
  end
  if good, break ; end
end

if ~good, ci = [] ; end
