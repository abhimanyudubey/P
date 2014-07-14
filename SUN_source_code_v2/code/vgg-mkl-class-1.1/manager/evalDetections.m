function match = evalDetections(gtBoxes, gtDifficult, detBoxes, detScores, overlThresh)
% EVALDETECTIONS
%   MATCH = EVALDETECTIONS(GTBOXES, GTDIFFICUTL, DETBOXES, DETSCORES)
%
%   MATCH.DETBOXFLAGS: +1 good, 0 match to difficult, -1 wrong
%   MATCH.DETBOXTOGT:  map to matched GT,  NaN if no match
%   MATCH.GTBOXTODET:  map to matched Det, NaN if missed, 0 if difficult
%   MATCH.SCORES:      for PR curve (miss boxes have -inf score)
%   MATCH.LABELS:      for PR curve
%
%   Auhtor:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

if nargin < 5
  overlThresh = 0.5 ;
end

nGtBoxes  = size(gtBoxes, 2) ;
nDetBoxes = size(detBoxes, 2) ;

gtBoxToDet  = NaN * ones(1, nGtBoxes) ;
detBoxToGt  = NaN * zeros(1, nDetBoxes) ;
detBoxFlags = - ones(1,nDetBoxes) ;

if isempty(gtBoxes)
  match.detBoxFlags = detBoxFlags ;
  match.detBoxToGt  = detBoxToGt ;
  match.gtBoxToDet  = [] ;
  match.scores      = detScores ;
  match.labels      = -ones(1,size(detBoxes,2)) ;
  return ;
end

[overl, allDetBoxToGt] = max(calcBoxOverlap(detBoxes, gtBoxes), [], 2) ;

% prematch to difficult
selDiff = find((overl > overlThresh) & gtDifficult(1,allDetBoxToGt)') ;
detBoxFlags(selDiff) = 0 ;
detBoxToGt(selDiff)  = allDetBoxToGt(selDiff) ;
gtBoxToDet(gtDifficult) = 0 ;

% now match the rest
selDetOk = find(overl > overlThresh) ;

nMiss = sum(~gtDifficult) ;
for oki = 1:length(selDetOk)
  if nMiss == 0, break ; end

  dei = selDetOk(oki) ;
  gti = allDetBoxToGt(dei) ;

  % already matched?
  if ~isnan(gtBoxToDet(gti))
    continue ;
  end

  gtBoxToDet(gti)  = dei ;
  detBoxToGt(dei)  = gti ;
  detBoxFlags(dei) = + 1 ;
  nMiss = nMiss - 1 ;
end

% now calculate equivalent (scores, labels) pair
selM   = find(detBoxFlags == +1) ;
selDM  = find(detBoxFlags == -1) ;

scores = [detScores(selM),      detScores(selDM),        -inf*ones(1,nMiss)] ;
labels = [ones(1,length(selM)), -ones(1, length(selDM)), ones(1,nMiss)] ;

match.detBoxFlags = detBoxFlags ;
match.detBoxToGt  = detBoxToGt ;
match.gtBoxToDet = gtBoxToDet ;
match.scores = scores ;
match.labels = labels ;
