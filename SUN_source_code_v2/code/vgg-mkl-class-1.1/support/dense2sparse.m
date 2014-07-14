function feat = dense2sparse(dfeat)
% DENSE2SPARSE  Convert sparse feature representation to to dense
%   FEAT = DENSE2SPARSE(SFEAT)
%
%   See also:: SPARSE2DENSE()
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

sel = find(dfeat.wordMap) ;
[i,j] = find(dfeat.wordMap) ;

sel = sel' ;
i = i' ;
j = j' ;

feat.frames = [dfeat.xr(j) ; dfeat.yr(i)] ;
feat.words  = dfeat.wordMap(sel) ;

if isfield(dfeat, 'scaleMap')
  feat.frames = cat(1, feat.frames, dfeat.scaleMap(sel)) ;
end

if isfield(feat, 'weightMap')
  feat.weights = dfeat.weightMap(sel) ;
end
