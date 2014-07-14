function matrix = gbDistance(a, b)
% GBDISTANCE Geometric-Blur descriptor sets distance
%   MATRIX = GBDISTANCE(A, B) computes a similarity matrix between
%   images A and B. A and B are cells array, each element of which is
%   a colllection of features (e.g. SIFT descriptors). The similarity
%   between a pair of images is the average minimum distance between
%   descriptors from the two images.
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

matrix = zeros(length(a), length(b)) ;

tic

for na=1:length(a)
  kd = vl_kdtreebuild(single(a{na}), 'thresholdmethod', 'mean') ;
  for nb=1:length(b)
    [result, ndists] = vl_kdtreequery(kd, single(a{na}), single(b{nb}), ...
                                      'maxcomparisons', 50) ;
    matrix(na,nb) = mean(ndists) ;
  end
  if any(any(isnan(matrix))), keyboard ; end
  fprintf('%5.2f %%: %5.1fm remaining.\n', ...
          na/(length(a)+length(b))*100, ...
          toc / na * (length(a)+length(b)-na) / 60) ;
end
for nb=1:length(b)
  kd = vl_kdtreebuild(single(b{nb}), 'thresholdmethod', 'mean') ;
  for na=1:length(a)
    [result, ndists] = vl_kdtreequery(kd, single(b{nb}), single(a{na}), ...
                                      'maxcomparisons', 50) ;
    matrix(na,nb) = matrix(na,nb) + mean(ndists) ;
  end
  fprintf('%5.2f %%: %5.1fm remaining.\n', ...
          (nb+length(a))/(length(a)+length(b))*100, ...
          toc / (nb+length(a)) * (length(a)+length(b)-na) / 60) ;
end
