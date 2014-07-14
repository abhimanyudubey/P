function roi = newRoi(n)
% NEWROI
%  ROI = NEWROI returns a new ROI structure. ROI = NEWROI(N) returns a
%  structure-of-arrays of ROIs (see AOSTOSAO()).
%
%  Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

if nargin < 1
  n = 1 ;
end

roi.id        = zeros(1,n) ;
roi.imageId   = zeros(1,n) ;
roi.set       = zeros(1,n,'uint8') ;
roi.class     = zeros(1,n,'uint8') ;
roi.aspect    = zeros(1,n,'uint8') ;
roi.difficult = false(1,n) ;
roi.truncated = false(1,n) ;
roi.occluded  = false(1,n) ;
roi.box       = zeros(4,n) ;
roi.jitter    = zeros(1,n,'uint8') ;
