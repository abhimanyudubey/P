function sel = findKernels(kerdb, varargin)
% FINDKERNELS  Get names of the kernels
%   SEL = FINDKERNELS(CONF, QUERY...) returns a list of indeces
%   of the entry in KERDB matching the QUERY. It works similarly to
%   FINDROIS().
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

sel = helperFindDb(kerdb,                     ...
                   {'type', 'feat', 'name'},  ...
                   {'pyrLevel'},              ...
                   varargin{:}) ;
