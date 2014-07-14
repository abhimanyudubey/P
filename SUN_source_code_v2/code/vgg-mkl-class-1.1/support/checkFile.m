function ok = checkFile(fileName, varargin)
% CHECKFILE  Check for existence and sanity of file
%   OK = CHECKFILE(FILENAME) returns TRUE if the file exists (possibly
%   after appending the .MAT suffix).
%
%   OK = CHECKFULE(FILENAME, 'SHALLOW', FALSE) also check that the
%   file is a .MAT file and it is not corrupted by attempting to load
%   it (this can be slow).
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

opts.shallow = true ;
opts.verbose = true ;
opts = vgg_argparse(opts, varargin) ;

ok = 1 ;

if ~ exist(fileName, 'file') && ...
   ~ exist([fileName '.mat'], 'file') 
  ok = 0 ;
  return ;
end

if ~ opts.shallow
  try
    load(fileName) ;
  catch
    ok = 0 ;
    if opts.verbose, 
      fprintf('checkFile: corrupted ''%s''!\n', fileName) ;
    end
    return ;
  end
end
