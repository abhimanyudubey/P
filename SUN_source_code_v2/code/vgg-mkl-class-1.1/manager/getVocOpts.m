function VOCopts = getVocOpts(root, edition)
% GETVOCOPTS  Retrieve the VOCopts structure from VOC 2007/2008 devkit
%   VOCopts = GETVOCOPTS(ROOT) retrieves the VOCopts structure from
%   the VOC devkit found at ROOT for the 2008 data.
%
%   VOCopts = GETVOCOPTS(ROOT, '2007') does the same for the 2007
%   data.
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

switch edition
  case '2009'
    edition = 'VOC2009' ;

  case '2008'
    edition = 'VOC2008' ;

  case '2007'
    edition = 'VOC2007' ;

  otherwise
    error('Uknown edition ''%s''', edition) ;
end

c = pwd ;
try
  cd(fullfile(root, 'VOCcode')) ;
  s = patchVOCinit(edition) ;
  eval(s) ;
catch
  error(sprintf('Could not find VOCinit in ''%s/VOCCode''.', root)) ;
end
cd(c) ;

% --------------------------------------------------------------------
function s = patchVOCinit(edition)
% --------------------------------------------------------------------

name = which('VOCinit') ;
dirName = strrep(fileparts(fileparts(name)),'\','/') ;

% source
f = fopen(name,'r') ;
s = fread(f,+inf,'*char')' ;
fclose(f) ;

lineEnds = find(s == sprintf('\n')) ;

header = sprintf('VOCopts.dataset=''%s'' ;\ndevkitroot=''%s'';\n', ...
                 edition, dirName) ;

s = [header, s(lineEnds(16)+1:end)] ;
