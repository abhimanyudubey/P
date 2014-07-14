function a = soaSubsAsgn(a, subs, b)
% SOASUBSASGN Subscript Assignment of structure-of-arrays
%
%  Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

names = fieldnames(a)' ;

for name = names
  name = char(name) ;
  a.(name)(:,subs) = b.(name) ;
end
