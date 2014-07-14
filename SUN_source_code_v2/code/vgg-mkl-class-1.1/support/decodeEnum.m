function name = decodeEnum(enum, value)
% DECODEENUM
%
%  NAME = DECODEENUM(ENUM, VALUE)
%
%  Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

names = fieldnames(enum)' ;

for name = names
  name = char(name) ;
  thisValue = enum.(name) ;
  ok = false ;
  if isnumeric(thisValue)
    ok = isequalwithequalnans(double(thisValue), double(value)) ;
  else
    ok = isequa(thisValue, value) ;
  end
  if ok, return ; end
end

% not found !
name = [] ;
