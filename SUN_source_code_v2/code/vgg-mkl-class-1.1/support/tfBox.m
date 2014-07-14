function box = tfBox(imgW, imgH, tfName, box)
% TFBOX
%   BOX = TFBOX(IMGW, IMGH, TFNAME, BOX)
%
%   See also:: TFIMAGE().
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

zoom  = 1 ;
angle = 0 ;
sx    = 1 ;

switch lower(tfName)
  case {'', 'none', 'identity'}
  
  case {'zm1'}
    zoom = 1.1 ;
    
  case {'zm2'}
    zoom = 1.2 ;
    
  case {'zm3'}
    zoom = 1.3 ;
    
  case {'fliplr'}
    sx    = -1 ;
  
  case {'rp5'}
    angle = 5/180*pi ;
    
  case {'rm5'}
    angle = -5/180*pi ;
    
  case {'fliplr_rp5'}
    angle = 5/180*pi ;
    sx    = -1 ;
    
  case {'fliplr_rm5'}
    angle = - 5/180*pi ;
    sx    = -1 ;
    
  otherwise
    error('Unknown transformation ''%s''', tfName) ;
end

if imgW >= imgH
  a = imgW ;
  b = imgH ;
else
  a = imgH ;
  b = imgW ;
end

zm = cos(atan(a/b)-abs(angle)) ;
zm = zoom * sqrt(a^2 + b^2) / b * zm ;
R = zm * [sx * cos(angle),  - sx * sin(angle) ; sin(angle), cos(angle)] ;
T = [(imgW+1)/2 ; (imgH+1)/2] ;
T = -R*T + T ;

q1 = waffine(R,T, box([1 2])) ;
q2 = waffine(R,T, box([3 4])) ;
q3 = waffine(R,T, box([1 4])) ;
q4 = waffine(R,T, box([3 2])) ;

p1 = min([q1 q2 q3 q4],[],2) ;
p2 = max([q1 q2 q3 q4],[],2) ;

p1(1) = max(1,min(imgW, p1(1))) ;
p1(2) = max(1,min(imgH, p1(2))) ;
p2(1) = max(1,min(imgW, p2(1))) ;
p2(2) = max(1,min(imgH, p2(2))) ;

box = round([p1;p2]) ;
