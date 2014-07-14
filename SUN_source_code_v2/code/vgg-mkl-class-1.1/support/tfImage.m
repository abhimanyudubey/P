function I = tfImage(tfName, I)
% TFIMAGE
%   I = TFIMAGE(TFNAME, I)
%
%   See also:: TFBOX().
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

fprintf('tfImage: applying transformation ''%s''.\n', tfName) ;

[imgH,imgW,K] = size(I) ;

ur = 1:imgW ;
vr = 1:imgH ;
[u,v] = meshgrid(ur,vr) ;

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
R = zm * [ sx * cos(angle),  - sx * sin(angle) ; sin(angle), cos(angle)] ;
T = [(imgW+1)/2 ; (imgH+1)/2] ;
T = -R*T + T ;

T = - inv(R) * T ; 
R = inv(R) ;
[u,v] = waffine(R,T,u,v) ;
I = imwbackward(ur,vr,I,u,v) ;

% some corner pixels may still be "black", which is signaled by NaNs
% values.
I(isnan(I)) = 0 ;
