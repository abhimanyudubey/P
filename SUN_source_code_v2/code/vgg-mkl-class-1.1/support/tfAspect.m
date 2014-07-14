function aspect = tfAspect(tfName, aspect)
% TFASPECT
%
%   See also:: TFIMAGE()
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

switch lower(tfName)
  case {'', 'none', 'identity'}
    return ;

  case {'rp5', 'rm5'}
    return ;

  case {'fliplr_rp5', 'fliplr_rm5', 'fliplr'}
    switch lower(aspect)
      case {'front', 'rear', 'misc'}
        return ;
      case {'left'}
        aspect = 'right' ;
      case {'right'}
        aspect = 'left' ;
      otherwise
        error(sprintf('Unknown aspect ''%s'', aspect')) ;
    end

  otherwise
    error(sprintf('Uknown tf ''%s''', tfName)) ;
end
