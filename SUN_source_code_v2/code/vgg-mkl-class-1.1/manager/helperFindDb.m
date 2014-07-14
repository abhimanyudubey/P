function sel = helperFindDb(X, stringFields, numericFields, varargin)
% HELPERFINDDB  Helper functions to find stuff in DBs
%
%   See also:: GETKERNELNAMES(), FINDROIS().
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

useRegex = 0 ;
sel = 1:length(X) ;

i = 1 ;
while i <= length(varargin)
  field = varargin{i} ;
  neg   = 0 ;

  if ~ischar(field)
    error('String expected.') ;
  end

  % turn regex on
  if strcmp('regexp', field)
    useRegex = 1 ;
    i = i + 1 ;
    continue ;
  end

  % turn regex off
  if strcmp('noregexp', field)
    useRegex = 0 ;
    i = i + 1 ;
    continue ;
  end

  % get value
  value = varargin{i+1} ;

  % handle value = '*'
  if ischar(value) & strcmp('*', value)
    i = i + 2 ;
    continue ;
  end

  % handle field begins with '~'
  if field(1) == '~'
    neg = 1 ;
    field = field(2:end) ;
  end

  switch field
    case 'subset'
      if ~ neg
        sel = intersect(sel, value) ;
      else
        sel = setdiff(sel, value) ;
      end

    case numericFields
      match = cellfun(@(x) isequalwithequalnans(x, value), {X(sel).(field)}) ;
      if neg, match = ~ match ; end
      sel = sel(match) ;

    case stringFields
      if ~ useRegex
        match = strcmp(value, {X(sel).(field)}) ;
      else
        match = cellfun(@(x)~isempty(x), regexp({X(sel).(field)}, value)) ;
      end
      if neg, match = ~ match ; end
      sel = sel(match) ;

    otherwise
      warning(sprintf('''%s'' is not a recognized field. Skipping.', field)) ;
  end
  i = i + 2 ;
end
