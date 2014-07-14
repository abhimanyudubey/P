function sel = findImages(imdb, varargin)
% FINDIMAGES  Searches images database for matching imags
%   SEL = FINDIMAGES(IMDB, FIELD, VAL, ...) returns the indexes of the
%   images of the database IMDB that match the query.  The query is a
%   list of image structure fields and corresponding values.
%
%   The function mirrors FINDROIS().
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

useRegex = 0 ;
sel = 1:length(imdb.images.id) ;

i = 1 ;
while i <= length(varargin)
  field = varargin{i} ;
  i = i + 1 ;

  if ~ischar(field)
    error('String expected.') ;
  end

  neg = 0 ;
  if field(1) == '~'
    field = field(2:end) ;
    neg = 1 ;
  end

  value = varargin{i} ;
  i = i + 1 ;

  switch field
    case 'subset'
      if ~ neg
        sel = intersect(sel, value) ;
      else
        sel = setdiff(sel, value) ;
      end

    case {'set'}
      if isnumeric(value)
        match = double([imdb.images.(field)(:,sel)]) == double(value) ;
      else
        enumName = [field 's'] ;
        enumItems = fieldnames(imdb.(enumName))' ;
        match = false(1, length(sel)) ;
        for enumItem = enumItems
          enumItem = char(enumItem) ;
          str = regexpi(enumItem, value, 'match', 'once') ;
          if length(str) == length(enumItem)
            match = match | ([imdb.images.(field)(:,sel)] == imdb.(enumName).(enumItem)) ;
          end
        end
      end
      if neg
        match = ~match ;
      end
      sel = sel(match) ;

    case {'id'}
      match = double([imdb.images.id(:,field)]) == double(value) ;
      if neg
        match = ~match ;
      end
      sel = sel(match) ;

    case {'name'}
      error('Not implemented yet.') ;

    otherwise
      error(sprintf('Uknown field ''%s''.', field)) ;
  end % field

end % next predicate
