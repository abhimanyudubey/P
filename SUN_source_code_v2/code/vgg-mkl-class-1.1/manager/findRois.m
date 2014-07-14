function sel = findRois(roidb, varargin)
% FINDROIS  Searches ROIs database for matching ROIs
%   SEL = FINDROIS(ROIDB, FIELD, VAL, ...) returns the indexes of the
%   ROIs of the database ROIDB.ROIS that match the query.  The query
%   is a list of ROI structure fields and corresponding values.
%
%   An empty value matches any value (except for the SUBSET field).
%
%   To negate a match prefix the character '~' to the field name.
%
%   To restrict the search to a subset SUB of ROIs use
%   FINDROIS(ROIDB,'SUBSET',SUB).
%
%   All ROI fields have numerical values (either logical, integer, or
%   floating point). All values are casted to DOUBLE before
%   comparison.
%
%   The fields SET, CLASS, ASPECT, and JITTER have also symbolic
%   names. It is possible to match such fields by specifying as value
%   a regular expression rather than a number.
%
%   JITTER is treated a little differently from the other
%   fields. First, the jitter name NONE defaults to zero, even if no
%   such a name is defined in ROIDB.JITTERS. Second, not specifying
%   JITTER or SUBSET returns only the ROIs with no jitter (specify
%   '.*' or [] to pickup all ROIs).
%
%   Examples:: To find all the ROIs in the TRAIN set, use either:
%
%       sel = findRois(roidb, 'set', roidb.sets.TRAIN)
%       sel = findROis(roidb, 'set', 'train'
%
%     To find all non-difficult, non-jittered ROIs of LEFT
%     facing AEROPLANES or CARS use:
%
%       sel = findRois(roidb, ...
%                      'jitter', 0, ...
%                      'class', 'aeroplane|car', ...
%                      'aspect', 'left') ;
%
%     To find all the ROIs which are not an aeroplane use:
%
%       sel = findRois(roidb, '~class', 'aeroplane')
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

useRegex = false ;
jitterSpecified = false ;
sel = 1:length(roidb.rois.id) ;

i = 1 ;
while i <= length(varargin)
  field = varargin{i} ;
  i = i + 1 ;

  if ~ischar(field)
    error('String expected.') ;
  end

  neg = false ;
  if field(1) == '~'
    field = field(2:end) ;
    neg = true ;
  end

  value = varargin{i} ;
  i = i + 1 ;

  % empty value matches anything
  if isempty(value) && ~ strcmp(field, 'subset')
    continue ;
  end

  switch field

    case 'subset'
      if ~ neg
        sel = intersect(sel, value) ;
      else
        sel = setdiff(sel, value) ;
      end
      jitterSpecified = true ;

    case {'set', 'class', 'aspect', 'jitter'}
      if isnumeric(value) | islogical(value)
        match = double(roidb.rois.(field)(:,sel)) == double(value) ;
        jitterSpecified = jitterSpecified | strcmp(field, 'jitter') ;
      else
        if strcmp(field, 'class')
          enumName = 'classes' ;
        else
          enumName = [field 's'] ;
        end
        enum = roidb.(enumName) ;
        enumItems = fieldnames(enum)' ;
        if strcmp(field, 'jitter')
          enum.NONE = uint8(0) ;
          enumItems = {'NONE', enumItems{:}} ;
          jitterSpecified = true ;
        end
        match = false(1, length(sel)) ;
        for enumItem = enumItems
          enumItem = char(enumItem) ;
          str = regexpi(enumItem, value, 'match', 'once') ;
          if length(str) == length(enumItem)
            match = match | all(roidb.rois.(field)(:,sel) == enum.(enumItem), 1) ;
          end
        end
      end
      if neg
        match = ~match ;
      end
      sel = sel(match) ;

    case {'id', 'imageId', 'difficult', 'truncated', 'jitter', 'occluded', 'box'}
      match = all(double(roidb.rois.(field)(:, sel)) == double(value), 1) ;
      if neg
        match = ~match ;
      end
      sel = sel(match) ;

    otherwise
      error('Unknown field ''%s''.', field) ;
  end % field cases

end % next predicate


% if no jitter has been specified, select null jitter
if ~ jitterSpecified
  match = (roidb.rois.jitter(sel) == uint8(0)) ;
  sel = sel(match) ;
end
