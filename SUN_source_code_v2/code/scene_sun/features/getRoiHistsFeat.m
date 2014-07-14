function levels = getRoiHistsFeat(feat, featOpts, pyrLevels, boxes, varargin)
% GETROIHISTS
%
%  wordPath:
%  featOpts:
%  pyrLevels:
%
%  project:
%
%  comprMap
%    Relabel any word by the words specified by this vector. This can
%    be used, for instance, to apply class-dependent compression to
%    the vocabulary. The total number of words becomes equal to
%    MAX(COMPRMAP).
%
%  vocabWeights
%    Reweight feature masses by these word-depending weights. This
%    can be use, for instance, to apply class-dependent
%    discriminative weights.
%
%  forceDense
%    Causes the feature representation to be converted to dense format
%    before going on with the calculations.
%
%  forceIntHist
%    Causes the feature representation to be converted to integral
%    histograms.
%
%  Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
% 
% This file is part of VGG MKL classification and detection code,
% available in the terms of the GNU General Public License version 2.


% TODO: adjust pyrBoxes to work with negative coordinates
% TODO: verify tile borders

opts.project = [] ;
opts.vocabWeights = [] ;
opts.comprMap = [] ;
opts.magnif = 0 ;
opts.normalize = true ;
opts.forceDense = false ;
opts.forceIntHist = false ;

opts = vl_argparse(opts, varargin{:}) ;

% is sparse? do we project?
spar = strcmp(featOpts.format, 'sparse') ;
proj = ~ isempty(opts.project) ;
numWords = featOpts.vocabSize ;

% load word (feature data) file
% if exist(wordPath, 'file')
%   feat = load(wordPath) ;
% elseif exist([wordPath '.mat'], 'file')
%   feat = load([wordPath '.mat']) ;
% else
%   error('Could not find file %s[.mat]', wordPath) ;
% end

%spar = true ;
%feat = dense2sparse(feat) ;

% generate weights if missing
weighed = true  ;
if spar
  if ~ isfield(feat, 'weights')
    feat.weights = ones(size(feat.words)) ;
    weighed = false ;
  end
else
  if ~ isfield(feat, 'weightMap')
    feat.weightMap = ones(size(feat.wordMap)) ;
    weighed = false ;
  end
end

% apply compression map if any
if ~ isempty(opts.comprMap)
  % fixes null caused by zero prob. features
  opts.comprMap = max(opts.comprMap, 1) ;
  if spar
    feat.words = opts.comprMap(feat.words) ;
  else
    sel = find(feat.wordMap) ;
    feat.wordMap(sel) = opts.comprMap(feat.wordMap(sel)) ;
  end
  numWords = max(opts.comprMap) ;
end

if (opts.forceDense | opts.forceIntHist) & spar
  feat = sparse2dense(feat) ;
  spar = false ;
end

% apply vocabWeights to weights if any
if ~ isempty(opts.vocabWeights)
  if spar
    feat.weights = feat.weights .* opts.vocabWeights(feat.words) ;
    weighed = true ;
  else
    feat.weightMap = feat.weightMap .* opts.vocabWeights(feat.wordMap) ;
    weighed = true ;
  end
end

% decide the representation format
useIntImage = opts.forceIntHist | (size(boxes, 2) > 300) ;

if useIntImage
  if spar
    feat = sparse2dense(feat) ;
    spar = false ;
  end
  feat.wordMap = uint32(feat.wordMap) ;
  if weighed
    if ~ proj
      weightIntHist = vl_inthist(feat.wordMap, ...
                                 'mass', feat.weightMap, ...
                                 'numLabels', numWords) ;
    else
      massIntHist = vl_imintegral(feat.weightMap .* (feat.wordMap ~= 0)) ;
    end
  else
    if ~ proj
      weightIntHist = vl_inthist(feat.wordMap, ...
                                 'numLabels', numWords) ;
    else
      massIntHist = vl_imintegral(double(feat.wordMap ~= 0)) ;
    end
  end
  format = 'inthist' ;
else
  if spar
    format = 'sparse' ;
  else
    format = 'dense' ;
  end
end

% apply magnification ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

boxes = double(boxes) ;

if opts.magnif ~= 0
  sz = boxes(3:4,:) - boxes(1:2,:) ;
  cn = .5*(boxes(3:4,:) + boxes(1:2,:)) ;
  sz = sz * (1 + opts.magnif) ;
  boxes = [cn(1,:) - .5  * sz(1,:) ; ...
           cn(2,:) - .5  * sz(2,:) ; ...
           cn(1,:) + .5  * sz(1,:) ; ...
           cn(2,:) + .5  * sz(2,:)  ] ;
end

% --------------------------------------------------------------------
%                                                             go go go
% --------------------------------------------------------------------

if proj
  numWordsBeforeProj = numWords ;
  numWords = 1 ;
end

for li = 1:length(pyrLevels)
  L = pyrLevels(li) ;
  name = sprintf('L%d', L) ;

  % now ...
  numTiles = 2^L ;
  numTiles2 = numTiles^2 ;
  numBoxes = size(boxes, 2) ;
  dx = (boxes(3,:) - boxes(1,:) + 1) / numTiles ;
  dy = (boxes(4,:) - boxes(2,:) + 1) / numTiles ;


  % Integrate tiles --------------------------------------------------

  % The box is the set [xmin -.5, xmax +.5) x [ymin -.5 x ymax +.5) and
  % it is bronek in numTiles x numTiles tiles.

  for x=1:numTiles
    for y=1:numTiles

      % tile geometry
      tiles = [boxes(1:2,:) - .5 + [dx * (x - 1) ; dy * (y - 1)] ;
               boxes(1:2,:) - .5 + [dx * (x    ) ; dy * (y    )]] ;

      % tile index
      i = y + (x - 1) * numTiles ;

      if proj
        % if projection is enabled, extract from opts.project the
        % word weights for this tile
        histComps = numWordsBeforeProj * (i - 1) ...
            + (1:numWordsBeforeProj) ;
        wordCoeffs = opts.project{li}(histComps) ;

        % add a padding for words equal to 0
        wordCoeffs = [0 wordCoeffs(:)'] ;
      end
      hacc = zeros(numWords, numBoxes) ;
      macc = zeros(1, numBoxes) ;

      % use one of three representation
      switch format
        case 'sparse' % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
          if ~ proj
            w = double(feat.weights) ;
            r = double(feat.words) ;
          else
            m = double(feat.weights) .* (feat.words ~= 0) ;
            w = m .* wordCoeffs(feat.words + 1) ;
            r = ones(size(feat.words)) ;
          end

          for bi = 1:numBoxes
            match = ...
                tiles(1,bi) <= feat.frames(1,:) & ...
                tiles(2,bi) <= feat.frames(2,:) & ...
                tiles(3,bi) >  feat.frames(1,:) & ...
                tiles(4,bi) >  feat.frames(2,:) ;
            hacc(:, bi) = vl_binsum(hacc(:, bi), ...
                                    w(match),    ...
                                    r(match)     ) ;
            if proj, macc(bi) = sum(m(match)) ; end
          end

        case 'dense' % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
          if ~ proj
            w = double(feat.weightMap) ;
            r = double(feat.wordMap) ;
          else
            m = double(feat.weightMap) .* (feat.wordMap ~= 0) ;
            w = m .* wordCoeffs(feat.wordMap + 1) ;
            r = ones(size(feat.wordMap)) ;
          end

          % map image coordinates to map indeces
          jmin = vl_binsearch(feat.xr, tiles(1,:)) + 1 ;
          imin = vl_binsearch(feat.yr, tiles(2,:)) + 1 ;
          jmax = vl_binsearch(feat.xr, tiles(3,:))     ;
          imax = vl_binsearch(feat.yr, tiles(4,:))     ;

          for bi = 1:numBoxes
            w_ = w(imin(bi):imax(bi), jmin(bi):jmax(bi)) ;
            r_ = r(imin(bi):imax(bi), jmin(bi):jmax(bi)) ;
            hacc(:, bi) = vl_binsum(hacc(:, bi), w_, r_) ;
            if proj,
              m_ = m(imin(bi):imax(bi), jmin(bi):jmax(bi)) ;
              macc(bi) = sum(sum(m_)) ;
            end
          end

        case 'inthist' % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

          % TODO fix wordMap entires = 0 when projecting and
          % computing mass

          if ~ proj
            w = weightIntHist ;
            % r is implicit
          else
            m = massIntHist ;
            w = vl_imintegral(...
              feat.weightMap .* wordCoeffs(feat.wordMap + 1)) ;
            % r is implicit and equal to 1
          end

          % map image coordinates to map indeces
          jmin = vl_binsearch(feat.xr, tiles(1,:)) + 1 ;
          imin = vl_binsearch(feat.yr, tiles(2,:)) + 1 ;
          jmax = vl_binsearch(feat.xr, tiles(3,:))     ;
          imax = vl_binsearch(feat.yr, tiles(4,:))     ;

          b = uint32([imin ; jmin ; imax ; jmax]) ;
          hacc = vl_samplinthist(w, b) ;

          if proj,
            macc = vl_samplinthist(m, b) ;
          else
            hacc = max(hacc, 0) ; % for small num. errors
          end

        otherwise
          assert(false, 'case unknown') ;
      end

      hists{i} = hacc ;
      if proj, mass{i} = macc ; end

    end % y
  end % x

  % Produce final aggregate and normalized histograms
  hists = double(cat(1, hists{:})) ;

  if opts.normalize
    if proj
      hists = sum(hists, 1) ;
      mass = sum(cat(1, mass{:})) + 1e-5 ;
    else
      mass = sum(hists,1) + 1e-5 ;
    end
    tmp = 1 ./ mass ;
    hists = hists .* (ones(size(hists,1),1) * tmp)  ;
  end

  levels{li} = hists ;
  clear mass hists ;
end
