function vocab = getVocab(vocabPath)
% GETVOCAB Cached load of a vocabulary
%   VOCAB = GETBOCAB(VOCABPATH)
%
%   Author:: Andrea Vedaldi

% AUTORIGHTS
% Copyright (C) 2008-09 Andrea Vedaldi
%
% This file is part of the VGG MKL Class and VGG MKL Det code packages,
% available in the terms of the GNU General Public License version 2.

vocab = cacheGet(mfilename, {}, {vocabPath}) ;
if isempty(vocab)
  vocab = load(vocabPath) ;
  vocab = vocab.V ;
  cacheStore(mfilename, vocab, {}, {vocabPath}) ;
end
