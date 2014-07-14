function words = ikmeansQuantize(vocab, descrs)
% BOWQUANTIZE  Quantize bag-of-words features
%
%  Author:: Andrea Vedaldi

words = ikmeanspush(descrs, vocab) ;

