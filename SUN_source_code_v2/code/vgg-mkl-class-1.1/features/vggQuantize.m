function words = vggQuantize(vocab, descrs)
% VGGQUANTIZE  Quantize bag-of-words features
%
%  Author:: Andrea Vedaldi

if ~isa(vocab, 'double') ; vocab = double(vocab) ; end
if ~isa(descrs, 'double') ; descrs = double(descrs) ; end

words = MEXfindnearestl2(descrs, vocab);
