function feat = phog(featOpts, im)
% PHOG  Extract PHOG features
%   FEAT = PHOG(FEATOPTS, IM) returns the PHOG features of
%   image IM. IM is an image in the standard MATLAB format. WORDS is
%   the word map image, and WEIGHTS is the weight map.
%
%   Author:: Andrea Vedaldi

conf_.period         = 360 ;
conf_.vocabSize      = 16 ;
conf_.gradientThresh = 1 / 256 ;
conf = override(conf_, featOpts) ;

% make gray scale
im = im2double(im) ;
if size(im, 3) > 1, im = rgb2gray(im) ; end

% sanity check
if 0
  im = 0 * double(im) ;
  [v,u] = meshgrid(1:size(im,1), 1:size(im,2)) ;
  im  = (v - size(im,1)/2).^2 + (u - size(im,2)/2).^2  < 100*100 ;
  im = 256*double(im) ;
end

E = edge(im, 'canny') ;
[imx,imy] = gradient(imsmooth(double(im),.5)) ;

wgt = sqrt(imx.^2 + imy.^2) ;
period = conf.period / 180 ;
angle = mod(atan2(imy, imx) / pi, period) / period ;

vocabSize = conf.vocabSize ;
angle = angle * vocabSize ;

pre   = floor(angle) ;
postw = angle - pre ;
prew  = 1 - postw ;
post  = pre + 1 ;
pre   = mod(pre, vocabSize) + 1 ;
post  = mod(post, vocabSize) + 1 ;

% convert to sparse format for uniformity
sel = find(wgt .* E >= conf.gradientThresh) ;
[i, j] = ind2sub(size(im), sel) ;

feat.frames  = [j(:)' j(:)'; i(:)' i(:)'] ;
feat.descrs  = [pre(sel); post(sel)]' ;
feat.weights = [wgt(sel) .* E(sel) .* prew(sel) ; wgt(sel) .* E(sel) .* postw(sel)]' ;
