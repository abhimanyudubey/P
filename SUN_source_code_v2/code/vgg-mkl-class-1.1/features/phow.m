function feat = phow(conf, im)
% PHOW  Extract PHOW features
%
%   Author:: Andrea Vedaldi

conf_.color = false ;
conf_.sizes = [10 20] ;
conf_.step  = 5 ;
conf_.fast  = false ;
conf = override(conf_, conf) ;

im = im2double(im) ;

opts = {'windowsize', 1, 'verbose'} ;
if conf.fast, opts{end+1} = 'fast' ; end

if conf.color
  if size(im, 3) == 1, im = cat(3, im, im, im) ; end

  frames = [] ;
  descrs = [] ;
  for si = 1:length(conf.sizes)
    s = conf.sizes(si) ;
    ims = imsmooth(im, sqrt((s/6)^2 - 0)) ;
    hsv = rgb2hsv(ims) ;
    hsv = im2single(hsv) ;

    d_ = [] ;
    for k=1:3
      % This offset causes features of different scales to have the same
      % centers. While not strictly required, it makes the
      % representation much more compact.
      off = 3/2 * max(conf.sizes) + 1 - 3/2 * s ;

      [f, d] = vl_dsift(squeeze(hsv(:,:,k)), ...
                        'step', conf.step, ...
                        'size', s, ...
                        'bounds', [off off +inf +inf], ...
                        'norm', opts{:}) ;

      sel = find(f(3,:) < 0.002) ;
      d(:,sel) = 0 ;

      f = [f(1:2,:) ; 2*s * ones(1, size(f,2))] ;
      d_ = cat(1, d_, d) ;
    end
    frames = cat(2, frames, f) ;
    descrs = cat(2, descrs, d_) ;
  end

else
  if size(im, 3) == 3, im = rgb2gray(im) ; end
  im = im2single(im) ;

  frames = [] ;
  descrs = [] ;
  for si = 1:length(conf.sizes)
    s = conf.sizes(si) ;

    % This offset causes features of different scales to have the same
    % centers. While not strictly required, it makes the
    % representation much more compact.
    off = 3/2 * max(conf.sizes) + 1 - 3/2 * s ;

    ims = imsmooth(im, sqrt(max((s/6)^2 -.25,0))) ;

    [f, d] = vl_dsift(ims, ...
                      'step', conf.step, ...
                      'size', s, ...
                      'bounds', [off off +inf +inf], ...
                      'norm', opts{:}) ;

    sel = find(f(3,:) < 0.005) ;
    d(:,sel) = 0 ;

    %    figure ; hist(f(3,:),100) ;
    %drawnow ;

    f = [f(1:2,:) ; 2*s * ones(1, size(f,2))] ;

    frames = cat(2, frames, f) ;
    descrs = cat(2, descrs, d) ;
  end
end

feat = struct('frames', frames, 'descrs', descrs) ;
