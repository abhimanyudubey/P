function gist = gistGaborPadding(img, w, G)
% 
% Input:
%   img = input image  [nrows, ncols]
%   w = number of windows (w*w)
%   G = precomputed transfer functions
%
% Output:
%   gist: w*w*Nfilters response image

vis = 0;

if ndims(img)~=2
    fprintf('Error, needs to be a grayscale image\n')
    return
end

border_pix = round(size(img) * 0.15);

img = add_reflect_border(img, border_pix);

if(vis)
    figure(997)
    o = 128+128*img/max(abs(img(:)));
    imshow(uint8(o))
end

[n n Nfilters] = size(G);

% g = zeros(W*Nfilters, 1);
gist = zeros(w, w, Nfilters);

img = single(fft2(img)); 

for n = 1:Nfilters
    ig = abs(ifft2(img.*repmat(G(:,:,n), [1 1 1])));
    ig = ig(border_pix(1) + 1:end-border_pix(1), border_pix(2) + 1:end - border_pix(2));
    if(vis)
        figure(998)
        imagesc(ig)
        pause
    end
    v = downN(ig, w);
    gist(:,:,n) = v;
end

vis = 0;
if(vis)
    for i = 1:Nfilters
        imagesc(gist(:,:,i))
        pause
    end
end

gist = gist(:);


function y=downN(x, N)
% 
% averaging over non-overlapping square image blocks
%
% Input
%   x = [nrows ncols nchanels]
% Output
%   y = [N N nchanels]

nx = fix(linspace(0,size(x,1),N+1));
ny = fix(linspace(0,size(x,2),N+1));
y  = zeros(N, N, size(x,3));
for xx=1:N
  for yy=1:N
    v=mean(mean(x(nx(xx)+1:nx(xx+1), ny(yy)+1:ny(yy+1),:),1),2);
    y(xx,yy,:)=v(:);
  end
end
