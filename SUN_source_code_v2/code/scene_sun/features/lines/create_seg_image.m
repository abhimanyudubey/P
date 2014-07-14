function [seg_image, nsegs, npixels] = create_seg_image(im_height, im_width, nrows, ncols)
%create_seg_image(8, 9, 2, 3) creates a segmentation image like so
% 1 1 1 2 2 2 3 3 3 
% 1 1 1 2 2 2 3 3 3 
% 1 1 1 2 2 2 3 3 3 
% 1 1 1 2 2 2 3 3 3 
% 4 4 4 5 5 5 6 6 6
% 4 4 4 5 5 5 6 6 6
% 4 4 4 5 5 5 6 6 6
% 4 4 4 5 5 5 6 6 6
%this can be used by APPgetLargeConneectedEdges

nsegs = nrows * ncols;
seg_image = zeros(im_height, im_width,1);
npixels = zeros(nsegs, 1);

x_step = im_width / ncols; %not necessarily integer
y_step = im_height / nrows;

for i = 0:nsegs-1
    
    x_seg = mod(i, ncols) + 1;
    y_seg = floor( i / ncols) + 1;
    
    x_range = round((x_seg - 1) * x_step + 1) : round((x_seg) * x_step);
    y_range = round((y_seg - 1) * y_step + 1) : round((y_seg) * y_step);
    
    npixels(i+1) = length(x_range) * length(y_range);
    
    seg_image(y_range, x_range) = i + 1;
end