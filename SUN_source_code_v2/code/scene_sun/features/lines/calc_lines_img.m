function [lines, conf, spdata] = calc_lines_img(current_image, input_mask)
%current_image might need to be int.  But input_mask can and should stay a double.

% if(isa(input_mask, 'uint8'))
%     ;
% else
%     input_mask = uint8(input_mask * 255);
% end

% fprintf('Calculating line statistics for image\n');

if(size(current_image,3) > 1)
    grayIm = rgb2gray(current_image);
else
    grayIm = current_image;
end

if(isa(grayIm, 'uint8'))
    ;
else
    grayIm = uint8(grayIm * 255);
end

grayIm = rescale_max_size(grayIm, 800, 0);

% figure(1)
% imshow(grayIm)
% pause(1)

if(~exist('input_mask', 'var'))
    input_mask = zeros(size(grayIm));
else
    input_mask = round(rescale_max_size(input_mask, 800));
end

% save('tmp.mat', 'grayIm');
% pause

% lines = [];
% conf = [];
spdata = [];

diagonal = sqrt(size(grayIm,1)^2 + size(grayIm,2)^2);
minLen = 0.02 * diagonal;

% imsegs = struct('imsize',   [size(grayIm,1) size(grayIm,2)], ...
%                 'segimage', ones(size(grayIm,1), size(grayIm,2)), ...
%                 'nseg',     1, ...
%                 'npixels',  size(grayIm,1) * size(grayIm,2), ...
%                 'adjmat',   logical(1)); %not used?
% 
% n_seg_cols = 2;
% n_seg_rows = 2;
% 
% [imsegs.segimage, imsegs.nseg, imsegs.npixels] = create_seg_image(size(grayIm,1), size(grayIm,2), n_seg_rows, n_seg_cols);
% 
% [lines, conf, spdata] = APPgetLargeConnectedEdges(grayIm, minLen, imsegs, input_mask); 

%checks out- functionally equivalent.


%uses much faster cvCanny
% tic
[lines, conf] = APPgetLargeConnectedEdges_fast2(grayIm, minLen, input_mask);
% toc
% [lines, conf, edgeIm] = APPgetLargeConnectedEdges_fast(grayIm, minLen, input_mask);

display = 0;
if(display)
    edgeIm = zeros(size(grayIm));
    for i = 1:size(lines,1)
        edgeIm = draw_line_image2(edgeIm, lines(i, 1:4)', conf(i)/800,  i);
    end

    color_edge_im = zeros(size(grayIm,1), size(grayIm,2), 3);
    color_edge_im(:,:,1) = edgeIm;
    color_edge_im(:,:,2) = double(grayIm)/(2*255);
    color_edge_im(:,:,3) = double(grayIm)/(2*255);
    figure(7182)
    imshow(color_edge_im)
    pause
end