function [final_color_hist] = calc_color_hist_weighted(LAB_image, weight_image)
%jhhays 5/25/2009
%like calc_color_hist_masked, except it allows a smoothly varying
%weighting. 

%how to do this efficiently? we want to weight each color as we add it to
%the histogram according to the mask. we could make a histogram with an
%extra dimension based on that weight? seems like overkill.

%builds a joint color histogram with 'color_bins' color bins and 'int_bins'
%intensity bins.  Works in Lab space.

%histograms aren't normalized, but chi squared distance will take care of
%that.

%needs to have calc_textons/histograms in the path.

%image should be [0 1], color, double

% fprintf('Calculating color histogram\n')
% addpath(['calc_textons' filesep 'histogram'])

% LAB_image  = rescale_max_size(LAB_image, 400);
% weight_image = rescale_max_size(weight_image, 400);

% input_image = RGB2Lab(input_image);

int_channel = LAB_image(:,:,1);
a_channel   = LAB_image(:,:,2);
b_channel   = LAB_image(:,:,3);

int_channel = int_channel(:);
a_channel   = a_channel(:);
b_channel   = b_channel(:);

weight_channel = weight_image(:);

% int_channel = int_channel(mask == 0);
% a_channel   = a_channel(mask == 0);
% b_channel   = b_channel(mask == 0);

% a_channel = a_channel + 35;
% b_channel = b_channel + 35;
% 
% a_channel(a_channel < 0)   = 0;
% a_channel(a_channel > 70) = 70;
% b_channel(b_channel < 0)   = 0;
% b_channel(b_channel > 70) = 70;

% color_bin_w = 70 / color_bins;
% int_bin_w   = 100 / int_bins;

% int_channel = ceil(int_channel/ int_bin_w );
% a_channel   = ceil( a_channel / color_bin_w );
% b_channel   = ceil( b_channel / color_bin_w );

% whos
color_points = [a_channel b_channel int_channel weight_channel];

% min(color_points,[],1)
% max(color_points,[],1)

color_edges  = [-inf   -30   -25   -20   -15   -10    -5     0     5    10    15    20    25    30    inf];
int_edges    = [-inf 55 70 90 inf];
weight_edges = 0:(1/256):1;

%this is basically three joint color histograms for shadows, mid range, and
%highlight portions of the image.
color_hist = histnd(color_points, color_edges, color_edges, int_edges, weight_edges);
% for i = 1: size(color_hist,4)
%     sum(sum(sum(color_hist(:,:,:,i))))
% end
% pause

% sum(sum(sum(color_hist(:,:,:,101)))) %is not necessarily empty.
% sum(sum(sum(sum(color_hist))))
color_hist = color_hist(1:end-1, 1:end-1, 1:end-1, 1:end); %the last bin in each dimension is empty, except the weight dimension
% sum(sum(sum(sum(color_hist))))

%now we want to marginalize over the last dimension
final_color_hist = zeros(14, 14, 4);

for i = 2:size(color_hist,4)-1
    final_color_hist  = final_color_hist + ((i-.5)/size(color_hist,4)) * color_hist(:,:,:,i);
end
final_color_hist = final_color_hist + color_hist(:,:,:,end);

% sum(sum(sum(final_color_hist)))
% sum(sum(weight_image))
% pause

% img_area = size(input_image,1) * size(input_image,2);
% bad_area = sum( sum( 1-weight_image ) );
% valid_perc = (img_area - bad_area) / img_area;
% 
% color_hist = color_hist ./ valid_perc;

%range for first dimension, intensity, is [0 100]
%what is the range for for the color dimensions? [-50 50]? more it seems.
% figure(1)
% imagesc(color_hist(:,:,1))
% figure(2)
% imagesc(color_hist(:,:,2))
% figure(3)
% imagesc(color_hist(:,:,3))
% figure(4)
% imagesc(color_hist(:,:,4))

final_color_hist = single(final_color_hist);


