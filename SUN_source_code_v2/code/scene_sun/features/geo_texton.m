function [feat boxes]= geo_texton(conf, input_image)
%function [feat texton_map geom_c_map]= geometry_texton_histogram(input_image,conf.filterBank,conf.textons,conf.classifiers)

%resize input image
input_image = rescale_max_size(input_image, 400);
%if black and white
if(size(input_image,3) < 3)
    %black and white image
    input_image = cat(3, input_image, input_image, input_image); %make it a trivial color image
end
boxes = [1;1;size(input_image,1);size(input_image,2)];

%compute geo-cont
try
    [pg, data, imsegs] = ijcvTestImage(input_image, [], conf.classifiers);
    [cimages, cnames] = pg2confidenceImages(imsegs, {pg});
    % only keep ground pourous sky vertical [1 5 7 8]
    geom_c_map = cimages{1}(:,:,[1 5 7 8]);
catch
    geom_c_map = ones([size(input_image,1) size(input_image,2) 4])*0.25;
end

%boundaries aren't trustworthy
geom_c_map(1:4, :, :) = 0;
geom_c_map(end-3:end, :, :) = 0;
geom_c_map(:, 1:4, :) = 0;
geom_c_map(:, end-3:end, :) = 0;

% Texton map.
texton_input_image = rgb2gray(input_image);
[texton_map] = calc_texton_map(texton_input_image, conf.filterBank, conf.textons);
%texton_map   = uint16(texton_map);
texton_map = double(texton_map);

%create texton histograms for each geometric class %five features
feat.hists.gnd = assign_weighted_map(texton_map, geom_c_map(:,:,1));
feat.hists.por = assign_weighted_map(texton_map, geom_c_map(:,:,2));
feat.hists.sky = assign_weighted_map(texton_map, geom_c_map(:,:,3));
feat.hists.vrt = assign_weighted_map(texton_map, geom_c_map(:,:,4));
feat.hists.all = assign_weighted_map(texton_map, ones(size(texton_map))); %global

% feat.texton_map = texton_map;
% feat.geom_c_map = geom_c_map;
