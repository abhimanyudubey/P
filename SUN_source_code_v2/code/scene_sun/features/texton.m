function [feat boxes]= texton(conf, input_image)
%function [feat texton_map geom_c_map]= geometry_texton_histogram(input_image,conf.filterBank,conf.textons,conf.classifiers)

%resize input image
input_image = rescale_max_size(input_image, 400);
%if black and white
if(size(input_image,3) < 3)
    %black and white image
    input_image = cat(3, input_image, input_image, input_image); %make it a trivial color image
end
boxes = [1;1;size(input_image,1);size(input_image,2)];

% Texton map.
texton_input_image = rgb2gray(input_image);
[texton_map] = calc_texton_map(texton_input_image, conf.filterBank, conf.textons);
%texton_map   = uint16(texton_map);
texton_map = uint32(texton_map);

feat.words.wordMap = texton_map;
feat.words.yr = 1:size(input_image,1);
feat.words.xr = 1:size(input_image,2);
