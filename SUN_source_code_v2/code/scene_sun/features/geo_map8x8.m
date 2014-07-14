function [feat boxes]= geo_map8x8(conf, input_image)
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

geom_cont_feature = single(down_sample(geom_c_map, 8, 8));
feat.geo_map8x8 = geom_cont_feature(:);

%feat.texton_map = texton_map;
%feat.geom_c_map = geom_c_map;
