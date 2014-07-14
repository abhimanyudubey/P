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

LAB_img  = RGB2Lab(input_image);

color_hist_gnd = calc_color_hist_weighted( LAB_img, geom_c_map(:,:,1) );
color_hist_por = calc_color_hist_weighted( LAB_img, geom_c_map(:,:,2) );
color_hist_sky = calc_color_hist_weighted( LAB_img, geom_c_map(:,:,3) );
color_hist_vrt = calc_color_hist_weighted( LAB_img, geom_c_map(:,:,4) );
color_hist     = calc_color_hist_weighted( LAB_img, ones([size(LAB_img,1), size(LAB_img,2)])); %global

feat.hists.gnd = color_hist_gnd(:);
feat.hists.por = color_hist_por(:);
feat.hists.sky = color_hist_sky(:);
feat.hists.vrt = color_hist_vrt(:);
feat.hists.all = color_hist(:);

%feat.texton_map = texton_map;
%feat.geom_c_map = geom_c_map;
