function [feat boxes]= denseSIFT(conf, input_image)
%% input image process
%resize input image
input_image = rescale_max_size(input_image, 640);
%if black and white
if(size(input_image,3) < 3)
    %black and white image
    input_image = cat(3, input_image, input_image, input_image); %make it a trivial color image
end
boxes = [1;1;size(input_image,1);size(input_image,2)];

%% features
feat = phow(conf, input_image);

%% quantization
feat.words = ikmeanspush(feat.descrs,conf.textons);
feat.words = sparse2dense(feat);

