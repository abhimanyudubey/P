function [feat boxes] = hog2x2(conf, input_image)

input_image = input_image * 255;

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
d = features(input_image,conf.interval);
xmax = size(d,1);
ymax = size(d,2);

n = 1;
for xx=1:xmax-1
	for yy=1:ymax-1
		frames(1,n) = 1+ size(input_image,1) / xmax * xx;
		frames(2,n) = 1+ size(input_image,2) / ymax * yy;
		descrs( 1: 31, n) = uint8( min(d(xx,yy,:),1) * 255);
		descrs(32: 62, n) = uint8( min(d(xx+1,yy,:),1) * 255);
		descrs(63: 93, n) = uint8( min(d(xx,yy+1,:),1) * 255);
		descrs(94:124, n) = uint8( min(d(xx+1,yy+1,:),1) * 255);
		n=n+1;
	end
end

feat.frames = frames;
feat.descrs = descrs;
feat.words = ikmeanspush(feat.descrs,conf.textons);
feat.words = sparse2dense(feat);

