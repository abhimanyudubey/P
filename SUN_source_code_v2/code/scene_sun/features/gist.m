function [feat boxes] = gist(conf, img)
    if size(img,3) > 1
        img = rgb2gray(img) ;
    end
    if sum(size(img) == [conf.imageSize  conf.imageSize])~=2
        img = imresize(img, [conf.imageSize conf.imageSize], 'bicubic');
    end
    boxes = [1;1;size(img,1);size(img,2)];
    
    feat.descrs = gistGabor(prefilt(img, conf.fc_prefilt), conf.numberBlocks, conf.G);


